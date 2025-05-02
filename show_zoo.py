import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import CNN
from torch.utils.data import DataLoader

class BlackBoxL2Attack:
    def __init__(self, model, targeted=True, confidence=20, lr=0.1, 
                 max_iter=500, binary_search_steps=9, initial_const=0.01):
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.lr = lr
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const  # 修正变量名错误
        
    def attack(self, image, target_class):
        """
        image: 原始输入图像 [C,H,W], 值域[0,1]
        target_class: 整数，希望模型误判的目标类别（0-9）
        """
        device = image.device
        original_image = image.clone().detach()
        c, h, w = original_image.shape
        
        # 二进制搜索参数
        const = self.initial_const
        lower_bound = 0.0
        upper_bound = 1e10
        
        best_adv = None
        best_l2 = 1e10
        
        # 创建目标掩码（关键修改点）
        target_mask = torch.zeros(10, device=device)
        target_mask[target_class] = 1.0

        for search_step in range(self.binary_search_steps):
            # 转换到tanh空间（更稳定的实现）
            image_tanh = torch.atanh((original_image * 1.9999 - 0.9999).clamp(-0.9999, 0.9999))
            delta = torch.zeros_like(image_tanh, requires_grad=True)
            optimizer = torch.optim.Adam([delta], lr=self.lr)

            previous_loss = float('inf')
            
            for step in range(self.max_iter):
                optimizer.zero_grad()
                
                # 生成对抗样本（保持数值稳定性）
                adv_tanh = image_tanh + delta
                adv_image = (torch.tanh(adv_tanh) + 1) / 2
                adv_image = torch.clamp(adv_image, 0.0, 1.0)  # 确保在[0,1]范围内
                
                # 模型预测
                outputs = self.model(adv_image.unsqueeze(0))[0]  # 输入需要保持batch维度
                
                # 计算目标类和其他类的分数（关键修正）
                target_score = outputs[target_class]
                
                # 正确排除目标类：将目标类分数设为负无穷
                masked_outputs = outputs - 10000 * target_mask
                max_other = torch.max(masked_outputs)
                
                # 分类损失计算
                if self.targeted:
                    # 目标攻击：使目标类分数 > 其他类最大分数 + confidence
                    loss_cls = torch.max(torch.tensor(0.0), (max_other - target_score) + self.confidence)
                else:
                    # 非目标攻击：使其他类最大分数 > 原始类分数 + confidence
                    loss_cls = torch.max(torch.tensor(0.0), (target_score - max_other) + self.confidence)
                
                # L2距离计算
                l2_dist = torch.norm(adv_image - original_image)
                
                # 总损失
                total_loss = const * loss_cls + l2_dist
                
                # 反向传播（添加梯度裁剪）
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_([delta], 0.1)  # 防止梯度爆炸
                optimizer.step()
                
                # 早停检查（每50步检查损失变化）
                if step % 50 == 0 and step > 0:
                    current_loss = total_loss.item()
                    if step > 100 and current_loss > 0.999 * previous_loss:
                        print(f"Early stopping at step {step}, loss={current_loss:.4f}")
                        break
                    previous_loss = current_loss
                
                # 记录最佳结果（满足攻击成功条件）
                if loss_cls <= 1e-6 and l2_dist < best_l2:
                    best_l2 = l2_dist.item()
                    best_adv = adv_image.detach().clone()
            
            # 调整正则化系数（动态平衡分类和距离损失）
            if best_adv is not None:
                upper_bound = const
                const = (lower_bound + upper_bound) / 2
                print(f"Search step {search_step}: Found valid attack, new const={const:.4f}")
            else:
                lower_bound = const
                const = const * 10 if upper_bound > 1e9 else (lower_bound + upper_bound) / 2
                print(f"Search step {search_step}: No valid attack, new const={const:.4f}")
        
        return best_adv if best_adv is not None else original_image

def show():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型加载
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load("mnist_cnn_weights.pth"))
    cnn.eval()

    # 数据加载（保持原始归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = DataLoader(
        datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=1, shuffle=True
    )

    # 攻击参数配置
    attack = BlackBoxL2Attack(
        model=cnn,
        targeted=True,
        confidence=1,
        lr=0.001,
        max_iter=10000,
        binary_search_steps=5,
        initial_const=1
    )
    
    fake_target = 6  # 目标类别
    num_samples = 6  # 展示样本数

    original_images, perturbed_images = [], []
    gt_labels, pred_labels, perturbed_preds = [], [], []

    for idx, (data, target) in enumerate(test_loader):
        if idx >= num_samples:
            break
            
        data, target = data.to(device), target.to(device)
        
        # 原始预测
        with torch.no_grad():
            output = cnn(data)
            pred = output.argmax(dim=1)
        
        # 跳过原始预测为目标的样本
        if pred.item() == fake_target:
            continue
            
        # 反归一化到原始像素空间 [0,1]
        data_denorm = data * 0.3081 + 0.1307
        data_denorm = data_denorm.clamp(0, 1)
        
        # 生成对抗样本（输入输出都保持原始像素空间）
        perturbed_denorm = attack.attack(
            data_denorm.squeeze(0),  # 输入需要是[C,H,W]
            target_class=torch.tensor(fake_target, device=device)
        ).unsqueeze(0)
        
        # 重新归一化后输入模型
        perturbed_norm = (perturbed_denorm - 0.1307) / 0.3081
        with torch.no_grad():
            perturbed_output = cnn(perturbed_norm)
            perturbed_pred = perturbed_output.argmax(dim=1)
        
        # 保存结果（原始像素空间）
        original_image = data_denorm.squeeze().cpu().numpy()
        perturbed_image = perturbed_denorm.squeeze().cpu().numpy()
        
        original_images.append(original_image)
        perturbed_images.append(perturbed_image)
        gt_labels.append(target.item())
        pred_labels.append(pred.item())
        perturbed_preds.append(perturbed_pred.item())

    # 可视化
    fig, axs = plt.subplots(2, len(original_images), figsize=(12, 6))
    for i in range(len(original_images)):
        axs[0, i].imshow(original_images[i], cmap='gray')
        axs[0, i].set_title(f"Original\nTrue: {gt_labels[i]}\nPred: {pred_labels[i]}", fontsize=8)
        axs[0, i].axis('off')
        
        axs[1, i].imshow(perturbed_images[i], cmap='gray')
        axs[1, i].set_title(f"Adversarial\nTrue: {gt_labels[i]}\nPred: {perturbed_preds[i]}", fontsize=8)
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("adversarial_examples1.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    show()