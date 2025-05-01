import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import CNN, substitute_CNN  # 确保导入替代模型
import PGD

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def generate_attack_results(num_samples=5, epsilon=0.3, T=50, alpha=0.01):
    # 加载目标模型
    target_model = CNN(in_channel=1).to(device)
    target_model.load_state_dict(torch.load("mnist_cnn_weights.pth"))
    target_model.eval()
    
    # 加载替代模型
    substitute_model = substitute_CNN().to(device)
    substitute_model.load_state_dict(torch.load("trained_substitute.pth"))  # 使用训练好的替代模型
    substitute_model.eval()
    
    # 加载测试数据
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # 结果存储
    results = {
        'original_images': [],
        'perturbed_images': [],
        'gt_labels': [],
        'target_preds': [],
        'substitute_preds': [],
        'adv_target_preds': [],
        'adv_substitute_preds': []
    }
    
    # 攻击统计
    attack_success = 0
    total = 0
    
    for idx, (data, target) in enumerate(test_loader):
        if idx >= num_samples:
            break
            
        data, target = data.to(device), target.to(device)
        
        # 原始预测
        with torch.no_grad():
            target_pred = target_model(data).argmax(dim=1)
            substitute_pred = substitute_model(data).argmax(dim=1)
        
        # 使用替代模型生成对抗样本

        perturbed_data = PGD.pgd(substitute_model,data,target,epsilon,alpha,T)

        # 对抗样本预测
        with torch.no_grad():
            adv_target_pred = target_model(perturbed_data).argmax(dim=1)
            adv_substitute_pred = substitute_model(perturbed_data).argmax(dim=1)
        
        # 统计攻击成功率（目标模型预测变化即为成功）
        if target_pred.item() != adv_target_pred.item():
            attack_success += 1
        total += 1
        
        # 保存结果
        results['original_images'].append(data.squeeze().cpu().numpy())
        results['perturbed_images'].append(perturbed_data.squeeze().cpu().numpy())
        results['gt_labels'].append(target.item())
        results['target_preds'].append(target_pred.item())
        results['substitute_preds'].append(substitute_pred.item())
        results['adv_target_preds'].append(adv_target_pred.item())
        results['adv_substitute_preds'].append(adv_substitute_pred.item())
    
    # 打印统计信息
    print(f"\nAttack Success Rate: {100*attack_success/total:.1f}%")
    print("Target Model Accuracy:")
    print(f" - Original: {100*sum(np.array(results['target_preds']) == np.array(results['gt_labels']))/num_samples:.1f}%")
    print(f" - Adversarial: {100*sum(np.array(results['adv_target_preds']) == np.array(results['gt_labels']))/num_samples:.1f}%")
    
    return results

def visualize_results(results):
    num_samples = len(results['gt_labels'])
    
    # 创建可视化面板
    fig, axs = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # 绘制原始样本
    for i in range(num_samples):
        axs[0,i].imshow(results['original_images'][i], cmap='gray')
        axs[0,i].set_title(
            f"Original\nGT: {results['gt_labels'][i]}\n"
            f"T_pred: {results['target_preds'][i]}\n"
            f"S_pred: {results['substitute_preds'][i]}"
        )
        axs[0,i].axis('off')
    
    # 绘制对抗样本
    for i in range(num_samples):
        axs[1,i].imshow(results['perturbed_images'][i], cmap='gray')
        axs[1,i].set_title(
            f"Adversarial\nGT: {results['gt_labels'][i]}\n"
            f"T_pred: {results['adv_target_preds'][i]}\n"
            f"S_pred: {results['adv_substitute_preds'][i]}"
        )
        axs[1,i].axis('off')
    
    plt.tight_layout()
    plt.savefig("attack_results.png")
    plt.show()

if __name__ == "__main__":
    # 参数设置
    attack_params = {
        'num_samples': 5,    # 展示样本数量
        'epsilon': 0.7,      # 扰动强度
        'T': 10,             # PGD迭代次数
        'alpha': 0.07       # 单步扰动步长
    }
    
    # 生成并可视化结果
    attack_results = generate_attack_results(**attack_params)
    visualize_results(attack_results)