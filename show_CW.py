import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import CNN
from torch.utils.data import DataLoader

import CW

def show():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型加载
    cnn = CNN(in_channel=1).to(device)
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

    # 攻击参数
    cw = CW.CW(cnn, c=1,k=0, lr=0.001, T=10000)
    fake_target = 6
    num_samples = 6

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
        
        # 生成对抗样本（原始像素空间）
        perturbed_denorm = cw.generate(data_denorm, fake_target)
        
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
    fig, axs = plt.subplots(2, len(original_images), figsize=(10, 6))
    for i in range(len(original_images)):
        axs[0, i].imshow(original_images[i], cmap='gray')
        axs[0, i].set_title(f"Original\nGT: {gt_labels[i]}, Pred: {pred_labels[i]}")
        axs[0, i].axis('off')
        
        axs[1, i].imshow(perturbed_images[i], cmap='gray')
        axs[1, i].set_title(f"Perturbed\nGT: {gt_labels[i]},Pred: {perturbed_preds[i]}")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

if __name__ == "__main__":
    show()