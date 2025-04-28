import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import CNN
import FGSM

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show():
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    cnn = CNN(in_channel=1).to(device)
    cnn.load_state_dict(torch.load("mnist_cnn_weights.pth"))
    cnn.eval()

    original_images = []
    perturbed_images = []
    gt_labels = []
    pred_labels = []
    perturbed_preds = []

    epsilon = 0.6
    former_idx = 5

    for idx, (data, target) in enumerate(test_loader):
        if idx >= former_idx:  # 仅处理前former_idx个样本
            break

        data, target = data.to(device), target.to(device)
        
        # 原始预测
        output = cnn(data)
        pred = output.argmax(dim=1)
        
        # 保存结果
        original_images.append(data.squeeze().cpu().detach().numpy())
        gt_labels.append(target.item())
        pred_labels.append(pred.item())

        # 生成对抗样本
        perturbed_x = FGSM.fgsm(cnn, data, target, epsilon)
        perturbed_output = cnn(perturbed_x)
        perturbed_pred = perturbed_output.argmax(dim=1)

        # 保存结果
        perturbed_images.append(perturbed_x.squeeze().cpu().detach().numpy())
        perturbed_preds.append(perturbed_pred.item())

    # 创建子图
    fig, axs = plt.subplots(2, former_idx, figsize=(10, 6))
    
    # 绘制原始图像（第一行）
    for i in range(former_idx):
        axs[0, i].imshow(original_images[i], cmap='gray')
        axs[0, i].set_title(f"Original\nGT: {gt_labels[i]}, Pred: {pred_labels[i]}")
        axs[0, i].axis('off')

    # 绘制对抗样本（第二行）
    for i in range(former_idx):
        axs[1, i].imshow(perturbed_images[i], cmap='gray')
        axs[1, i].set_title(f"Perturbed\nGT: {gt_labels[i]}, Pred: {perturbed_preds[i]}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

if __name__ == "__main__":
    show()