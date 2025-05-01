import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image

from model import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataLoad():
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)

    num_recorder = [0] * 10
    image_x_recorder = []
    image_y_recorder = []

    cnn = CNN(in_channel=1).to(device)

    cnn.load_state_dict(torch.load("mnist_cnn_weights.pth"))

    cnn.eval()

    # 收集每个类别的前100个样本
    for x, y in dataLoader:
        y_item = y.item()  # 将张量转换为整数
        if num_recorder[y_item] < 100:
            num_recorder[y_item] += 1
            image_x_recorder.append(x)
            
            with torch.no_grad():
                output = cnn(x)
            
            pred = output.argmax(dim=1,keepdim=True)
            image_y_recorder.append(pred.item())
        # 当所有类别都收集满时提前退出
        if all(count >= 100 for count in num_recorder):
            break

    # 创建保存目录
    if not os.path.exists("./subset"):
        os.makedirs("./subset")

    # 保存标签到Excel
    df = pd.DataFrame({'label': image_y_recorder})
    df.to_excel("./sub_label.xlsx", index=False)

    # 保存图像并添加噪声
    for idx, x in enumerate(image_x_recorder):
        # 添加噪声并限制数值范围
        x_noised = x + torch.randn_like(x) * 0.2  # 添加适度噪声
        x_noised = torch.clamp(x_noised, 0.0, 1.0)  # 确保值在[0,1]范围内
        
        # 转换张量为图像格式
        img_tensor = x_noised.squeeze()  # 移除批次和通道维度 [28, 28]
        img_array = (img_tensor * 255).byte().cpu().numpy()  # 转换为0-255的numpy数组
        image = Image.fromarray(img_array, mode='L')
        image.save(f"./subset/{idx+1}.png")

if __name__ == "__main__":
    dataLoad()