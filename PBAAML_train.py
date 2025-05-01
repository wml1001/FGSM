import os
import torch
import pandas as pd
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类修正（原代码存在self.images未定义问题）
class PBADataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])  # 过滤非图片文件
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")  # 转换为灰度图
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # 标签占位符，实际标签将通过目标模型获取

# 模型定义（假设已导入模型结构）
from model import substitute_CNN, CNN

class PBATrainer:
    def __init__(self, args):
        # 初始化目标模型（假设已预训练）
        self.target_model = CNN().to(device)
        self.target_model.load_state_dict(torch.load("mnist_cnn_weights.pth"))
        self.target_model.eval()
        
        # 初始化替代模型
        self.substitute_model = substitute_CNN().to(device)
        
        # 训练参数
        self.lambda_ = args.lambda_       # 扰动步长
        self.rho = args.rho               # 增强迭代次数
        self.batch_size = args.batch_size
        self.epochs_per_rho = args.epochs_per_rho
        self.lr = args.lr
        
        # 数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def get_target_labels(self, dataloader):
        """通过目标模型获取所有数据的真实标签"""
        all_labels = []
        with torch.no_grad():
            for images, _ in dataloader:  # 忽略原始标签
                images = images.to(device)
                outputs = self.target_model(images)
                labels = outputs.argmax(dim=1)
                all_labels.append(labels.cpu())
        return torch.cat(all_labels)
    
    def jacobian_augment(self, images, labels):
        """执行Jacobian数据增强"""
        images.requires_grad = True
        
        # 计算替代模型在目标标签方向的梯度
        outputs = self.substitute_model(images)
        selected = outputs[torch.arange(len(labels)), labels]
        selected.sum().backward()
        
        # 生成扰动样本
        perturbations = self.lambda_ * images.grad.data.sign()
        x_new = torch.clamp(images + perturbations, 0, 1).detach()
        
        # 获取新标签
        with torch.no_grad():
            y_new = self.target_model(x_new).argmax(dim=1)
            
        return x_new.cpu(), y_new.cpu()
    
    def train(self):
        # 初始化基础数据集
        train_dataset = PBADataset("./subset", self.transform)
        test_dataset = PBADataset("./subset_test", self.transform)
        
        # 获取目标模型标签（覆盖原始标签）
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 获取目标模型生成的标签
        train_labels = self.get_target_labels(train_loader)
        test_labels = self.get_target_labels(test_loader)
        
        # 转换为TensorDataset
        train_data = torch.utils.data.TensorDataset(
            torch.stack([img for img, _ in train_dataset]), 
            train_labels
        )
        test_data = torch.utils.data.TensorDataset(
            torch.stack([img for img, _ in test_dataset]), 
            test_labels
        )
        
        # 主训练循环
        optimizer = torch.optim.Adam(self.substitute_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for rho_step in range(self.rho):
            print(f"=== Augmentation Iteration {rho_step+1}/{self.rho} ===")
            
            # 训练阶段
            self.substitute_model.train()
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            
            for epoch in range(self.epochs_per_rho):
                total_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.substitute_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                print(f"Epoch {epoch+1}/{self.epochs_per_rho} | "
                      f"Loss: {total_loss/len(train_loader):.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")
            
            # 数据增强阶段
            augmented_images, augmented_labels = [], []
            augment_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
            
            for images, labels in augment_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 生成增强数据
                x_new, y_new = self.jacobian_augment(images, labels)
                
                augmented_images.append(x_new)
                augmented_labels.append(y_new)
            
            # 合并数据集
            augmented_images = torch.cat(augmented_images)
            augmented_labels = torch.cat(augmented_labels)
            train_data = torch.utils.data.ConcatDataset([
                train_data,
                torch.utils.data.TensorDataset(augmented_images, augmented_labels)
            ])
            
            # 评估当前模型
            self.evaluate(test_data)
            
        # 保存最终模型
        torch.save(self.substitute_model.state_dict(), "trained_substitute.pth")
    
    def evaluate(self, test_data):
        self.substitute_model.eval()
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.substitute_model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f"Test Accuracy: {100.*correct/total:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", type=float, default=0.1, dest="lambda_")
    parser.add_argument("--rho", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs_per_rho", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()
    
    trainer = PBATrainer(args)
    trainer.train()