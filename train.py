import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

from model import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    cnn = CNN(in_channel=1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimer = optim.Adam(cnn.parameters(),args.learn_rate)

    cnn.train()

    running_loss = 0

    correct = 0

    total = 0

    for epoch in range(args.epochs):

        for idx,(data,target) in enumerate(train_loader):

            data,target = data.to(device),target.to(device)

            output = cnn(data)

            loss = criterion(output,target)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            running_loss += loss.item()

            _,predict = output.max(1)

            total += target.size(0)

            correct += predict.eq(target).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save(cnn.state_dict(),"mnist_cnn_weights.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="mnist分类训练")
    
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--learn_rate", "-l", type=float, default=0.0005)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    
    args = parser.parse_args()
    
    train(args)