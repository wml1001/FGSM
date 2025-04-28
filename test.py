import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

from model import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(args):
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    cnn = CNN(in_channel=1).to(device)

    cnn.load_state_dict(torch.load("mnist_cnn_weights.pth"))

    cnn.eval()
    
    correct = 0

    test_loss = 0

    with torch.no_grad():
        for _,(data,target) in enumerate(test_loader):

            data,target = data.to(device),target.to(device)

            output = cnn(data)

            pred = output.argmax(dim=1,keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="mnist分类测试")
    
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    
    args = parser.parse_args()
    
    test(args)