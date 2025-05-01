import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,in_channel,hidden_channel=8,kernel_size=3,categories=10):
        super(CNN,self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel,hidden_channel,kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(hidden_channel,hidden_channel*8,kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(hidden_channel*8,hidden_channel,kernel_size),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channel*3*3,hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel,categories)
        )

    def forward(self,x):

        x = self.conv_layers(x)

        return self.fc_layers(x.view(x.size(0),-1))

class substitute_CNN(nn.Module):
    def __init__(self,in_channel=1,hidden_channel=8,kernel_size=4,categories=10):
        super(substitute_CNN,self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel,hidden_channel*8,kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(hidden_channel*8,hidden_channel,kernel_size),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channel*9*9,hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel,categories)
        )

    def forward(self,x):

        x = self.conv_layers(x)

        return self.fc_layers(x.view(x.size(0),-1))
