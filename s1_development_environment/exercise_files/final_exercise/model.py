from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=3)
        self.ReLU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(800,10)

    def forward(self,x):
        x = self.ReLU1(self.pool1(self.conv1(x)))
        x = self.ReLU2(self.pool2(self.conv2(x)))
        return self.fc1(self.flat(x))

