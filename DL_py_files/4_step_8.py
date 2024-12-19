import torch
import torch.nn as nn

def create_simple_conv_cifar():
    class FirstModel(nn.Module):
        def __init__(self):
            super(FirstModel, self).__init__()

            # Сверточные слои
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 32 x 32 x 16
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)  # 16 x 16 x 16

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 16 x 16 x 32
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)  # 8 x 8 x 32

            # Полносвязные слои
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(8 * 8 * 32, 1024)
            self.relu_fc1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 128)
            self.relu_fc2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            # Сверточные слои
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)

            # Полносвязные слои
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu_fc1(x)
            x = self.fc2(x)
            x = self.relu_fc2(x)
            x = self.fc3(x)

            return x

    return FirstModel()
