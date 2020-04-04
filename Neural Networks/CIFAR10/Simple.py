import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Simple CNN structure
# Performs a series of 2d convolutions on the 3 color channels.
# Begins with 3 input to 6 output with a 5x5 kernel.
# Max pooling by a factor of 2 to reduce the output feature maps by 1/4
# Repeat the 2d convolutions and max pooling one more time
# Further increase channels from 6 to 16
# Squash the dimensions to a 1D list
# Pass this to a FCNN (16*5*5 -> 120 -> 84 -> 10)

# After 108 epochs, overall success against the test batch was 62%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x