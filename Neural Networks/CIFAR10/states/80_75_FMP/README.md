Overall best results: 64%

Ship success rate: 85%

Structure
```
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.fracpool1 = nn.FractionalMaxPool2d(3, output_ratio=.8)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fracpool2 = nn.FractionalMaxPool2d(2, output_ratio=.75)
        self.fc1 = nn.Linear(16 * 13 * 13, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.fracpool1(x)
        x = F.relu(self.conv2(x))
        x = self.fracpool2(x)
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```