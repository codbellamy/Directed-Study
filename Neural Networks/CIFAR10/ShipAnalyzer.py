from LayeredCNN import Net
from PIL import Image
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Image
IMAGE = './unnamed.jpg'

# Which state to load
STATE_PATH = './states/'            # Path to states folder
NETWORK_TYPE = '3xConv_MaxPool'     # Which CNN structure to use
EPOCH = '26'                        # Which training epoch to load the state from

# Generate filename
STATE_FILE = STATE_PATH + NETWORK_TYPE + '/cifar_net_' + EPOCH

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

start = time.time()

print('Reading image...')
image = Image.open(IMAGE)
image_t = data_transforms(image)
batch_t = torch.unsqueeze(image_t, 0)

print('Initializing network...')
net = Net()
net.load_state_dict(torch.load(STATE_FILE))
net.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Processing image...')
out = net(batch_t)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

end = time.time()
processing_time = end-start
print('{0:2.3f}s of processing time'.format(processing_time))
print('{0:s}:\t{1:3.3f}% Confident\n\n'.format(classes[8], percentage[8].item()))

_, indices = torch.sort(out, descending=True)
print('Top 5 Categories\n')
for classification in enumerate([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]], start=1):
    print('{0:d}:\t{1:s}\t{2:3.3f}%'.format(classification[0],classification[1][0], classification[1][1]))