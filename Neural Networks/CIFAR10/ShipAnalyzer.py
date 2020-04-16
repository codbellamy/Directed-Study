from LayeredCNN import Net
from PIL import Image
import time
from math import sqrt
from math import ceil
# import matplotlib
# import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# This script is a dependency
# To use this script, call the cnn.bat file using the image to analyze as the argument
# EX: 'cnn.bat unnamed.jpg'

# Which state to load
STATE_PATH = './states/'            # Path to states folder
NETWORK_TYPE = '3xConv_MaxPool'     # Which CNN structure to use
EPOCH = '62'                        # Which training epoch to load the state from

# Generate filename
STATE_FILE = './cifar_net_' + EPOCH

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if len(sys.argv) is not 2:
    raise IOError('Script called with incorrect number of arguments.')
else:
    IMAGE = sys.argv[1]
    if IMAGE[-3:] == 'png':
        raise Exception('Extension cannot be png.')

# Transformations and data normalizations to adjust input similar to training data
data_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Create a dictionary of activations for visualizations
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

start = time.time()

print('Reading image...')
try:
    image = Image.open(IMAGE) # Image to pass through the network
except FileNotFoundError as identifier:
    print('{0:s} not found! Did you include the extension?'.format(IMAGE))
    raise

image_t = data_transforms(image) # Perform transformations to the image
batch_t = torch.unsqueeze(image_t, 0) # Convert the matrix to a vector

print('Initializing network...')
net = Net() # Create the network, this must be the exact same structure as the trained network
net.load_state_dict(torch.load(STATE_FILE)) # Load the trained state from file
net.eval() # Set the network to evaluation mode
criterion = nn.CrossEntropyLoss() # Determine the loss function (not completely necessary)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Set up optimizer (not completely necessary)

# Register forward pass hooks for each convolutional layer in the sequence
# CONV_LAYERS = (0,3,7,10,14,17)
# for layer in CONV_LAYERS:
#     name = 'conv_layer'+str(layer)
#     net.conv_layer[layer].register_forward_hook(get_activation(name))

print('Processing image...')
out = net(batch_t) # Run the transformed image through the network
_, index = torch.max(out, 1) # Read outputs
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 # Convert outputs to percentages

# Compute processing times for the network
end = time.time()
processing_time = end-start
print('{0:2.3f}s of processing time'.format(processing_time))

# # Print the confidence in classification of a ship
# print('{0:s}:\t{1:3.3f}% Confident\n\n'.format(classes[8], percentage[8].item()))

_, indices = torch.sort(out, descending=True)
print('Top 5 Categories\n')
for classification in enumerate([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]], start=1):
    print('{0:d}:\t{1:s}\t{2:3.3f}%'.format(classification[0],classification[1][0], classification[1][1]))

# # Save images of feature maps
# print('\nProcessing convolutional feature map visualization...')
# for name in activation:
#     act = activation[name].squeeze()

#     # Create a "square" grid for matplotlib
#     size = act.size(0)
#     square = int(ceil(sqrt(size)))
#     fig, axarr = plt.subplots(square, square)

#     i = 0
#     j = 0
#     for idx in range(size):
#         axarr[i][j].axis("off")
#         try:
#             axarr[i][j].imshow(act[idx])
#         except IndexError: # If there are more images than grid spots
#             break
#         i += 1
#         if idx % square == square-1:
#             i = 0
#             j += 1
#     plt.savefig(name+".png", bbox_inches='tight') # Save image
#     plt.clf()

# # Show one example of the kernels used to perform convolutions
# print('\nProcessing kernel visualization...')
# kernels = net.conv_layer[0].weight.detach().clone()
# kernels = kernels - kernels.min()
# kernels = kernels / kernels.max()
# img = make_grid(kernels)
# plt.imshow(img.permute(1, 2, 0))
# plt.savefig("kernels.png")