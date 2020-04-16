#Loads necessary (and probably some unnecessary modules)
from glob import glob
import numpy as np
import os, cv2, itertools
import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Adjustable parameters
TYPE = 'Derm9'
LAST_STATE = 0                                 # Last network state that was saved
TRAINING_EPOCHS = 100                            # Number of epochs of training
LOAD_FLAG = False                                 # Load from a previous state
TRAINING_FLAG = True                            # Whether or not to train the network, must not be false if ASSESSMENT_MODE is 0 or 1
ASSESSMENT_MODE = 1                             # 0-None, 1-During Training, 2-Quick Assessment

# Non-adjustable parameters
PATH = './states/'+TYPE+'/cifar_net_'            # Path and name of NN save states
STATE_NAME = './states/'\
    +TYPE+'/cifar_net_'+str(LAST_STATE)         # Name of state to load

TRAIN_DATA_PATH = 'derm/Train'
TEST_DATA_PATH = 'derm/Test'
ALL_DATA_PATH = glob(os.path.join('derm', '*', '*.jpg'))
classes = ('actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion')

def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 512, 512
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        try:
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (img_h, img_w))
        except Exception as e:
            print(str(e))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

compute_img_mean_std(ALL_DATA_PATH)

transform_train = transforms.Compose([
    transforms.Resize((600,600)),
    transforms.RandomCrop(512, padding=5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

# Create the CNN and Linear NN structure
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )


        self.fc_layer = nn.Sequential(
            nn.Linear(41472, 15488),
            nn.ReLU(inplace=True),
            nn.Linear(15488, 3000),
            nn.ReLU(inplace=True),
            nn.Linear(3000, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 9)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

# NN overall assessment
def assessment(epoch, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

def classAssessment(net):
    # Load test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # NN class assessment
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def training(epoch_start, epoch_end, net, criterion, optimizer):
    for epoch in range(epoch_start, epoch_end):  # loop over the dataset multiple times

        start_time = time.time()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[{0:d}, {1:5d}] loss: {2:.3f}\t{3:.2f} hrs'\
                    .format(epoch + 1, i + 1, running_loss / 100, (time.time()-start_time)/(60*60)))
                running_loss = 0.0
        
        # Assess state
        if(ASSESSMENT_MODE == 1):
            assessment(epoch+1, net)
        torch.save(net.state_dict(), PATH+str(epoch+1))

    print('Finished Training')


def main():
    # Create the network
    net = Net()
    if(LOAD_FLAG):
        net.load_state_dict(torch.load(STATE_NAME))

    # Optimize and define parameters of NN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=.9)

    if(TRAINING_FLAG and ASSESSMENT_MODE is not 2):
        net.train(True)
        training(LAST_STATE, TRAINING_EPOCHS, net, criterion, optimizer)
    elif(ASSESSMENT_MODE == 2):
        assessment(0, net)
        classAssessment(net)
    else:
        print('Error: Incorrect parameters for TRAINING_FLAG and ASSESSMENT_MODE')

def run():
    torch.multiprocessing.freeze_support()
    main()
    print('loop')
if __name__ == '__main__':
    run()