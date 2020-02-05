#%%

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

EPOCHS = 10
#%%
#Gets MNIST training set
train = datasets.MNIST("", train=True, download=True, 
                       transform = transforms.Compose([transforms.ToTensor()]))

#Gets MNIST testing set
test  = datasets.MNIST("", train = False, download = True,
                       transform = transforms.Compose([transforms.ToTensor()]))

#sets training and test set
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset  = torch.utils.data.DataLoader(test,  batch_size = 10, shuffle = True)
#%%
#Loads in the test set
testFile = open("./USvideos.csv", 'r')
testList = testFile.readlines()
testFile.close()

#Defaults lists of inputs and labels
trainingSet = []
testingSet  = []

mins  = np.asfarray(testList[1].strip().split(','))
maxes = np.asfarray(testList[2].strip().split(','))

#Loops through each line of the csv file
for datum in testList[3:]:
    
    
    #Takes the values except for the labels
    vals = np.asfarray(datum.strip().split(','))
    
    #Scales the image
    vals = (vals - mins) / ( maxes - mins )
    
    x = torch.tensor(vals[:-1], dtype = torch.float).view(-1, 4)
    y = torch.tensor(vals[-1])
    
    if random.randrange(4) == 0:
        
        #Appends the scaled label and datum to the inputs and labels list
        testingSet.append( (x, y) )
        
    else:
        
        trainingSet.append( (x, y) )

#sets training and test set
trainset = torch.utils.data.DataLoader(trainingSet, batch_size = 100,
                                       shuffle = True)
testset  = torch.utils.data.DataLoader(testingSet,  batch_size = 100,
                                       shuffle = True)

#%%

class Net(nn.Module):
    
    #Class constructor
    def __init__(self):
        
        #Runs nn.Module constructor
        super().__init__()
        
        #fully connected layer 1
        self.fc1 = nn.Linear(4, 10) #784 -> size of training set
                                     #64 -> the number of outputs for the layer  
                                     
        #fully connected layer 2
        self.fc2 = nn.Linear(10, 5) #takes in the 64 outputs and ouputs 
                                     #64 unique values
        
        #fully connected layer 3
        self.fc3 = nn.Linear(5, 3) #same as above
        
        #fully connected layer 4
        self.fc4 = nn.Linear(3, 1) #takes in 64, outputs 10
        
    def forward(self, x):
        
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        
        return F.sigmoid(x)
    
    
net = Net()
optimizer = optim.Adam(net.parameters(), lr = 100)
#loops through each epoch
for epoch in range(EPOCHS):
    
    #for each batch
    for data in trainset:
        
        #loads in an input and output batch
        x, y = data
        
        net.zero_grad() #zeros out the gradient before it updates it
        
        output = net(x)
        loss   = F.mse_loss(output, y.view(-1, 1).float())    
        loss.backward()
        optimizer.step()
    

    #Checks for test set
    correct = 0
    total   = 0
    
    with torch.no_grad():
        
        for data in testset:
            x, y = data
            output = net(x.float())
            
            for idx, i in enumerate(output):
                
                correct += float(i) == y[idx]
                total += 1
                
    print('Accuracy:', round(float(correct)/total, 5))

#%%

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(-1, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        if self.to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            return x
    def forward(self, x):
        
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        