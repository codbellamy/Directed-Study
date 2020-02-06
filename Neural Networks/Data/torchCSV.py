#%%

#Loads necessary (and probably some unnecessary modules)
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

#The maximum number of epochs to run
EPOCHS = 50

#%%
#Loads in the dataset as a csv file
testFile = open("./USvideos.csv", 'r')
testList = testFile.readlines()
testFile.close()

#Defaults lists of inputs and labels
trainingSet = []
testingSet  = []

#Gets the minimum and maximum value for each column (for scaling purposes)
mins  = np.asfarray(testList[1].strip().split(','))
maxes = np.asfarray(testList[2].strip().split(','))

#Loops through each line of the csv file
for datum in testList[3:]:
    
    
    #Takes the values except for the labels
    vals = np.asfarray(datum.strip().split(','))
    
    #Scales the image
    vals = (vals - mins) / ( maxes - mins )
    
    #loads in the data using a datatype that torch can deal with
    x = torch.tensor(vals[:-1], dtype = torch.float).view(-1, 4)
    y = torch.tensor(vals[-1])
    
    #1/4 chance of being put in the test set as opposed to the training set
    if random.randrange(4) <= 1:
        
        #Appends the scaled label and datum to the inputs and labels list
        testingSet.append( (x, y) )
        
    else:
        
        trainingSet.append( (x, y) )

#loads the training and test set into batches for the network
trainset = torch.utils.data.DataLoader(trainingSet, batch_size = 10000,
                                       shuffle = True)
testset  = torch.utils.data.DataLoader(testingSet,  batch_size = 10000,
                                       shuffle = True)

#%%

class Net(nn.Module):
    
    #Class constructor
    def __init__(self):
        
        #Runs nn.Module constructor
        super().__init__()
        
        #fully connected layer 1
        self.fc1 = nn.Linear(4, 10) 
                                     
        #fully connected layer 2
        self.fc2 = nn.Linear(10, 10)
        
        self.fc3 = nn.Linear(10, 1)
        
    #forward pass
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        
        return x
    
    
#Creates the network object
net = Net().float()

#loops through each epoch
for epoch in range(EPOCHS):
    optimizer = optim.Adam(net.parameters(), lr = 5e-2)
    #for each batch
    for data in trainset:
        
        #loads in an input and output batch
        x, y = data
        
        net.zero_grad() #zeros out the gradient before it updates it
        
        #calculates the output using a forward pass
        output = net(x.view(-1, 4).float())
        #calculates loss
        loss   = F.mse_loss(output, y.view(-1, 1))   
        #calculates weight adjustments using backpropagation
        loss.backward()
        #updates weights and biases
        optimizer.step()
    

    #Checks for test set
    correct = 0
    total   = 0
    
    with torch.no_grad():
        
        #loops through testing set
        for data in testset:
            x, y = data
            output = net(x.view(-1, 4).float())
            
            #loops through labels
            for idx, i in enumerate(output):
                
                if i == y[idx]:
                    correct += 1
                total += 1
                
    print('Accuracy:', round(correct/total, 5))
