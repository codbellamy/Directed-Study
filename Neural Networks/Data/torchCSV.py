#%%

#Loads necessary (and probably some unnecessary modules)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

#The maximum number of epochs to run
EPOCHS = 50
eta    = 1e-2
batch  = 100
columns_in = 13
columns_out = 1

#%%
#Loads in the dataset as a csv file
testFile = open("./heart.csv", 'r')
testList = testFile.readlines()
testFile.close()

#Defaults lists of inputs and labels
trainingSet = []
testingSet  = []

#Gets the minimum and maximum value for each column (for scaling purposes)
mins  = np.asfarray(testList[1].strip().split(','))
maxes = np.asfarray(testList[2].strip().split(','))

#Loops through each line of the csv file
for datum in testList[columns_in:]:
    
    
    #Takes the values except for the labels
    vals = np.asfarray(datum.strip().split(','))
    
    #Scales the image
    vals = (vals - mins) / ( maxes - mins )
    
    #loads in the data using a datatype that torch can deal with
    x = torch.tensor(vals[:-columns_out], dtype = torch.float).view(-1, columns_in)
    y = torch.tensor(vals[-columns_out])
    
    #1/4 chance of being put in the test set as opposed to the training set
    if random.randrange(4) == 0:
        
        #Appends the scaled label and datum to the inputs and labels list
        testingSet.append( (x, y) )
        
    else:
        
        trainingSet.append( (x, y) )

#loads the training and test set into batches for the network
trainset = torch.utils.data.DataLoader(trainingSet, batch_size = batch,
                                       shuffle = True)
testset  = torch.utils.data.DataLoader(testingSet,
                                       shuffle = True)

#%%

class Net(nn.Module):
    
    #Class constructor
    def __init__(self):
        
        #Runs nn.Module constructor
        super().__init__()
        
        #fully connected layer 1
        self.fc1 = nn.Linear(columns_in, 50) 
        self.fc2 = nn.Linear(50, 50) 
              
#        #fully connected layer 2
        self.fc3 = nn.Linear(50, columns_out)
        
    #forward pass
    def forward(self, x):
        
        #relu layers to make back propagation feasible in deep networks
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #sigmoid to make a probability of being true or false reasonable
        x = torch.sigmoid(self.fc3(x))

        
        return x
    
    
#Creates the network object
net = Net().float()
optimizer = optim.Adam(net.parameters(), lr = eta)

#loops through each epoch
for epoch in range(EPOCHS):
    #for each batch
    for data in trainset:
        
        #loads in an input and output batch
        x, y = data
        
        net.zero_grad() #zeros out the gradient before it updates it
        
        #calculates the output using a forward pass
        output = net(x.view(-1, columns_in).float())
        #calculates loss
        loss   = F.mse_loss(output, y.view(-1, columns_out))   
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
            output = net(x.view(-1, columns_in).float())
            
            #loops through labels
            for idx, i in enumerate(output):
                
                if (i >= 0.5) == y[idx]:
                    correct += 1
                total += 1
                
    print('Accuracy:', round(correct/total, 15))
