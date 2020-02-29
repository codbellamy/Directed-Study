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

#Eta is a greek letter. This is the learning rate constant
eta    = 1e-3

#Number of rows to train with
batch  = 100

#Structure for 3 inputs, 2 hidden layers, and one output
#DO NOT RELY ON THIS FOR A SIMPLE TWEAK TO THE LAYERS
structure = [3,1]
columns_in = structure[0]
columns_out = structure[-1]

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
        #Appends the scaled label and datum to the inputs and labels list for training
        trainingSet.append( (x, y) )

#loads the training and test set into batches for the network
trainset = torch.utils.data.DataLoader(trainingSet, batch_size = batch,
                                       shuffle = True)
testset  = torch.utils.data.DataLoader(testingSet,
                                       shuffle = True)

#%%

class Net(nn.Module):
    
    #Class constructor
    def __init__(self, structure):
        
        #Runs nn.Module constructor
        super(Net, self).__init__()

        self.fc1 = nn.Linear(columns_in,5)
        self.fc2 = nn.Linear(5,3)
        self.fc3 = nn.Linear(3,columns_out)
        
    #feed-forward pass
    def forward(self, x):
        
        #relu is an activation function that we are choosing to replace sigmoid for these layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #sigmoid to make a probability of being true or false reasonable for our output
        x = torch.sigmoid(self.fc3(x))

        return x
    
    
#Creates the network object
net = Net(structure).float()

#Optimizes our network, gotta go fast!
optimizer = optim.Adam(net.parameters(), lr = eta)

#loops through each epoch
for epoch in range(EPOCHS):
    #for each batch
        
    # Set NN to training mode
    net.train()

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

        #Set NN to evaluation mode
        net.eval()
        
        #loops through testing set
        for data in testset:
            x, y = data
            output = net(x.view(-1, columns_in).float())
            
            #loops through labels
            for idx, i in enumerate(output):
                
                if (i >= .5) == y[idx]:
                    correct += 1
                total += 1
                
    print('Epoch: %d \tAccuracy: %.2f%%' % (epoch+1, round((correct/total)*100, 2)))

print('----------Weights and Biases----------')
for p in net.named_parameters():
    print(p)