###############################################################################
#                                                                             #
#                               creditcards.py                                #
#                               J. Steiner                                    #
#                               C. Bellamy                                    #
#                                                                             #  
###############################################################################

#%%################################ IMPORTS ###################################

#Imports matrix and vector operations
import numpy as np

#Imports the sigmoid function
import scipy.special as sp

#Imports randomness algorithms
import random

#Imports the Network3 class
from Network3 import Network3

#%%############################### CONSTANTS ##################################

#The size of each layer in the network
NETWORK_SIZE = [29, 20, 1]

#A constant, the learning rate
ETA = 1000

#The number of epochs to train for
MAX_EPOCHS = 5

#The size of each batch for backpropagation
BATCH_SIZE = 10000

#Loads in sigmoid function from the scipy library
sigmoid = lambda x: sp.expit(x)

#%%############################## LOAD DATA ###################################

#Formatting
print('---------------- LOADING DATA ----------------')

#Creates a Network3 object
net3 = Network3(NETWORK_SIZE, ETA, None)

#Loads in the dataset
trainFile = open("./creditcard.csv", 'r')
trainList = trainFile.readlines()
trainFile.close()

#Defaults lists of inputs and labels
trainingSet = []
testingSet  = []

maxes = np.asfarray(trainList[1].split(',')[:-1])
mins  = np.asfarray(trainList[2].split(',')[:-1])

#Loops through each line of the csv file
for datum in trainList[3:]:
    
    #Takes all the values in the row and splits them by the comma
    rowVals = datum.split(',')
    
    #Takes the values except for the labels
    vals = (np.asfarray(rowVals[:-1]) - mins) / ( maxes + mins )
    
    #Scales the image
    x = vals[1:]
    
    #Takes the label of the data and scales it
    y = int(rowVals[-1].strip())
    
    
    if random.randrange(4) == 0:
        
        testingSet.append( (x, y) )
        
    else:
        
        trainingSet.append( (x, y) )
    
    
#%%############################### EXECUTION ##################################

#Whether or not the network got a prediction correct
predicted = net3.test(testingSet)

#The array of whether or not the prediction was correct
correct = np.asarray(predicted)

#Prints the success rate
print ('Success Rate = ' + str((correct.sum() / correct.size) * 100) + '%')
    
#Formatting
print('---------------- START TRAINING --------------')
    
#Trains the neural network
net3.train(trainingSet, MAX_EPOCHS, BATCH_SIZE,
           verbose = True, testingSet = testingSet, 
           convolutionVectors=None)

#Formatting
print('---------------- DONE TRAINING ----------------')

#Whether or not the network got a prediction correct
predicted = net3.test(testingSet)

#The array of whether or not the prediction was correct
correct = np.asarray(predicted)

#Prints the success rate
print ('Success Rate = ' + str((correct.sum() / correct.size) * 100) + '%')