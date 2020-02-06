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

#Import random test set generator
from TestSample import generateRandomTestSample as generate

#%%############################### CONSTANTS ##################################

#The size of each layer in the network
NETWORK_SIZE = [29, 15, 15, 1]

#A constant, the learning rate
ETA = 0.1

#The number of epochs to train for
MAX_EPOCHS = 5

#The size of each batch for backpropagation
BATCH_SIZE = 100

#Loads in sigmoid function from the scipy library
sigmoid = lambda x: sp.expit(x)

#%%############################## LOAD DATA ###################################

#Formatting
print('---------------- LOADING DATA ----------------')

#Creates a Network3 object
net3 = Network3(NETWORK_SIZE, ETA, None)

#Defaults lists of inputs and labels
trainingSet = []
testingSet  = []

#Generate tuple of class zeros and class ones to set aside for training set (zeros, ones)
randomTestSelections = generate()

#Read in all data into class[zeros, ones]
readData = []
zeros = open('./class0.csv','r')
ones = open('./class1.csv','r')
readData.append(zeros.readlines())
readData.append(ones.readlines())
zeros.close()
ones.close()

MAX_ROWS = 448 #Zero index (449)

#Loops through each line of the csv file
for each in range(2):
    for rowIndex in randomTestSelections[each]:

        #Takes all the values in the row and splits them by the comma
        rowVals = readData[each][rowIndex].split(',')
        
        #Takes the values except for the labels
        vals = np.asfarray(rowVals[:-1])
        
        #Scales the image
        x = vals
        
        #Takes the label of the data and scales it
        y = int(rowVals[-1].strip())

        testingSet.append((x, y))

    for datum in range(len(readData[each])):

        if datum == 0:
            continue

        if not len(trainingSet) < 1:
            if each == 0 and len(trainingSet[0]) == MAX_ROWS + 1:
                break

        if datum in randomTestSelections[each]:
            continue

        #Takes all the values in the row and splits them by the comma
        rowVals = readData[each][datum].split(',')
        
        #Takes the values except for the labels
        vals = np.asfarray(rowVals[:-1])
        
        #Scales the image
        x = vals
        
        #Takes the label of the data and scales it
        y = int(rowVals[-1].strip())

        trainingSet.append((x, y))


    
    
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