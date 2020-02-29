###############################################################################
#                                                                             #
#                                  digits.py                                  #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%################################ IMPORTS ###################################

#Imports matrix and vector operations
import numpy as np

#Imports the Network3 class
from Network3 import Network3

#%%############################### CONSTANTS ##################################

#The size of each layer in the network
NETWORK_SIZE = [784, 200, 10]

#A constant, the learning rate
ETA = 0.1

#The number of epochs to train for
MAX_EPOCHS = 10

#The size of each batch for backpropagation
BATCH_SIZE = 750

#%%############################## LOAD DATA ###################################

#Formatting
print('---------------- LOADING DATA ----------------')

#Creates a Network3 object
net3 = Network3(NETWORK_SIZE, ETA)

#Loads in the training set
trainFile = open("./mnist_train.csv", 'r')
trainList = trainFile.readlines()
trainFile.close()

#Loads in the test set
testFile = open("./mnist_test.csv", 'r')
testList = testFile.readlines()
testFile.close()

#Defaults lists of inputs and labels
trainInputs = []
trainLabels = []
testInputs = []
testLabels = []

#Loops through each line of the csv file
for datum in trainList:
    
    #Takes all the values in the row and splits them by the comma
    rowVals = datum.split(',')
    
    #Takes all values except the labels and scales them to be intensity from
    #0.01 to 0.99
    x = (np.asfarray(rowVals[1:]) / 255.0 * 0.99) + 0.01
    
    #Takes the label of the data and scales it
    y = np.zeros(NETWORK_SIZE[-1]) + 0.1
    y[int(rowVals[0])] = 0.99
    
    #Appends the scaled label and datum to the inputs and labels list
    trainInputs.append(x)
    trainLabels.append(y)

#Loops through each line of the csv file
for datum in testList:
    
    #Takes all the values in the row and splits them by the comma
    rowVals = datum.split(',')
    
    #Takes all values except the labels and scales them to be intensity from
    #0.01 to 0.99
    x = (np.asfarray(rowVals[1:]) / 255.0 * 0.99) + 0.01
    
    #Calculates the true label
    y = int(rowVals[0])
    
    #Appends the scaled input and true label to the inputs and labels list
    testInputs.append(x)
    testLabels.append(y)
    
    
#%%############################### EXECUTION ##################################
    
#Formatting
print('---------------- START TRAINING --------------')
    
#Trains the neural network
net3.train(trainInputs, trainLabels, MAX_EPOCHS, BATCH_SIZE,
           verbose = True, testSet = testInputs, testLabels = testLabels)

#Formatting
print('---------------- DONE TRAINING ----------------')

#Whether or not the network got a prediction correct
predicted = net3.test(testInputs, testLabels)

#The array of whether or not the prediction was correct
correct = np.asarray(predicted)

#Prints the success rate
print ('Success Rate = ' + str((correct.sum() / correct.size) * 100) + '%')