###############################################################################
#                                                                             #
#                                   Network3.py                               #
#                                   J.Steiner                                 #
#                                                                             #
###############################################################################

#%%################################ IMPORTS ###################################

#Imports matrix and vector operations
import numpy as np

#Imports the sigmoid function
import scipy.special as sp

#Imports randomness algorithms
import random


#%%############################# NETWORK3 CLASS ###############################
class Network3:
    
    
    ############################## CONSTRUCTOR ###########################  
    def __init__(self, size, eta):
        
        #The number of layers in the network
        self.numLayers = len(size)
        
        #Defaults an empty list to append layer weights to
        self.weights = []
        
        #Loops through the first two layers sizes
        for l in range(len(size)-1):
            
            #Appends an optimal weight. a normal distribution centered at 
            #1 / the square root of the layer size
            #we subtact -0.5 to have negative and positive weights
            self.weights.append(np.random.normal(0.0, pow(size[l], -0.5), 
                                                 (size[l+1], size[l])))
        
        #A constant. the rate at which the network learns
        self.eta = eta
        
        #Loads in sigmoid function from the scipy library
        self.sigmoid = lambda x: sp.expit(x)
        
        
        
    ####################### BACKPROPAGATION FUNCTION #####################
    def backprop(self, data, actual):
        
        #Converts the inputs and targets into two dimensional arrays
        inputs      = np.array(data, ndmin=2).T
        targets     = np.array(actual, ndmin=2).T
        
        #Defaults a list of activations from different levels of the network
        activations = [ inputs ]
        
        #Loops through the range of layers for the forward pass
        for l in range(self.numLayers-1):
            
            #Calculates the output of each layer
            z = np.dot(self.weights[l], activations[l])
            
            #Gets the output of each layer after it has been run through the
            #sigmoid function
            activations.append(self.sigmoid(z))
        
        
        #Calculates the error in the output layer
        outputErrors = targets - activations[-1]
        
        #Stores the output error in a list of errors
        errors       = [outputErrors]
        
        #Loops through the range of layers for the backward pass
        for l in range(1, self.numLayers-1):
        
            #Calculates the error for each hidden layer
            error = np.dot(self.weights[-l].T, errors[l-1]) 
            
            #Appends the error term to the list of errors
            errors.append(error)
            
        #Loops through the range of layers to update the weights
        for l in range(1, self.numLayers):
            
            #How much error there was in the calculation
            dCost    = (errors[l-1] * activations[-l] * 
                       (1.0         - activations[-l]   ))
            
            #How much to change the weights by
            dWeights = np.transpose(activations[-(l+1)])
            
            #Updates each layer's weights
            self.weights[-l] += self.eta * np.dot(dCost, dWeights)
        
    ########################## TRAINING FUNCTION #########################
    def train(self, trainingSet, trainingLabels, maxEpochs, batchSize, 
              verbose = True,
              testSet = None, testLabels = None, ):
        
        #Loops through each epoch
        for epoch in range(maxEpochs):
            
            #If we would like to print status
            if verbose:
                
                #Prints the epoch
                print('EPOCH:', epoch+1)
               
            #All the possible values that could be chosen randomly
            rands = list(range(len(trainingSet)))
            
            #An empty list for random values
            randomSample = []
            
            #For each randomly selected value
            for m in range(batchSize):
                #Choose a random index with replacement
                randomSample.append(rands.pop(random.randrange(len(rands))))
            
            #For each input and output in the random sample of the trianing
            #and test set
            for r in randomSample:
                
                #Back propagates
                self.backprop(trainingSet[r], trainingLabels[r])
                
            #If a training and test set were passed in
            if testSet and testLabels:
                
                #Tests the network
                correct = np.asarray(self.test(testSet, testLabels))
                
                #Prints the success rate
                print ('Success Rate = ' + str((correct.sum() / correct.size) 
                                         * 100) + '%')
            
            
    ########################## QUERYING FUNCTION #########################
    def query(self, data):
        
        #Converts the input data into a 2 dimensional array
        inputs = np.array(data, ndmin=2).T
        
        #Loads in the input array as the first activation
        activation = inputs
        
        #Loops through each layer
        for l in range(self.numLayers-1):
            
            #Computes the output of the layer
            z = np.dot(self.weights[l], activation)
            
            #Calculates the sigmoid function of the layer output
            activation = self.sigmoid(z)
        
        #Returns the final output
        return activation
    
    ############################ TEST FUNCTION ###########################
    def test(self, testInputs, testLabels):
    
        #An empty lisst of if the given prediction was correct or not
        predicted = []
        
        #Loops through the input and label in the test set
        for x, y in zip(testInputs, testLabels):
            
            #Query's the network to get the output
            outputs = self.query(x)
            
            #The index of the value with the highest predicted output
            label = np.argmax(outputs)
            
            #If the prediciton was correct
            if label == y:
                
                #Appends that this was a correct prediction
                predicted.append(1)
            
            #If the prediction was incorrect
            else:

                #Appends that this was an incorrect prediction
                predicted.append(0)
                
        #Returns the list of correct or incorrect predictions
        return predicted