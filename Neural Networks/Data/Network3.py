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
    # param: size - the sizes of each layer in the neural network
    #        eta  - a constant, the rate at which the network learns 
    def __init__(self, size, eta, convolutionLayers):
        
        #The number of layers in the network
        self.numLayers = len(size)
        
        #Defines the convolutions set up
        self.convolutionLayers = convolutionLayers
        
        #Defaults an empty list to append layer weights to
        self.weights = []
        
        #Loops through the first two layers sizes
        for l in range(len(size)-1):
            
            #Appends an optimal weight. a normal distribution centered at 
            #1 / the square root of the layer size
            #we subtact -0.5 to have negative and positive weights
            self.weights.append(np.random.normal(0.0, pow(size[l], -0.5), 
                                                 (size[l+1], size[l])))
            
        #Defaults an empty list to append layer biases to
        self.biases = []
        
        #Loops through the first two layer sizes
        for l in range(len(size )-1):
            
            #Appends a random bias
            self.biases.append(np.random.normal(size = size[l+1]))
        
        #A constant. the rate at which the network learns
        self.eta = eta
        
        #Loads in sigmoid function from the scipy library
        self.sigmoid = lambda x: sp.expit(x)
        
    ######################################################################
        
    ####################### BACKPROPAGATION FUNCTION #####################
    # param: data   - a single data point x values
    #        actual - the labels for the data point
    def backprop(self, data, actual, convolutionVectors):
        
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
        errorWeight  = [outputErrors]
        errorBias    = [outputErrors]
        
        #Loops through the range of layers for the backward pass
        for l in range(1, self.numLayers-1):
        
            #Calculates the error for each hidden layer
            errorW = np.dot(self.weights[-l].T, errorWeight[l-1]) 
            errorB = (sum(errorBias[-l]) / len(targets)) - activations[-(l+1)]
            
            #Appends the error term to the list of errors
            errorWeight.append(errorW)
            
            #Appends the error term to the list of errors
            errorBias.append(errorB)
        
        #Loops through the range of layers to update the weights
        for l in range(1, self.numLayers):
            
            #How much error there was in the calculation
            dCost    = (errorWeight[l-1] * activations[-l] * 
                       (1.0              - activations[-l]  ))
            
            #How much to change the weights by
            dWeights = np.transpose(activations[-(l+1)])
            
            #How much to change the biases by
            dBiases  = np.transpose(errorBias[l-1])[0]
            
            #Updates each layer's weights
            self.weights[-l] += self.eta * np.dot(dCost, dWeights)
            self.biases[-l]  += self.eta * dBiases
           
        #Loops through each convolution layer index
        if convolutionVectors is not None:
            #Loops through each convolution layer
            for i in self.convolutionLayers:
                #Resets weights
                for w, c in zip(self.weights[i], convolutionVectors[i]):
                    w *= c
        
    ######################################################################
        
    ########################## TRAINING FUNCTION #########################
    # param: trainingSet    - the x values for the training set
    #        trainingLabels - the labels for the training data
    #        maxEpochs      - how many iterations we want to train for
    #        batchSize      - the size of the random sample we generate
    #        verbose        - whether or not to print diagnostics
    #        testSet        - test input vector
    #        testLabels     - test labels vector
    def train(self, trainingSet, maxEpochs, batchSize, 
              verbose = False, testingSet = None,convolutionVectors = None):
        
        randomSample = list(trainingSet)
        
        #Loops through each epoch
        for epoch in range(maxEpochs):
            
            #If we would like to print status
            if verbose:
                
                #Prints the epoch
                print('EPOCH:', epoch+1)
            
            #Shuffles the trainingSet
            random.shuffle(randomSample)
            
            #For each input and output in the random sample of the trianing
            #and test set
            for data in randomSample[0:batchSize]:
                
                #Back propagates
                self.backprop(data[0], data[1], convolutionVectors)
                
            #If a training and test set were passed in
            if testingSet:
                
                #Tests the network
                correct = np.asarray(self.test(testingSet))
                
                #Prints the success rate
                print ('Success Rate = ' + str((correct.sum() / correct.size) 
                                         * 100) + '%')
            
    ######################################################################
            
    ########################## QUERYING FUNCTION #########################
    # param:  data - the datapoint we are predicting the classification of
    # return: the final vector of activations of the output layer
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
    
    ######################################################################
    
    ############################ TEST FUNCTION ###########################
    # param:  testInputs - the input vector of the test set
    #         testLabels - the labels of each data point in the test set
    # return: whether or not for each test data point, the prediction was
    #         correct
    def test(self, testingSet):
    
        #An empty lisst of if the given prediction was correct or not
        predicted = []
        
        #Loops through the input and label in the test set
        for data in testingSet:
            
            #Query's the network to get the output
            outputs = self.query(data[0])
            
            #The index of the value with the highest predicted output
            label = np.argmax(outputs)
            
            #If the prediciton was correct
            if label == data[1]:
                
                #Appends that this was a correct prediction
                predicted.append(1)
            
            #If the prediction was incorrect
            else:

                #Appends that this was an incorrect prediction
                predicted.append(0)
                
        #Returns the list of correct or incorrect predictions
        return predicted
    
    ######################################################################