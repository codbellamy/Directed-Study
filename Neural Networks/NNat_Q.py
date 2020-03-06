###############################################################################
#                                                                             #
#                                  NNat_Q.py                                  #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%############################# LOADS MODULES ################################

#Imports the ability to easily work with matricies
import numpy as np

#%%########################## ACTIVATION FUNCTIONS ############################

###############################################################################
# Name:   sigmoid
# Param:  a - the input to the sigmoid
# Return: z - the output to the sigmoid
# Notes:  calculates the sigmoid of at x value of the input
def sigmoid(a):

    #Calculates the sigmoid
    z = 1.0 / ( 1 + np.exp(-a) )

    #Returns the output of the function
    return z
###############################################################################
    
###############################################################################
# Name:   softmax
# Param:  a - the input to the softmax function
# Return: z - the output of the softmax function
# Notes:  calculates a confidence value of the network for each output
def softmax(a):

    #Calculates a confidence value of the network for each output
    z = np.exp(a) / sum( np.exp(a) )

    #Returns the output of the softmax function
    return z
###############################################################################

#%%############################ CLASS DEFINITION ##############################
class NNat_Q:

    ###########################################################################
    # Name:  __init__
    # Param: size         - a list of neuron counts for each layer in the
    #                       network
    #        learningRate - a hyper parameter to control how drastically the 
    #                       corrects for incorrect predictions
    # Notes: class constructor
    def __init__(self, size, learningRate):

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
            
        #Defaults an empty list to append layer biases to
        self.biases = []
        
        #Loops through the first two layer sizes
        for l in range(len(size )-1):
            
            #Appends a random bias
            self.biases.append(np.random.normal(size = size[l+1]))
        
        #A constant. the rate at which the network learns
        self.eta = learningRate
    ###########################################################################

    ###########################################################################
    # Name:   query
    # Param:  inputVector - the input vector to pass through the network
    #         desired     - the desired output of the network
    # Return: z           - the activations for each layer, returns the final  
    #                       layer's activations
    # Notes:  returns the output of the network and corrects for the desired 
    #         output of the network
    def query(self, inputVector, desired):

        #Converts the inputs and targets into two dimensional arrays
        inputs      = np.array(inputVector, ndmin=2).T
        targets     = np.array(desired,     ndmin=2).T
        
        #Defaults a list of activations from different levels of the network
        activations = [ inputs ]
        
        #Loops through the range of layers for the forward pass
        for l in range(self.numLayers-1):
            
            #Calculates the output of each layer
            a = np.dot(self.weights[l], activations[l])

            #Runs all the outputs through the sigmoid function except for the 
            #last one
            activations.append(sigmoid(a))
        
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

        #After learning has been done, returns the output of the network prior
        #to learning
        return activations[-1]
    ###########################################################################

###############################################################################