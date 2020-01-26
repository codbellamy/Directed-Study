###############################################################################
#                                                                             #
#                                   Neuron.py                                 #
#                                                                             #
#                                   J. Steiner                                #
#                                                                             #
###############################################################################

#Imports the ability to easily work with matricies
import numpy as np

#A global function for determining neuron output
def sigmoid(z):
    #The sigmoid function value of z
    return 1.0/(1.0+np.exp(-z))

#Neruon class
class Neuron:
    
    #Neuron class constructor
    def __init__(self, numInputs):
        
        #Initializes random weights from a uniform distribution with a range
        #of 0, 1 as a numpy array with 1 row and a column for each input the
        #neuron will recieve
        self.weights = np.random.uniform(size = numInputs)
        
        #Initializes a random bias from a 
        self.bias    = np.random.uniform()
        
        #Defaults an output to be -1, this is because since range of possible
        #outputs is in the interval [0, 1], -1 is treated as an indication that
        #the neuron output was not calculated
        self.output  = -1
        
    #Calculates neuron output
    def evaluate(self, data):
        
        #Calculates the value z as the wieghted input plus the neuron bias
        z = np.dot(self.weights, data) + self.bias
        
        #Calculates the neuron output to be the sigmoid function of z
        self.output = sigmoid(z)
        
        #Returns the neuron output
        return self.output
        
    #Updates weights given the values deltaWeights
    def updateWeights(self, deltaWeights, eta = 0.1):
        
        #Updates the weights by the corresponding deltaWeight*eta
        #eta being the learning rate which is defualted to 0.1
        self.weights -= eta * deltaWeights
        
    #Updates bias given the value deltaBias
    def updateBias(self, deltaBias, eta = 0.1):
        
        #Updates the neuron's bias by the corresponding deltaBias*eta
        #eta being the learning rate which is defaulted to 0.1
        self.bias    -= eta * deltaBias
    
    #Gets the neuron output
    def getOutput(self): return self.output
    
