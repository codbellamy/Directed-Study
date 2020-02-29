###############################################################################
#                                                                             #
#                                   Network.py                                #
#                                                                             #
#                                   J. Steiner                                #
#                                                                             #
###############################################################################

import numpy as np
import math
import random
from Neuron import Neuron
from Neuron import sigmoid

class Network:
    
    def __init__(self, size):
        
        #Defaults the network to be an empty list of Neuron object
        self.network = []
        
        #Loops through each layer size index from the first hidden layer to the
        #output layer
        for i in range(1, len(size)):
            
            #Appends a list of neurons the size of the layer's size with as
            #many inputs as the previous layers
            self.network.append([Neuron(size[i-1])] * size[i])
    
    #Conducts a forward pass of the network for a single data vector
    def forwardPass(self, data):
        
        #Initializes the array activation to be exactly equal to the input
        #vector passed as a parameter
        activation  = data
        
        #Loops through every layer in the network
        for layer in self.network:
            
            #Defaults an empty list recieve the output for each neuron in each
            #layer
            layerOutput = []
            
            #Loops through every neuron in each layer
            for neuron in layer:
                
                #Appends the neuron's output when the array activation is
                #passed as input
                layerOutput.append(neuron.evaluate(activation))
                
            #Converts the layer's outputs to a numpy array and resets
            #activation to be the array of outputs from the layer
            activation  = np.array(layerOutput)
        
        #Returns the final activation vector as a numpy array
        return activation
    
    #Calculates the cost with respect to a single input vector
    def cost(self, data, actual):
        
        #Calculates the activations with respect to a single input vector
        activations = self.forwardPass(data)
        
        #Calculates the sum of costs for each element in the output vector
        cost        = sum( [ y*math.log(a) + (1-y)*math.log(1-a) 
                             for a, y in zip(activations, actual)] )
    
        #Averages the sum of costs with respect to the length of the output
        #vector
        cost       /= -len(actual)
        
        #Returns the cost for these single input and output vectors
        return cost
    
    #Calculates the partial deriavatives with respect to weights and biases
    def backwardPass(self, data, actual):
        
        #Defaults the lists that will be used to store the partial derivatives
        #of weight and bias
        dw = []
        db = []
        
        #Conducts a forward pass using the given input vector so that the
        #neuron's instance variables are the outputs corresponding to the
        #given input vector
        self.forwardPass(data)
        
        #Defualts an set of variables to store and update response vectors
        y = actual
        ynew = []
        
        #Loops through layers in the network
        for l in range(1, len(self.network)):
            
            #Resets weight and bias lists
            dw.append([])
            b = []
            
            #Loops through each neuron in the network
            for j in range(len(self.network[-l])):
                
                #Calulates the average of the response vector
                avgY = sum(y) / len(y)
                
                #Calculates the partial derivative of cost with respect to
                #every neuron's weights
                dw[l-1].append(np.array([a.getOutput() *
                                        (self.network[-l][j].getOutput()-avgY) 
                                        for a in self.network[-l-1]]))
    
                #Calculates the partial derivative of the cost with respect to
                #every neuron's bias
                b.append(np.array(self.network[-l][j].getOutput()-avgY))

                #Appends the old neuron outputs to the response vecto
                ynew.append(self.network[-l][j].getOutput())
                
            #Updates the response vector
            y = ynew
            
            #Updates the derivatives of the bias
            db.append(b)
        
        #Resets what we store the weight and bias derivatives in for the
        #following special case
        dw.append([])
        b = []
        
        #Calculates the end of the network
        L = len(self.network)
        
        #Loops through each neuron in the first hidden layer
        for j in range(len(self.network[0])):
            
            #Calulates the average of the response vector
            avgY = sum(y) / len(y)
            
            #Calculates the partial derivative for each neuron in the first
            #hidden layer (weights)
            dw[L-1].append(np.array([x * (self.network[-L][j].getOutput()-avgY) 
                                     for x in sigmoid(data) ]))
    
            #Caluculates the partial derivative for each neuron in the first
            #hidden layer (bias)
            b.append(np.array(self.network[-L][j].getOutput()-avgY))
        
        #Adds to the derivative of bias vector
        db.append(b)
        
        #Re-orients the derivative of weight and bias lists
        dw.reverse()
        db.reverse()
        
        #Returns the partial derivative of cost with respect to each weight and
        #bias in the network
        return dw,db
    
    
    def train(self, trainingInput, trainingResponse, 
                    maxEpochs = 10, batchSize = 2):
       
        #Loops through each epoch to train the network
        for epoch in range(maxEpochs):
            print('EPOCH:', epoch+1)
            #Generates random indicies for a random sample
            randIndicies = [ random.randrange(0, len(trainingInput)) 
                             for _ in range(batchSize)              ]
            
            #Defualts the partial derivative values
            dW = None
            dB = None
            
            #For each data point in the index
            for r in randIndicies:

                #Takes the backward pass partial derivative values
                w, b = self.backwardPass(trainingInput[r], 
                                         trainingResponse[r])
                
                #If it is the not the first round
                if dW is not None and dB is not None:
                    
                    #Sums the weights
                    for l in range(len(w)):
                        for j in range(len(w[l])):
                            dW[l][j] += w[l][j]
                        
                    #Sums the biases
                    for l in range(len(b)):
                        for j in range(len(w[l])):
                            dB[l][j] += b[l][j]
                
                #If the derivative and bias lists were empty
                else:
                    
                    #Initializes them
                    dW = w
                    dB = b
            
            #Updates weights and biases
            for l in range(len(dW)):
                for j in range(len(self.network[l])):
                    
                    dW[l][j] /= batchSize
                    dB[l][j] /= batchSize
                    
                    self.network[l][j].updateWeights(dW[l][j])
                    self.network[l][j].updateBias(dB[l][j])
    
    #Runs through the test set to determine accuracy
    def test(self, testSet, testResponse):
        
        #Counts how many correct predictions there were
        count = 0
        
        #For each index in the test set
        for i in range(len(testSet)):
            
            #Conducts a forward pass and stores the predicted value
            predicted = self.forwardPass(testSet[i])
            
            #Increments count if there was a correct prediciton
            count += ((predicted >= 0.5) == testResponse[i])
        
        #Returns the number of successful predictions
        return count
    