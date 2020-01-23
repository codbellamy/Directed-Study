###############################################################################
#                                                                             #
#                                    checkers.py                              #
#                                                                             #
#                                    J. Steiner                               #
#                                                                             #
###############################################################################

#%%################################ IMPORTS ###################################

#Imports the ability to work with matricies easily
import numpy as np

#Imports the ability to work with advanced mathematic functions
import math

#Imports the ability to work with images easily
import cv2 as cv

#%%############################################################################
#                                                                             #
#                                  NN Class                                   #
#                                                                             #
###############################################################################
class NN:
    
    ###########################################################################
    #                                                                         #  
    # Name:       __init__                                                    #
    # Parameters: p    - the number of features (AKA input neurons) initially #
    #                    passed into the network                              #
    #             size - the number of neurons at each layer of the network   #
    # Returns:    N/A                                                         #  
    # Notes:      class constructor, initializes instance variables           #
    #                                                                         #
    ###########################################################################
    
    def __init__(self,size):
        
        #stores the network size passed into the object created
        self.size = size
        
        #Defaults the weights as random numbers sampled from a Guassian 
        #distribution, the list comprehension set up puts a structure of 
        #weight matricies according to network size
        self.weights = [
                        np.array([ [ 0.5, 0.25, 0.5, 0.25 ], [ 0.25, 0.5, 0.25, 0.5 ] ]),
                        np.array([ [ 0.5, 0.25 ] ])
                       ]
                
        #Defaults the biases as random numbers sampled from a Guassian
        #distribution, the numpy function sets up a matrix containing all
        #biases according to network size
        self.biases = [
                        np.array([ 1, 2 ]),
                        np.array([ 3 ])
                      ]
        
        self.outputs = []

    
    ###########################################################################
    #                                                                         #
    # Name:       fwdPass                                                     #
    # Parameters: data       - a single instance of data input                #
    # Returns:    activation - the final output from the output neruon        #
    # Notes:      runs a single data point through the network                #
    #                                                                         #
    ###########################################################################
    
    def fwdPass(self, data):
        
        
        self.outputs = []
        
        #Sets the first output to be exactly equal to the input data
        activation = self.sigmoid(np.array(data))
        self.outputs.append(activation)
        
        #Loops through layers' weight and bias matricies for each neuron
        for weights, biases in zip(self.weights, self.biases):
            
            #Defaults the output vector
            layerOutput = []
            
            #Loops through each weight and bias vectors in the layers' total
            #weights and biases
            for weight, bias in zip(weights, biases):
                
                #Takes the output from a neuron
                z = np.dot(activation, weight) + bias

                #Appends to the layer output
                layerOutput.append(self.sigmoid(z))
                
            #Sets the new activation equal to the sigmoid function of the
            #neuron output
            activation = np.array(layerOutput)
            self.outputs.append(activation)
            
        #Returns the final activation before the sigmoid function is run
        return activation
    
    ###########################################################################
    #                                                                         #
    # Name:       sigmoid                                                     #  
    # Parameters: z     - the value passed in, typically the ouput activation # 
    #                     of a neuron                                         #
    # Returns:    [...] - the sigmoid function value of z                     #
    # Notes:      This allows our neuron output to be any number from 0 to 1  #
    #                                                                         #
    ###########################################################################
    
    def sigmoid(self, z):
        
        #The sigmoid function value of z
        return 1.0/(1.0+np.exp(-z))
    
    ###########################################################################
    #                                                                         #
    # Name:       cost                                                        #  
    # Parameters: data  - all the training data to run through a fwd pass     #
    #             y     - the labels for the training data                    #
    # Returns:    cost - the average cost for all the data passed in          #
    # Notes:      The cross-entropy cost funtion, this is a way to assesss    #
    #             success                                                     #
    #                                                                         #
    ###########################################################################
    
    #Cross-Entropy Cost Function
    def cost(self, data, y):
        
        #Defaults the cost to be 0
        cost = 0
        
        #For all the data that was passed in
        for d, actual in zip(data, y):
                        
            #Calculates the predicted response
            networkOutput = self.fwdPass(d)
            
            a1 = networkOutput[0]
            a2 = actual[0]

            #The actual response
            cost += (a2*math.log(a1)) + ((1-a2)*math.log(1-a1))
        
        #Divides by the -lenght of the data, we want the average cost
        cost /= -len(data)
        
        #Returns the average cost of the data
        return cost
    
    def backpropogate(self, actual):
        
        dW = [ [], [] ]
        dB = [ [], [] ]
        y = actual
        for i in range(1, len(n.outputs)): #layer
            
            for z in n.outputs[-i]: #ouput neurons
                
                dW[-i].append([ a*(z-y) for a in n.outputs[-i-1] ]) #layer neurons

                dB[-i].append(np.array(z)-np.array(y))
                
            y = n.outputs[-i]
                
        return dW,dB
        
    
    ###########################################################################
    #                                                                         #
    # Name:       test                                                        #
    # Parameters: data    - all the training data to run through the training #
    #                       algorithm                                         #
    #             y       - all the trianing labels to compare outputs to     #
    # Returns:    success - the percentage of observations the network got    #
    #                       correct                                           #
    # Notes:      assesses accuracy of the neural network                     #
    #                                                                         #
    ###########################################################################
    
    def test(self, data, y):
        
        #Defualts a counter of the observations correctly guessed
        ctr = 0
        
        #Prints results
        for o,actual in zip(self.outputs[-1], y):
            
            pred = o >= 0.5
            
            #Adds a 0 if incorrect, adds a 1 if correct
            ctr += pred == actual

        #Returns the success rate percentage
        return ctr/ len(data)
    
    ###########################################################################
    #                                                                         #
    # Name:       train                                                       #
    # Parameters: data    - all the training data to run through the training #
    #                       algorithm                                         #
    #             y       - all the trianing labels to compare outputs to     #
    #             eta     - the learning rate, defaulted to 10%               #
    #             epsilon - the stopping rule, defaulted to 5% error          #
    # Returns:    epoch   - the amount of epochs it took to finish training   #
    # Notes:      runs the training algorithm                                 #
    #                                                                         #
    ###########################################################################
    
    def train(self, data, y, eta = 0.1, epsilon = 1.0):
        
        #Defualts the epoch to be 0
        epoch = 0
        successRate = 0
        
        #While the success rate is less than 100%
        while successRate < 1.1:
            
            dW,dB = self.backpropogate(y)
            
            for i in range(len(n.weights)): #layer
                
                for j in range(len(n.weights[i])): #neurons
                    
                    n.weights[i][j] = [ w - eta*dw for w, dw in zip(n.weights[i][j], dW[i][j])]
                    n.biases[i][j] -= eta*dB[i][j]

            #Incrments the epoch
            epoch += 1
            
            successRate = self.test(data,y)
            
            print('Epoch:',epoch,'  Success Rate:',
                  str(successRate*100)+'%')
    
            #If there has been more than 10000 epochs
            if epoch >= 2:
                
                #Quits out of the loop
                return epoch
        
        #Returns how many epoch it took to get to the desired error rate
        return epoch

#%%############################# IMPLEMENTATION ###############################
      
####### Data #######

x = [ [ 255, 0, 0, 255 ] ]#, [ 0, 255, 255, 0 ] ]
y = [ [ 1 ] ]#, [ 0 ] ]


#Creates a neural network object
n = NN([4, 2, 1])

#Prints weights, bias, and inital cost
print()
print('Initial Cost:   ', n.cost(x, y))
print()
print('---------- Learning ----------')
print()
print('Epochs to Train:', n.train(x,y, epsilon = 0.8))
print('Final Cost:     ', n.cost(x,y))
print()

#Prints a success rate
print('Success Rate:', str(n.test(x,y) * 100)+'%')
    
#%%########################### SAMPLE OUTPUT ##################################
