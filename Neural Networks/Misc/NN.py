###############################################################################
#                                                                             #
#                                       NN.py                                 #
#                                                                             #
#                                    J. Steiner                               #
#                                                                             #
###############################################################################

#%%################################ IMPORTS ###################################

#Imports the ability to work with matricies easily
import numpy as np

#Imports the ability to work with advanced mathematic functions
import math

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
        self.weights = [ np.random.randn(self.size[i], self.size[i-1]) 
                         for i in range(1, len(self.size))             ]
        
        #Defaults the biases as random numbers sampled from a Guassian
        #distribution, the numpy function sets up a matrix containing all
        #biases according to network size
        self.biases = np.random.randn(len(size), 1)
    
    ###########################################################################
    #                                                                         #
    # Name:       fwdPass                                                     #
    # Parameters: data       - a single instance of data input                #
    # Returns:    activation - the final output from the output neruon        #
    # Notes:      runs a single data point through the network                #
    #                                                                         #
    ###########################################################################
    
    def fwdPass(self, data):
        
        #Sets the first output to be exactly equal to the input data
        activation = data
        
        #Loops through the weights and bias for each neuron
        for weights, bias in zip(self.weights, self.biases):
            
            #Takes the output from a neuron
            z = np.dot(activation, weights.transpose()) + bias
            
            #Sets the new activation equal to the sigmoid function of the
            #neuron output
            activation = self.sigmoid(z)
            
        #Returns the final activation before the sigmoid function is run
        return z
    
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
            a = self.sigmoid(self.fwdPass(d))
            
            #The actual response
            cost += (actual*math.log(a)) + ((1-actual)*math.log(1-a))
        
        #Divides by the -lenght of the data, we want the average cost
        cost /= -len(data)
        
        #Returns the average cost of the data
        return cost
    
    ###########################################################################
    #                                                                         #
    # Name:       cost_prime_W                                                #  
    # Parameters: data  - all the training data to run through a fwd pass     #
    #             y     - the labels for the training data                    #
    # Returns:    cost - the average derivative of cost for all the data      # 
    #                    passed in with respect to the weights                #
    # Notes:      The derivative of the cross-entropy cost funtion with       #
    #             respect to weights, this is how we will change the weights  #
    #                                                                         #
    ###########################################################################
    
    #Cross-Entropy Cost Function derivative with respect to weights
    def cost_prime_W(self, data, y, reg = 0.1):
        
        #Defaults the cost to 0
        cost = 0
        
        #Loops through the predicted and actual value of each data point
        for d, actual in zip(data, y):
            
            #The predicted data point
            a = self.sigmoid(self.fwdPass(d))
            
            #The actual data point
            cost += d*(a - actual)
            
        #Divides by the number of data points, we want the average derivative
        cost /= len(data)
        
        #Returns the average derivative with respect to weights
        return cost
    
    ###########################################################################
    #                                                                         #
    # Name:       cost_prime_B                                                #  
    # Parameters: data  - all the training data to run through a fwd pass     #
    #             y     - the labels for the training data                    #
    # Returns:    cost - the average derivative of cost for all the data      # 
    #                    passed in with respect to bias                       #
    # Notes:      The derivative of the cross-entropy cost funtion with       #
    #             respect to bias, this is how we will change bias            #
    #                                                                         #
    ###########################################################################
    
    #Cross-Entropy Cost Function derivative with respect to bias
    def cost_prime_B(self, data, y):
        
        #Defeaults the cost to be 0
        cost = 0
        
        #Loops through the predicted and actual value of each data point
        for d, actual in zip(data, y):
            
            #The predicted data point
            a = self.sigmoid(self.fwdPass(d))
            
            #Adds the predicted data point - the actual data point
            cost += (a - actual)
          
        #Divdes by the number of data points, we want the average derivative
        cost /= len(data)
        
        #Returns the average derivative with repect to bias
        return cost
    
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
        for d,actual in zip(x,y):
            
            #predicted
            pred = n.sigmoid(n.fwdPass(d)) >= 0.5
            
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
        
        #While the success rate is less than 100%
        while self.test(data, y) < epsilon:
            
            #Calculates partial derivatives
            dCost_dW = self.cost_prime_W(data, y)
            dCost_dB = self.cost_prime_B(data, y)
            
            #The new weight is the old weight - the learning rate * the 
            #derivative of the cost with respect to weight
            self.weights = [ w - eta*dCost_dW 
                             for w in self.weights ]
            
            #The new bias is the old bias - the learning rate * the derivative
            #of the cost with respect to bias
            self.biases = [ b - eta*dCost_dB 
                            for b in self.biases   ]
            
            #Incrments the epoch
            epoch += 1
            
            #If there has been more than 10000 epochs
            if epoch > 10000:
                
                #Quits out of the loop
                break
        
        #Returns how many epoch it took to get to the desired error rate
        return epoch

#%%############################# IMPLEMENTATION ###############################
      
####### Data #######
x = [[1, 1],       #
     [1, 0],       #
     [0, 1],       #
     [0, 0]]       #
y = [ 1, 0, 0, 0 ] #
####################

#Creates a neural network object
n = NN([2, 2, 1])

#Prints weights, bias, and inital cost
print()
print('Initial Weight: ', n.weights)
print('Initial Bias:   ', n.biases)
print('Initial Cost:   ', n.cost(x, y))
print()
print('---------- Learning ----------')
print()
print('Epochs to Train:', n.train(x,y))
print('Final Weight:   ', n.weights)
print('Final Bias:     ', n.biases)
print('Final Cost:     ', n.cost(x,y))
print()

#Prints results
for d,actual in zip(x,y):
    
    #predicted
    pred = n.sigmoid(n.fwdPass(d)) >= 0.5
    
    #Prints predicted, actual, predicted = Actual
    print('Predicted:',pred,'Actual:',actual,'Correct:', pred == actual)
    
#%%########################### SAMPLE OUTPUT ##################################
#Initial Weight:  [array([[-0.66391714, -1.12887629],
#       [ 0.54771647,  1.20667158]]), array([[0.0346137 , 1.34434781]])]
#Initial Bias:    [[ 2.78358205]
# [-0.21102442]
# [ 1.04729601]]
#Initial Cost:    1.1178081293910715
#
#---------- Learning ----------
#
#Epochs to Train: 24
#Final Weight:    [array([[-0.80164086, -1.27230312],
#       [ 0.40999275,  1.06324476]]), array([[-0.10311002,  1.20092098]])]
#Final Bias:      [array([1.91796633]), array([-1.07664014]), array([0.18168029])]
#Final Cost:      0.6664386798489834
#
#Predicted: [ True] Actual: 1 Correct: [ True]
#Predicted: [False] Actual: 0 Correct: [ True]
#Predicted: [False] Actual: 0 Correct: [ True]
#Predicted: [False] Actual: 0 Correct: [ True]
