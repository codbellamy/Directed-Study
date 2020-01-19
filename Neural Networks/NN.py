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

#Sigmoid function
def sigmoid(z):
    
    return 1.0/(1.0+np.exp(-z))

#Sigmoid function derivative
def sigmoid_prime(z):
    
    return sigmoid(z)*(1-sigmoid(z))

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
            activation = sigmoid(z)
            
        #Returns the final activation before the sigmoid function is run
        return z
    
    ###########################################################################
    #                                                                         #
    # Name:       cost                                                        #  
    # Parameters: data  - all the training data to run through a fwd pass     #
    #             y     - the labels for the training data                    #
    # Returns:    error - the error term of the cost function                 #
    # Notes:      
    #                                                                         #
    ###########################################################################
    
    #Quadratic Cost Function
    def cost(self, data, y):
        
        MSE = 0
        
        for d, actual in zip(data, y):
            
            MSE += (sigmoid(self.fwdPass(d)) - actual)**2
        
        MSE /= 2
        
        return MSE
    
    #Quadratic Cost Function derivative
    def cost_prime(self, data, y):
        
        cost = 0
        
        for d, actual in zip(data, y):
            
            cost += (sigmoid(self.fwdPass(d))-actual)* \
                     sigmoid_prime(self.fwdPass(d))
            
        cost /= len(data)
        
        return cost
    
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
    
    def train(self, data, y, eta = 0.1, epsilon = 0.05):
        
        #Defualts the epoch to be 0
        epoch = 0
        
        #While the error is greater than than 5%
        while self.cost(data, y) >= epsilon:
            
            #Chooses a random index in the trianing data set
            r = np.random.randint(len(data))
            
            #Finds the predicted output
            pred = sigmoid(self.fwdPass(data[r]))
            
            #Finds the true output
            target = y[r]
                        
            #Calculates partial derivatives
            dcost_pred = (pred-target)
            dpred_dz   = sigmoid_prime(self.fwdPass(data[r]))
            dz_dw = [ data[r][0], data[r][1]]
            
            #The new weight is the old weight - the learning rate * the 
            #derivative of the cost with respect to weight
            self.weights = [ w - eta*(dcost_pred*dpred_dz*dw) 
                             for w, dw in zip(self.weights, dz_dw) ]
            
            #The new bias is the old bias - the learning rate * the derivative
            #of the cost with respect to bias
            self.biases = [ b - eta*(dcost_pred*dpred_dz) 
                            for b in self.biases                   ]
            
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
    pred = sigmoid(n.fwdPass(d)) >= 0.5
    
    #Prints predicted, actual, predicted = Actual
    print('Predicted:',pred,'Actual:',actual,'Correct:', pred == actual)
    
#%%########################### SAMPLE OUTPUT ##################################
#Initial Weight:  [array([[ 1.71728863,  1.57549822],
#       [ 0.43519291, -0.43131004]]), array([[-0.62300017,  0.68496398]])]
#Initial Bias:    [[ 0.55354543]
# [ 0.27741642]
# [-1.47872712]]
#Initial Cost:    [0.56832537]
#
#---------- Learning ----------
#
#Epochs to Train: 3515
#Final Weight:    [array([[3.19456913, 3.05277872],
#       [1.91247342, 1.04597046]]), array([[3.4542791 , 4.76224325]])]
#Final Bias:      [array([-3.4616428]), array([-3.73777181]), array([-5.49391535])]
#Final Cost:      [0.0499118]
#
#Predicted: [ True] Actual: 1 Correct: [ True]
#Predicted: [False] Actual: 0 Correct: [ True]
#Predicted: [False] Actual: 0 Correct: [ True]
#Predicted: [False] Actual: 0 Correct: [ True]    
