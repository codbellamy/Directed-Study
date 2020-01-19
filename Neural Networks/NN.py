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

#Imports many randomness algorithms
import random

def sigmoid(z):
    
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    
    return sigmoid(z)*(1-sigmoid(z))

#%%############################################################################
#                                                                             #
#                                  NN Class                                   #
#                                                                             #
###############################################################################
class NN:
    
    def __init__(self):
        
        self.weights = [ [ np.random.randn(), np.random.randn() ]] 
        #self.weights = [ [1,1] ]                
        #self.biases  = [ -2]
        self.biases = [ np.random.randn() ] 
    
    def fwdPass(self, data):
        
        activation = data
        
        for weights, bias in zip(self.weights, self.biases):
            
            z = np.dot(activation, weights) + bias
            
            activation = z
            
        return activation
    
    def cost(self, data, y):
        
        MSE = 0
        
        for d, actual in zip(data, y):
            
            MSE += (sigmoid(self.fwdPass(d)) - actual)**2
        
        MSE /= len(data)
        
        return MSE
    
    def cost_prime(self, data, y):
        
        cost = 0
        
        for d, actual in zip(data, y):
            
            cost += 2 * (sigmoid(self.fwdPass(d))-actual) *sigmoid_prime(self.fwdPass(d))
            
        cost /= len(data)
        
        return cost
    
    def learn(self, data, y):
        
        epoch = 0
        
        while self.cost(data, y) >= 0.01:
            
            r = np.random.randint(len(data))
            
            pred = sigmoid(self.fwdPass(data[r]))
            target = y[r]
                        
            dcost_pred = 2*(pred-target)
            dpred_dz   = sigmoid_prime(self.fwdPass(data[r]))
            dz_dw = [ data[r][0], data[r][1]]
            
            self.weights = [ w - 0.1*(dcost_pred*dpred_dz*dw) for w, dw in zip(self.weights, dz_dw)]
            self.biases = [ b - 0.1*(dcost_pred*dpred_dz) for b in self.biases]
            
            epoch+= 1
            
            if epoch > 1000000:
                break
            
        print(epoch)
            

x = [[1, 1],
     [1, 0],
     [0, 1],
     [0, 0]]
y = [ 1, 0, 0, 0 ]


n = NN()

print(n.weights)
print(n.biases)
print(n.cost(x, y))
print(n.cost_prime(x,y))
n.learn(x,y)
print('---------------')
print(n.weights)
print(n.biases)
print(n.cost(x,y))
print(n.cost_prime(x,y))

for d in x:
    
    print(sigmoid(n.fwdPass(d)) >= 0.5)