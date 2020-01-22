###############################################################################
#                                                                             #
#                                       NN.py                                 #
#                                                                             #
#                                    J. Steiner                               #
#                                                                             #
#                               With Modifications by                         #
#                                                                             #
#                                    C. Bellamy                               #
#                                                                             #
###############################################################################

from Neuron import Neuron

#A hard coded set of inputs to the neuron
inputMatrix = [ ( [ 1, 1 ], 1 ) ,
                ( [ 1, 0 ], 0 ) ,
                ( [ 0, 1 ], 0 ) ,
                ( [ 0, 0 ], 0 )  ]

#%%########################### PROGRAM EXECUTION ##############################    

#Creates a neuron object
n = Neuron()

#Manually sets weight and bias
n.setWeights([ 1, 1 ])
n.setBias(-2)

#Loops through inputs in the input matrix
for i in inputMatrix:
    
    #Prints the neuron evaluation
    print(i, n.evaluate(i))