###############################################################################
#                                                                             #
#                                     Neuron.py                               #
#                                                                             #
#                                    J. Steiner                               #
#                                                                             #
#                               With Modifications by                         #
#                                                                             #
#                                    C. Bellamy                               #
#                                                                             #
###############################################################################

#%%############################################################################
#                                                                             #
#                                Nueron Class                                 #
#                                                                             #
###############################################################################
class Neuron:
    
    ###########################################################################
    #                                                                         #
    # Name:       __init__                                                    #
    # Parameters: N/A                                                         #
    # Reutrns:    N/A                                                         #  
    # Notes:      Class constructor, defaults instance variables: weights,    #
    #             bias, input, output                                         #
    #                                                                         #
    ###########################################################################
    
    def __init__(self):
        
        #Defaults instance variables
        self.weights, self.bias = ( None, None )
        self.input, self.output = ( None, None )
        
    ###########################################################################
    #                                                                         #
    # Name:       evaluate                                                    #
    # Parameters: inputs - the feature vector and classification              #
    # Reutrns:    True or False, if the neuron is activated                   #  
    # Notes:      test data should have a classification of None              #
    #                                                                         #
    ###########################################################################
        
    def evaluate(self, inputs):
        
        #Stores the feature vector into variable x
        x           = inputs[0]

        #Computes the element-wise multiplication of the features and weights
        weightedInput  = [ x[i] * self.weights[i] for i in range(len(x)) ]
        
        #Computes the dot product of features and weights, adds bias
        self.output = sum( weightedInput ) + self.bias
        
        #Returns if the output is greater than the threshold of 0
        return self.output >= 0
    
    ###########################################################################
    #                                                                         #
    # Name:       setWeights                                                  #
    # Parameters: newWeights - list of new weights                            #
    # Reutrns:    N/A                                                         #  
    # Notes:      Setter method, newWeights should be a list                  #
    #                                                                         #
    ###########################################################################
    
    def setWeights(self, newWeights):
        
        #Sets new weights
        self.weights = newWeights
    
    ###########################################################################
    #                                                                         #
    # Name:       setBias                                                     #
    # Parameters: newBias - the new bias                                      #
    # Reutrns:    N/A                                                         #  
    # Notes:      Setter method, newBias should be an integer or float        #
    #                                                                         #
    ###########################################################################
    
    def setBias(self, newBias):
        
        self.bias    = newBias 
        
    ###########################################################################
    #                                                                         #
    # Name:       getOutput                                                   #
    # Parameters: N/A                                                         #
    # Reutrns:    N/A                                                         #  
    # Notes:      Getter method, public access to private variable called     #
    #             output                                                      #
    #                                                                         #
    ###########################################################################
        
    def getOutput(self):
        
        return self.output
