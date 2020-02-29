import numpy  as np
import pandas as pd
import random
import math

#Enumeration of layer types
FULLY_CONNECTED = 0

#Enumeration of activation functions
SIGMOID         = 0
TANH            = 1
RELU            = 2
SOFTMAX         = 3
SWISH           = 4

#SIGMOID ACTIVATION FUNCTION
def sigmoid(z, derivative = False):
    if not derivative:
        return 1.0 / ( 1 + np.exp(-z) )
    else:
        return sigmoid(z) * ( 1 - sigmoid(z) )
    
#TANH    ACTIVATION FUNCTION
def tanh(z, derivative = False):
    if not derivative:
        return ( np.exp(2 * z) - 1 ) / ( np.exp(2 * z) + 1 )
    else:
        return 1 - ( tanh(z) ** 2)
    
#RELU    ACTIVATION FUNCTION
def relu(z, derivative = False):
    if not derivative:

        return np.minimum(np.maximum(0, z), 6)
    
#SOFTMAX ACTIVATION FUNCTION
def softmax(z, derivaitve = False):
    return np.exp(z) / sum( np.exp(z) )

#SWISH   ACTIVATION FUNCTION
def swish(z, derivative = False):
    if not derivative:
        return z * sigmoid(z)
    else:
        return sigmoid(z) + np.dot( z, sigmoid(z, True))
    
#%%######################## NETWORK CLASS DEFINITION ##########################
class Net4():
    
    ######################### NETWORK CONSTRUCTOR ########################
    
    def __init__(self):
        #Contains the list of layers as a Net4 instance variable
        #Defaults the network to an empty list
        self.network = []
    
    ######################################################################
    
    ############################ ADDING LAYERS ###########################
    #param: layerType  - what type of layer the layer will be
    #       numInputs  - the number of inputs that they layer will have
    #       numOutputs - the number of outputs that the layer will return
    #       activation - the activation function that will be used
    
    def addLayer( self, layerType, numInputs, numOutputs, activation ):
        
        #Determines which layer will added
        
        #If the layer is to be fully connected
        if layerType == FULLY_CONNECTED:
            
            #Appends a fully connected layer with the required specifications
            self.network.append( Fully_Connected( numInputs, numOutputs, 
                                                  activation ) )
            
    ######################################################################
    
    ########################### QUERY FUNCTION ###########################
    #param:  x - the inpput vector used for the forward pass
    #return: x - the output for the final layer in the network
    
    def query(self, x):
        
        x = np.array(x, ndmin=2).T
        
        #Loops through each layer in the network
        for layer in self.network:
        
            #Sets x equal to the new output of the network
            x = layer.getOutput(x)

        #Returns the final output of the network
        return x
    
    ######################################################################
    
    ####################### BACKPROPAGATION FUNCTION #####################
    # param: data   - a single data point x values
    #        actual - the labels for the data point
    def gradDescent(self, data, actual, eta):
        
        #Converts the inputs and targets into two dimensional arrays
        inputs      = np.array(data, ndmin=2).T
        targets     = np.array(actual, ndmin=2).T
        
        #Defaults a list of activations from different levels of the network
        activations = [ inputs ]
        
        #Loops through the range of layers for the forward pass
        for l in range(len(self.network)):

            #Calculates the output of each layer
            z = self.network[l].getOutput(activations[l])
            
            #Gets the output of each layer after it has been run through the
            #sigmoid function
            activations.append(z)
        
        
        #Calculates the error in the output layer
        outputErrors = targets - activations[-1]
        
        #Stores the output error in a list of errors
        errorWeight  = [outputErrors]
        
        #Loops through the range of layers for the backward pass
        for l in range(1, len(self.network)):
        
            #Calculates the error for each hidden layer
            errorW = np.dot(self.network[-l].getWeights().T, errorWeight[l-1]) 
            
            #Appends the error term to the list of errors
            errorWeight.append(errorW)
        
        #Loops through the range of layers to update the weights
        for l in range(1, len(self.network)+1):
            
            #How much error there was in the calculation
            dCost    = (errorWeight[l-1] * activations[-l] * 
                       (1.0              - activations[-l]  ))
            
            #How much to change the weights by
            dWeights = np.transpose(activations[-(l+1)])

            #Updates each layer's weights
            self.network[-l].setWeights(self.network[-l].getWeights() + (eta * np.dot(dCost, dWeights)))
        
    ######################################################################
            
    ########################## TRAINING FUNCTION #########################
    #param: trainData   - the x values for the training set
    #       trainLabels - the labels for the training data
    #       maxEpochs   - how many iterations we want to train for
    #       batchSize   - the size of the random sample we generate
    #       verbose     - whether or not to print diagnostics
    #       testData    - test input vector
    #       testLabels  - test labels vector
    
    def train(self, trainData, trainLabels, maxEpochs, batchSize, eta = 1e-2, 
              verbose = False, testData = None, testLabels = None):
        
        #Generates a random sample of indicies without replacement
        #corresponding to the number of random selections in the batch
        randomSample = random.sample(range(len(trainData)), batchSize)
        
        #Loops through each epoch
        for epoch in range(maxEpochs):
            
            #If we are in verbose mode
            if verbose:
                
                #Prints the epoch
                print('EPOCH:', epoch+1)
            
            #For each input and output in the random sample of the training
            #data and training labels
            for index in randomSample:
                
                #Back propagates the error with respect to each input in the
                #batch using gradient descent
                self.gradDescent(trainData[index], trainLabels[index], eta)
                
            #If we are in verbose mode a test set was passed in
            if verbose and testData and testLabels:
                
                #Tests the network
                correct = np.asarray(self.test(testData, testLabels, verbose))
                
                #Prints the success rate for the network
                print ('Success Rate = ' + str((correct.sum() / correct.size) 
                                         * 100) + '%')
            
    ######################################################################
    
    ############################ TEST FUNCTION ###########################
    # param:  testInputs - the input vector of the test set
    #         testLabels - the labels of each data point in the test set
    # return: whether or not for each test data point, the prediction was
    #         correct
    
    def test(self, testData, testLabels, verbose = False):
    
        #An empty lisst of if the given prediction was correct or not
        predicted = []
        cost      = 0
        
        #Loops through the input and label in the test set
        for index in range(len(testData)):
            
            #Query's the network to get the output
            outputs = self.query(testData[index]).flatten()
            
            #Calculates the cost
            cost += -((testLabels[index] * np.log(outputs)) + 
                     (1 - testLabels[index] * np.log(1 - outputs)))
            
            #If we have only 1 output neuron
            if len(outputs) == 1:

                #Treats the activation as a probability of success
                label = (outputs >= 0.5)
                
                if label == testLabels[index]:
                    predicted.append(1)
                else:
                    predicted.append(0)
                
            #Otherwise
            else:
                #Chooses the most confident answer
                label = np.argmax(outputs)
            
                #If the prediciton was correct
                if label == testLabels[index].argmax():
                    
                    #Appends that this was a correct prediction
                    predicted.append(1)
                
                #If the prediction was incorrect
                else:
    
                    #Appends that this was an incorrect prediction
                    predicted.append(0)
                
        if verbose:
                
            print('COST:', (sum(np.abs(cost)) / len(predicted)))
        #Returns the list of correct or incorrect predictions
        return predicted
    
    ######################################################################

###############################################################################

#%%################# FULLY CONNECTED LAYER CLASS DEFINITION ###################
class Fully_Connected:
    
    #################### FULLY CONNECTED LAYER CONSTRUCTOR ####################
    
    def __init__(self, numInputs, numOutputs, activation):
        
        #Initializes random weights and biases
        
        #Initializes an optimal weight. a normal distribution centered at 
        #1 / the square root of the layer size
        #we subtact -0.5 to have negative and positive weights
        self.weights = np.random.normal(0.0, pow(numOutputs, -0.5), 
                                       (numOutputs, numInputs))
        
        #Initializes the activation variable, this stores the enumeration
        #of which activation function to use
        self.activation = activation
        
    ######################################################################
    
    ########################## SETS NEW WEIGHTS ##########################
    
    def setWeights(self, weights):
        self.weights = weights
        
    ######################################################################
    
    ######################### GETS LAYER WEIGHTS #########################
    
    def getWeights(self):
        return self.weights
    
    ######################################################################
    
    ########################## GETS LAYER OUTPUT #########################
    #param: x - the input vector used to get the output vector
    
    def getOutput(self, x):
        
        #Takes the dot product of the weights and the input and adds the bias
        z = np.dot(self.weights, x)
        
        #Chooses which activation function to use based on the activation
        #function passed in the constructor
        if   self.activation == SIGMOID:
            return sigmoid(z)
        elif self.activation == TANH:
            return tanh(z)
        elif self.activation == RELU:
            return relu(z)
        elif self.activation == SOFTMAX:
            return softmax(z)
        elif self.activation == SWISH:
            return relu(z)
        #No activation function given
        else:
            return z
        
    ######################################################################
    
###############################################################################

#%%############################## DATA LOADER #################################
    
class Data:
    
    def __init__(self, filePath, responseName, 
                 testPercentage = 10, scale = True):
        
        self.df = pd.read_csv(filePath)
    
        self.trainPredictors = []
        self.trainResponse   = []
        
        self.testPredictors  = []
        self.testResponse    = []
        
        labelIndex = list(self.df.columns).index(responseName)
        
        mins = []
        maxes = []
        
        cols = list(self.df.columns)
        
        for col in cols[:labelIndex] + cols[labelIndex+1:]:
            
            locMin = float(min(self.df[col]))
            locMax = float(max(self.df[col]))
            if locMax == 0:
                locMax += 0.01
            
            mins.append(locMin)
            maxes.append(locMax * 0.99)
        
        #Loads in the training set
        trainFile = open(filePath, 'r')
        trainList = trainFile.readlines()
        trainFile.close()
                
        mins = np.asfarray(mins)
        maxes = np.asfarray(maxes)
        
        for datum in trainList[1:]:
    
            #Takes all the values in the row and splits them by the comma
            rowVals = datum.split(',')
            
            #Takes the values except for the labels
            vals = np.asfarray(rowVals[:-1])
            
            #Scales the image
            x = ((vals - mins) / maxes) + 0.01
            
            #Takes the label of the data and scales it
            y = np.zeros(2) + 0.1
            y[int(rowVals[labelIndex])] = 0.99
            
#            y = int(rowVals[labelIndex])
            
            if random.randrange(0, 100) >= 10:
                #Appends the scaled label and datum to the inputs and labels list
                self.trainPredictors.append( x )
                self.trainResponse.append(y)
            else:
                self.testPredictors.append(x)
                self.testResponse.append(y)
                
    def getTrainPredictors(self):
        
        return self.trainPredictors
    
    def getTrainResponse(self):
        
        return self.trainResponse
    
    def getTestPredictors(self):
        
        return self.testPredictors
    
    def getTestResponse(self):
        
        return self.testResponse
            
    
###############################################################################
    
    
print('LOADING DATA')
data = Data('./USvideos.csv', 'popular', testPercentage=20)
print('TRAINING MODEL')

def buildNetwork(networkSize):
    nat = Net4()
    for l in range(1, len(networkSize)):
    
        nat.addLayer(FULLY_CONNECTED, networkSize[l-1], networkSize[l], SIGMOID)
   
    return nat

nat = buildNetwork([8, 40, 2])
    
nat.train(data.getTrainPredictors(), data.getTrainResponse(), 
          maxEpochs = 50, batchSize = 10000, eta = 5e-2, verbose = True,
          testData = data.getTestPredictors(),testLabels= data.getTestResponse())