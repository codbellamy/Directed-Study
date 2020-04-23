###############################################################################
#                                                                             #
#                                 RNN_Test.py                                 #
#                                 J. Steienr                                  #
#                           Adapted from online tutorial                      #
#                                                                             #
###############################################################################

#%%############################ LOADS LIBRARIES ###############################

#Imports the torch neural network module
from torch import nn

#Imports the abiliy to work with matricies more easily
import numpy as np

#Imports general torch functionality
import torch

#Imports a dataset
import CAT

#%%########################### DEFINES CONSTANTS ##############################
NUM_EPOCHS = 100
LR         = 0.01

#%%############################ HELPER FUNCTIONS ##############################

###############################################################################
# Name:   oneHotEncode
# Param:  sequence  - the sentence we are going to one hot encode
#         dictSize  - the amount of characters we are working with
#         seqLen    - the length of the sentences
#         batchSize - the amount of data points to encode
# Return: the one hot encoded vector
def oneHotEncode(sequence, dictSize, seqLen, batchSize):

    #Defaults a matrix of zeros
    vectors = np.zeros((batchSize, seqLen, dictSize), dtype = np.float32)

    #Loops through each data point in the entire dataset
    for i in range(batchSize):

        #Loops through each character in the datapoitn
        for j in range(seqLen):

            #Encodes the character necessary with a 1, the rest are zeros
            vectors[i, j, sequence[i][j]] = 1

    #Returns the encoded vectors
    return vectors

###############################################################################

###############################################################################
# Name:   predict
# Param:  model     - the model we are passing into the prediction function
#         character - the input into the prediction function
# Return: the output from the model
# Notes:  preprocesses validation data and runs the NN through it
def predict(model, character):

    #Converts the input from characters to numbers
    character = np.array([[charToInt[c] for c in character]])

    #Converts the input from a matrix of numbers to a matrix of one hot encoded
    #vectors
    character = oneHotEncode(character, dictSize, character.shape[1], 1)

    #Converts from a numpy matrix to a torch tensor
    character = torch.from_numpy(character)
    
    #Gets the output from the model
    out, hidden = model(character)

    #Gets the confidences based on the softmax function
    prob = nn.functional.softmax(out[-1], dim=0).data

    #Gets teh index of the character
    charIndex = torch.max(prob, dim=0)[1].item()

    #Returns the predicted character
    return intToChar[charIndex], hidden

###############################################################################

###############################################################################
# Name:   run
# Param:  model  - the model we are running through
#         outLen - the length of the output desired
#         data   - the input into the network
# Return: the sentence predicted
# Notes:  a forward pass of the network using validataion data
def run(model, outLen, data):

    #Makes the model in eval mode
    model.eval()

    #Gets a list of characters in the input data
    chars = [ch for ch in data]

    #Finds how many characters still need to be filled in
    size = outLen - len(chars)

    #Loops through the remainder of the characters we need to fill in
    for _ in range(size):

        #Predicts the next character
        char, h = predict(model, chars)

        #Appends the predicted character
        chars.append(char)

    #Gets the ouput as one string
    return ''.join(chars)

###############################################################################

#%%########################## DATA PRE-PROCESSING #############################

#Defines our dataset, a small sample space of sentences for the sake of this 
#example
data = ['hey how are you', 'good i am fine', 'have a nice day']

#A grouping of unique characters used to construct the sentences above
chars = set(''.join(data))

#A dictionary that maps integers to characters in our set of characters
intToChar = dict(enumerate(chars))

#A dictionary that maps our set of characters to integers (the reverse of the)
#integer to character 
charToInt = {char: index for index, char in intToChar.items()}

#Finds the length of the longest sentence
maxLen = len(max(data, key=len))

#Loops through each sentence in the dataset
for i in range(len(data)):
    data[i] = data[i].lower()
    #Pads sentences that are not the longest sentence with whitespace until they
    #are the same length as the longest sentence
    while len(data[i]) < maxLen:
        data[i] += ' '

#The target sequence will always be one time unit ahead of the input sequence

#Initializes the input and target sequence lists
inputSeq  = []
targetSeq = []

#Loops through each sentence in the dataset
for datum in data:

    #Removes last character for the input sequence
    inputSeq.append(datum[:-1])

    #Removes the first character for the target sequence
    targetSeq.append(datum[1:])


#Loops through the range of inicies in the dataset
for i in range(len(data)):
    inputSeq[i]  = [charToInt[character] for character in inputSeq[i] ]
    targetSeq[i] = [charToInt[character] for character in targetSeq[i]]

#Defines the amount of characters in the charater to 
dictSize  = len(charToInt)
#Defines the length of each sentence
seqLen    = maxLen - 1
#Defines the size of batches we will train with as the entire training set
batchSize = len(data)

#One hot encodes the input sequence
inputSeq  = oneHotEncode(inputSeq, dictSize, seqLen, batchSize)

#Converts the input and target sequences to torch tensors
inputSeq  = torch.from_numpy(inputSeq)
targetSeq = torch.Tensor(targetSeq)

#%%####################### NEURAL NETWORK DEFINITION ##########################

class Net(nn.Module):

    ###########################################################################
    # Name:  __init__
    # Param: inputSize  - the size of the input into the RNN
    #        outputSize - the size of the output from the fully connected net
    #        rnnOutSize - the size of the output from the RNN
    #        numLayers  - the number of layers in the rnn
    def __init__(self, inputSize, outputSize, rnnOutSize, numLayers):

        #Runs the nn super constructor
        super(Net, self).__init__()

        #Defines the rnn output size as an instance variable
        self.rnnOutSize = rnnOutSize

        #Defines the number of layers within the rnn
        self.numLayers  = numLayers
        
        #Defines an rnn layer with tanh activation functions
        self.rnn = nn.RNN(inputSize, rnnOutSize, numLayers, batch_first = True)

        #A fully connected layer
        self.fc = nn.Linear(rnnOutSize, outputSize)

    ###########################################################################

    ###########################################################################
    # Name:   initHidden
    # Param:  batchSize - the size of the hidden layer to create
    def initHidden(self, batchSize):
        #Returns a matrix of zeros in the shape of our hidden states
        return torch.zeros(self.numLayers, batchSize, self.rnnOutSize)
    ###########################################################################

    ###########################################################################
    # Name:   forward
    # Param:  x      - the input to the network
    # Return: out    - the output of the network
    #         hidden - the hidden states of the RNN
    # Notes:  a forward pass of the RNN
    def forward(self, x):

        #Gets a dimension of our hidden state
        batchSize = x.size(0)

        #Defaults the hidden states
        hidden = self.initHidden(batchSize)

        #Gets the output and the hidden states
        out, hidden = self.rnn(x, hidden)

        #Gets output
        out = out.contiguous().view(-1, self.rnnOutSize)
        out = self.fc(out)

        #Gets output
        return out, hidden

    ###########################################################################

###############################################################################
 
#%%############################ MODEL EXECUTION ###############################

#Creates a NN model
net = Net(dictSize, dictSize, 12, 1)

#Defines our cost function
cost = nn.CrossEntropyLoss()

#Defines our optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

#Training loop
for epoch in range(1, NUM_EPOCHS + 1):

    #Resets the gradients at the start of the training loop
    optimizer.zero_grad()
    
    #Gets the output from the network forward pass
    output, hidden = net(inputSeq)

    #Calculates the loss for this loop
    loss = cost(output, targetSeq.view(-1).long())

    #Calculates the gradients based on the loss
    loss.backward()

    #Updates gradients based on the loss
    optimizer.step()

    #Prints the loss and epoch number every 10 epochs as a diagnostic
    if epoch % 10 == 0:
        print('Epoch: {}/{}............'.format(epoch, NUM_EPOCHS), end='')
        print("Loss: {:.4f}".format(loss.item()))


#Prints the output
print(run(net, maxLen, 'h'))