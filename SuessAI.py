
#%%########################## IMPORT DEPENDENCIES #############################

#Imports the core torch module
import torch
#Import the torch neural network module
import torch.nn            as nn
#Import torch functional modules
import torch.nn.functional as f
#Imports torch optimizer
import torch.optim         as optim

#Imports the ability to work with matricies easily
import numpy               as np

#%%############################### CONSTANTS ##################################

EMBEDDING_DIMS = 64
HIDDEN_DIMS    = 64

NUM_EPOCHS     = 40
LEARNING_RATE  = 0.001

TEST_WORD      = 'the'

VERBOSE        = True

#%%############################ HELPER FUNCTIONS ##############################

###############################################################################
# Name:   embed
# Param:  wordList      - the list of words to embed
#         embeddingDict - the dictionary to use to embed the words
# Return: a torch tensor of the embedded word list
# Notes:  prepares an input sentence, split over spaces, to be passed into the
#         model
def embed(wordList, embeddingDict):

    #embeds each word in the word list
    embeds = [ embeddingDict[word] for word in wordList ]

    #converts the embedding to a tensor and returns it
    return torch.tensor(embeds, dtype = torch.long)


###############################################################################
# Name:  train
# Param: model        - the model we are training
#        trainingSet  - the data set we are using to train
#        numEpochs    - the number of epochs we are going to train for
#        learningRate - the learning rate we are going to train with
def train(model, trainingSet, numEpochs, learningRate):

    #chooses the NLL loss function 
    cost = nn.NLLLoss()
    #creates an instance of the optimizer
    optimizer = optim.Adam(model.parameters(), lr = learningRate)

    #trains model for the number of epochs to train for
    for epoch in range(1, numEpochs+1):

        #gets the inputs and labels in the training data
        for ins, outs in trainingSet:

            #Resets the gradients
            model.zero_grad()

            #embeds the input vector
            inputVec  = embed(ins, wordEmbeddings)
            #embeds the target vector
            targetVec = embed(outs, wordEmbeddings)

            #runs a forward pass through the model, stores the prediction
            pred = model(inputVec)

            #computes loss for this pass through the network
            loss = cost(pred, targetVec)

            #Backpropagates error in the network
            loss.backward()
            optimizer.step()

        #Prints the epoch and the loss every 10 epochs
        if VERBOSE:
            print('EPOCH {}/{}||LOSS = {}'.format(epoch,numEpochs,loss.item()))

###############################################################################
# Name:   runModel
# Param:  testWord - the word we are going to run through
#         numWords - how long our sequence will be
# Return: sentence - the sequence we will output
# Notes:  runs a forward pass through the network for a test word
def runModel(testWord, numWords):

    #Initializes the sentence we will start
    sentence = testWord

    #Prepares to run the trained model
    with torch.no_grad():

        #the word we want to test
        newWord = [testWord]

        #outputs the length of the sequence
        for _ in range(numWords):

            #pre processes the input vector
            inputVec = embed(newWord, wordEmbeddings)
            
            #makes a model prediction
            pred = model(inputVec)
            
            #gets what the predicted word is
            newWord.append(embeddingToWords[int(torch.argmax(pred[-1]))])
            
            #concats to the sentence
            sentence += ' ' + newWord[-1]
    
    #returns the predicted sentence
    return sentence

#%%########################### PRE-PROCESSES DATA #############################

#sets the seed to 27, my favorite number
torch.manual_seed(27)
np.random.seed(27)

#Initializes a list for the training data
catdata = []

#Initializes a word embedding dictionary
wordEmbeddings = dict()

#Loads in the data file
dataFile = open('./suess.txt')

#Loops thorugh each line in the data file
for line in dataFile:

    #Strips off the new line chracter
    line = line.strip()

    #Only lowercase words
    line = line.lower()

    #Replaces punctuation with an empty string,
    #effectively removing it from consideration in the model
    line = line.replace('.', '')
    line = line.replace('!', '')
    line = line.replace('?', '')
    line = line.replace(',', '')
    line = line.replace('"', '')
    
    #Loops through each word in the sentence
    for word in line.split():

        #If the word embedding does not exist
        if word not in wordEmbeddings:
            #Adds it to the dictionary
            wordEmbeddings[word] = len(wordEmbeddings)

    #Creates the input sequence as all words in the sentence except the
    #last word
    inputSequence  = line.split()[:-1]

    #creates the target sequence as all the words in the sentence except
    #the first word
    targetSequence = line.split()[1:]
    
    #If there is data in the input and target sequence
    if len(inputSequence) != 0 and len(targetSequence) != 0:

        #Appends the input and target to the data list
        catdata.append((inputSequence, targetSequence))

#Closes the data file
dataFile.close()

#Initializes a dictionary to turn from embeddings to words
embeddingToWords = dict()

#Loops through each word in the vocab set
for key in wordEmbeddings:

    #Reverses the word embedding dict
    embeddingToWords[wordEmbeddings[key]] = key

#%%######################### CREATION OF THE MODEL ############################

#The Model class, defines model structure
class Model(nn.Module):

    ###########################################################################
    # Name:  __init__
    # Param: embeddingDims
    #        hiddenDims
    #        vocabSize
    # Notes: class constructor, defines model instance variables and runes
    #        super class constructor
    def __init__(self, embeddingDims, hiddenDims, vocabSize):

        #Runs superclass constructor
        super(Model, self).__init__()

        #Stores the model word embeddings as an instance variable
        self.embeddings = nn.Embedding(vocabSize, embeddingDims)

        #the lstm layer
        self.lstm = nn.LSTM(embeddingDims, hiddenDims)

        #the lstm layer feeds into a single linear output layer so we
        #can run log softmax on the output
        self.linear = nn.Linear(hiddenDims, vocabSize)

    ###########################################################################
    # Name:   forward
    # Param:  inputSequence - the list of inputs to the model
    # Return: ouput         - the output of the model
    # Notes:  a forward pass through the model
    def forward(self, inputSequence):

        #Gets the word embeddings from the input sequence
        embeds = self.embeddings(inputSequence)

        #gets the ouput from the lstm layer, discards the hidden states
        lstmOut, _ = self.lstm(embeds.view(len(inputSequence), 1, -1))

        #the output from the linear layer to reshape our data
        linear = self.linear(lstmOut.view(len(inputSequence), -1))

        #puts the ouput from the linear layer through a log softmax activation
        #function
        output = f.log_softmax(linear, dim = 1)

        #gets the output from the last linear layer
        return output

#creates an instance of the model
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, len(wordEmbeddings))

#trains the model
train(model, catdata, NUM_EPOCHS, LEARNING_RATE)

#prints the model output
print()
word = TEST_WORD
for i in range(1, 6):
    r = np.random.randint(6, 10)

    line = runModel(word, r)
    print(' '.join(line.split()[:-1]))
    word = line.split()[-1]

    if i % 5 == 0:
        print()