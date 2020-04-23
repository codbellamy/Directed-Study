import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt

#manually sets the seed to 1 so I get the same results as the test
torch.manual_seed(27)

def wordsToNums(sequence, vocabDict):

    #Converts the english sequence into numbers using their indexes in the 
    #conversion dictionary
    nums = [ vocabDict[word] for word in sequence]

    return torch.tensor(nums, dtype = torch.long)

dataset = [ ("the sun did not shine".split(), "ART NOU VER ADV VER".split()),
            ("it was too wet to play".split(), "PRO VER ADV ADJ PRE VER".split()),
            ("so we sat in the house".split(), "ADV PRO VER PRE ART NOU".split()),
            ("all that cold cold wet day".split(), "ADJ ART ADJ ADJ ADJ NOU".split())]

vocab = dict()
tags  = dict()
for sentence, labels in dataset:
    for word in sentence:
        if word not in vocab:
            #enumerates the words in the order that they appear
            vocab[word] = len(vocab)

    for label in labels:
        if label not in tags:
            #enumerate the labels in the order that they appear
            tags[label] = len(tags)

reverseTags = dict()

for key in tags:
    reverseTags[tags[key]] = key

#These in practice can be 32 or 64 ish but they are small for the sake of simplicity
EMBEDDING_DIMS = 32
HIDDEN_DIMS = 32

#Creates the model
class Model(nn.Module):

    #class constructor
    def __init__(self, embedDims, hiddenDims, vocabSize, tagSize):

        #Calls the superclass constructor for torch setup
        super(Model, self).__init__()

        #An instance variable to store the word embeddings in the vocab set 
        #from our fixed dictionary
        self.wordEmbedding = nn.Embedding(vocabSize, embedDims)

        #the LSTM will take our word embeddings as input and output the hidden
        #states with dimension specified with hiddenDims
        self.lstm = nn.LSTM(embedDims, hiddenDims)

        #A linear layer that will interpret our LSTM layer output
        self.fc = nn.Linear(hiddenDims, tagSize)

    #defines a forward pass
    def forward(self, sentence):

        #gets the output for the word embeddings
        embeds = self.wordEmbedding(sentence)

        #gets the output from the lstm and discards the hidden states
        lstmOut, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        #gets the outputs of the output layer
        linearOut = self.fc(lstmOut.view(len(sentence), -1))

        #gets the confidence scores by running the output thorugh softmax
        confidence = f.log_softmax(linearOut, dim = 1)

        #Returns the softmax output
        return confidence

#creates an instance of the model
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, len(vocab), len(tags))

#chooses the cost function
cost = nn.NLLLoss()

#an instance of the optimizer
optimizer = opt.SGD(model.parameters(), lr = 0.1)

#trains model
for epoch in range(100):

    for sentence, label in dataset:

        #clear the gradient so it doesn't accumulate
        model.zero_grad()

        #preprocesses data
        inputVec = wordsToNums(sentence, vocab)
        targets = wordsToNums(label, tags)

        #runs a forward pass
        scores = model(inputVec)

        #computes loss and backpropagates
        loss = cost(scores, targets)
        loss.backward()
        optimizer.step()
        
#see results after training
with torch.no_grad():

    for sentence, _ in dataset:

        print('INPUT SENTENCE:', ' '.join(sentence))

        inputs = wordsToNums(sentence, vocab)
        scores = model(inputs)
        
        pred = []
        for word, score in zip(sentence, scores):
            print(word, 'is a', reverseTags[int(torch.argmax(score))])
        
        print('--------------------------------------------')