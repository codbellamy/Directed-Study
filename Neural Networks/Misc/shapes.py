###############################################################################
#                                                                             #
#                                   Shapes.py                                 #
#                                                                             #
#                                   J. Steiner                                #
#                                                                             #
###############################################################################

#Imports the ability to work with matricies
import numpy as np

#Imports the ability to work with images as matricies
import cv2 as cv

#Imports reference to neural network classes
from Network import Network


#Creates the network
digits = Network([256, 8, 1])

#Defaults lists of input and response vectors
data = []
labels = []

#Loops through the numbers 0 and 1
for i in range(2):
    
    #Loops through all files for 0's and 1's
    for j in range(1, 56):
        
        #Reads in the image as a grayscale image
        img = cv.imread('./Digits/'+str(i)+'/'+str(i)+'_'+str(j)+'.png', 
                        cv.COLOR_BGR2GRAY)
        
        #Flattens the image so it is one continuous vector
        img = img.flatten()

        #Appends the input vector array to the data list
        data.append(img)        
        
        #calculates the value of the response vector
        label = [0.0]
        if i == 0:
            label = [1.0]
        
        #Appends the response vector array to the label list
        labels.append(np.array(label))

#Calculates average initial cost
cost = 0
for i in range(110):
    
    cost += digits.cost(data[i], labels[i]) 
cost /= 110

#Displays initial cost
print('Initial Cost:', cost)

#Trains the network for 100 epochs using a random sample of 10 items
digits.train(data, labels, maxEpochs=100, batchSize=100)

#Calculates average cost
cost = 0
for i in range(110):
    
    cost += digits.cost(data[i], labels[i])
cost /= 110

#Displays new cost
print('Post Training Cost:', cost)

#Prints the succes rate
print('Success Rate:', digits.test(data, labels) / 110)


################################ SAMPLE OUTPUT ################################

#Initial Cost: 4.134261626266017
#Post Training Cost: 0.41556165198956296
#Success Rate: [0.89090909]