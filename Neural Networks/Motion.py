###############################################################################
#                                                                             #
#                                 Motion.py                                   #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#%%################## IMPORTS NECESSARY LIBRARIES/MODULES #####################

#Imports the ability to load/display video
import cv2 as cv

#Imports the ability to perform matrix operations on loaded video
import numpy as np

###############################################################################

#%%############################### CONSTANTS ##################################

#A constant that determines what distance is considered a meaningful change in
#color
THRESHOLD    = 10

#A constant that determines the size of the resizing kernel
KERNEL_SIZE  = 5

#A constant that determines the stride of the resizing algorithm
STRIDE       = 10

#Constants that predecide the real size of the frames loaded in
FRAME_WIDTH  = 800 
FRAME_HEIGHT = 600

#A constant that determines the size of the cirlce we draw at the center of
#motion
MOTION_PEG   = 10 

#A constant used for linear interpolation between two points
T            = 0.1

#A constant for moving x and y coordinates from a smaller image to a larger
#image
SIZE_SCALE   = 2.1

###############################################################################

#%%############################### LOAD DATA ##################################

#Loads the video file
video = cv.VideoCapture('./Sail_Boat_Test/sail_boat_small.mp4')

#Loads in the first frame of the video:
#unpacking the frame (newFrame), and if the read was successful (success)
success, newFrame = video.read()

###############################################################################

#%%############################ HELPER FUNCTIONS ##############################

###############################################################################
# Name:   maxPool
# Param:  inputImage - the original image to be resized
#         kernelSize - the size of the box to maximum pixels of
#         stride     - the amount that box moves each iteration
# Return: pooled     - the resized image
# Notes:  A resizing algorithm for images, taking the maximum of a given set
#         of pixels

def maxPool(inputImage, kernelSize, stride):

    #Calculates the dimensions of the resized image
    nextHeight = ((inputImage.shape[0] - kernelSize)//stride) + 1
    nextWidth  = ((inputImage.shape[1] -  kernelSize)//stride) + 1

    #Defaults the resized image to a totally black image
    pooled = np.zeros((nextHeight, nextWidth, 3))

    #Defines a counter for which row in the resized image is being adjusted
    rowCtr = 0

    #Loops through all rows in the original image
    for row in range(0, inputImage.shape[0] - kernelSize, stride):

        #Defines a counter for which column in the resized image is being 
        #adjusted
        colCtr = 0
        
        #Loops through all columns in the original image
        for col in range(0, inputImage.shape[1] - kernelSize, stride):

            #Averages all the pixels within the kernel
            pooled[rowCtr][colCtr] = np.max(inputImage[row:row+kernelSize, \
                                                       col:col+kernelSize])

            #Increments the resized image column counter
            colCtr += 1

        #Increments the resized image row counter
        rowCtr += 1

    #Returns the resized image
    return pooled

###############################################################################

###############################################################################
# Name:   lerp
# Param:  v0    - the original value
#         v1    - the new value
#         t     - a constant, what proportion of the interpolated value is the
#                 new value
# Return: lerpV - the linearly interpolated value
# Notes:  linearly interpolates between two points

def lerp(v0, v1, t):
    #Linear interpolation function
    return (1-t)*v0 + t*v1

###############################################################################

###############################################################################

#%%############################ HELPER CLASSES ################################

################################# BLOB CLASS ##################################
# Notes: Used to track desired objects using bounding boxes

class Blob:

    #A threshold for how far away a desired pixel needs to be before it is
    #to be considered a new blob
    DISTANCE_THRESHOLD = 350

    #A buffer on all edges of the blob to catch pixels that may not be tracked
    PX_BUFFER          = 20

    ###########################################################################
    # Name:  __init__
    # Param: x - the initial x value of the blob
    #        y - the initial y value of the blob
    # Notes: class constructor

    def __init__(self,x, y):

        #Sets the maximum and minimum x values equal to the initial x value
        self.maxX, self.minX = x, x

        #Sets the maximum and minimum y values equal to the initial y value
        self.maxY, self.minY = y, y

    ###########################################################################

    ###########################################################################
    # Name:   isNear
    # Param:  x - the x value being considered as 'near'
    #         y - the y value being considered as 'near'
    # Return: True/False
    # Notes:  returns whether or not the distance from the center of the blob
    #         to the pixel considered as greater than some distance threshold
    def isNear(self,x, y):

        #Calculates the center x value of the blob
        centerX = (self.maxX-self.minX) // 2

        #Calculates the center y value of the blob
        centerY = (self.maxY-self.minY) // 2

        #If the distance from the center to the pixel considred is less than
        #some distance threshold
        if (centerX - x)**2 + (centerY - y)**2 <= self.DISTANCE_THRESHOLD**2:
            #Return True - the pixel is near to the blob
            return True

        #Otherwise
        #Return False - the pixel is not near the blob
        return False
    
    ###########################################################################

    ###########################################################################
    # Name:  add
    # Param: x - the x value added to the blob
    #        y - the y value added to the blob
    # Notes: updates the bounds of the blob area based on a new pixel

    def add(self, x, y):

        #The new max x is whichever is larger: the current max x or the new x
        self.maxX = max(self.maxX, x)

        #The new min x is whichever is smaller: the current min x or the new x
        self.minX = min(self.minX, x)

        #The new max y is whichever is larger: the current max y or the new y
        self.maxY = max(self.maxY, y)

        #The new min y is whichever is smaller: the current min y or the new y
        self.minY = min(self.minY, y)

    ###########################################################################

    ###########################################################################
    # Name:   minCorner
    # Param:  lerpMinX - the previous linearly interpolated min x
    #         lerpMinY - the previous linearly interpolated min y
    # Return: the topleft corner of the blob bounding box
    # Notes:  calculates where to draw the topleft corner of the blob bounding
    #         box

    def minCorner(self, lerpMinX, lerpMinY):

        #Updates the minimum x based on linear interpolation
        self.minX = lerp(lerpMinX, self.minX, T)

        #Updates the minimum y based on linear interpolation
        self.minY = lerp(lerpMinY, self.minY, T)

        #Returns the topleft corner of the blob bounding box
        return (int((self.minX-self.PX_BUFFER)*SIZE_SCALE), 
                int((self.minY-self.PX_BUFFER)*SIZE_SCALE))

    ###########################################################################

    ###########################################################################
    # Name:   maxCorner
    # Param:  lerpMaxX - the previous linearly interpolated max x
    #         lerpMaxY - the prevoius linearly interpolated max y
    # Return: the bottom right corner of the blob bounding box
    # Notes:  returns where  to draw the bottom right corner of the blob
    #         bounding box

    def maxCorner(self, lerpMaxX, lerpMaxY):

        #Updates the max x based on linear interpolation
        self.maxX = lerp(lerpMaxX, self.maxX, T)

        #Updates the max y based on linear interpolation
        self.maxY = lerp(lerpMaxY, self.maxY, T)

        #Returns the bottom right corner of the blob bounding box
        return (int((self.maxX+self.PX_BUFFER)*SIZE_SCALE), 
                int((self.maxY+self.PX_BUFFER)*SIZE_SCALE))

###############################################################################

#%%############################# DETECTION LOOP ###############################

#Defaults a 'previous frame' to be an all black screen
prevFrame = maxPool(np.zeros(newFrame.shape), KERNEL_SIZE, STRIDE)

#Stores the original frame loaded
originalFrame = newFrame

#Resizes the frame
newFrame = maxPool(newFrame, KERNEL_SIZE, STRIDE)

#Defaults the linearly interpolated x and y of average motion
lerpX, lerpY = 0, 0

#Initialize a list of linearly interpolated blob objects
lerpBlobs = []

#While reading the video is successful
while success:

    ############################# LOGIC #######################################

    #Initialize a list of blob objects
    blobs = []

    #Defaults the processed frame to be a totally black screen
    processedFrame = np.zeros(newFrame.shape)

    #Defaults a count for moving pixels
    countX, countY = 1, 1

    #Defaults a sum of the location of moving pixels
    locX, locY = 0, 0

    #Loops through the row indicies
    for row in range(0, newFrame.shape[0], 1):

        #Loops through the column indicies
        for col in range(0, newFrame.shape[1], 1):

            #The distance in the red colorspace
            disRSqr = int((newFrame[row][col][2] - prevFrame[row][col][2]))**2

            #The squared distance in the red colorspace
            disSqr  = disRSqr

            #If the distance in the colorspace is greater than the threshold
            #squared
            if disSqr >= THRESHOLD ** 2:

                #Sets the pixel of the processed frame equal to red
                processedFrame[row:row+STRIDE, \
                               col:col+STRIDE] \
                               = [ 0, 0, 255 ]

                #Defaults that a blob was found for this pixel
                found = False

                #Caluclates the x and y of the upscaled iamge
                x = (col * KERNEL_SIZE) + (KERNEL_SIZE - STRIDE)
                y = (row * KERNEL_SIZE) + (KERNEL_SIZE - STRIDE)

                #For every blob that exists in the list of blobs
                for b in blobs:
                    
                    #If the current blob is near the point considered
                    if b.isNear(x,y):

                        #Add this point to the blob
                        b.add(x, y)

                        #A blob has been found for this pixel
                        found = True
                
                #If no blob has been found for this pixel
                if not found:

                    #Add a new blob and have the pixel be a part of it
                    blobs.append(Blob(x, y))

                #Sets the location
                locX += x
                locY += y

                #Increments the count
                countX += 1
                countY += 1

    #Calculates the average pixels of motion
    avgX = locX // countX
    avgY = locY // countY

    #Linearly interpolates to make it prettier (but slower)
    lerpX = lerp(lerpX, avgX, T)
    lerpY = lerp(lerpY, avgY, T)

    #Calculates where to draw the x and y of average motion
    dispX, dispY = int(lerpX * SIZE_SCALE), int(lerpY * SIZE_SCALE)

    #Creates a copy of the original frame for alpha bleneding
    bgFrame = originalFrame.copy()

    #For each linearly interpolated blob and current blob
    for lerpB, b in zip(lerpBlobs, blobs):

        #Finds the topleft and bottom right corner
        topLeft     = b.minCorner(lerpB.minX, lerpB.minY)
        bottomRight = b.maxCorner(lerpB.maxX, lerpB.maxY)

        #Outline the blob bounding box
        originalFrame = cv.rectangle(originalFrame,(topLeft[0]-3,topLeft[1]-3),\
                                     (bottomRight[0]+3, bottomRight[1]+3),     \
                                     [0, 0, 0], 3)

        #Draw the blob bounding box in red
        originalFrame = cv.rectangle(originalFrame, topLeft, bottomRight, \
                                     [0, 0, 255], -1)

    #Adds transparency to the bounding boxes
    originalFrame = cv.addWeighted(originalFrame, 0.3, bgFrame, 0.7,0)

    #Draws and circle over the processed frame at the point of average motion
    originalFrame = cv.circle(originalFrame, (dispX, dispY), 10, \
                              [ 255, 0, 0 ] , -1)

    
    #Displays the frame to the screen
    cv.imshow('Motion Detection', originalFrame)

    ######################### EVENT PROCESSING ################################
    
    #Waits to error check
    k = cv.waitKey(1)

    #If the 'ESC' key is pressed
    if k == 27:

        #Destroyes all windows that have been created
        cv.destroyAllWindows()

        #Breaks out of the reading loop
        break

    ######################## PREPEARES FOR NEXT LOOP ##########################

    #Saves the current blobs for the next frame
    lerpBlobs = list(blobs)

    #Pushes the current frame to the previous frame
    prevFrame = newFrame

    #Reads in the next frame
    success, originalFrame = video.read()

    #Loads in the next frame and resizes it
    newFrame = maxPool(originalFrame, KERNEL_SIZE, STRIDE)

###############################################################################