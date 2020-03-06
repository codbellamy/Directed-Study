###############################################################################
#                                                                             #
#                                  Mazes.py                                   #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%############################# LOADS MODULES ################################
import pygame
import numpy as np
import time
from NNat_Q import *
#%%########################### GLOBAL CONSTANTS ###############################
#The reward matrix to give rewards to the AI
REWARD = np.zeros((10, 10))
###############################################################################

#%%######################### GLOBAL REWARD FUNCTION ###########################
# Name:   reward
# Param:  i           - the current tile i position
#         j           - the current tile j position
#         goal_i      - the current goal_i position
#         goal_j      - the current goal_j position
#         maxDistance - the intercept of the distance
# Return: the reward calculated as the distance from the goal
# Notes:  used to calculate the reward value of a tile
def reward(i, j, goal_i, goal_j, maxDistance = 0):
    #Calculates the reward based on distance from the goal
    reward = maxDistance - int( ((goal_i-i) ** 2 + (goal_j-j) ** 2) **(0.5))
    return reward
###############################################################################

#%%########################## AI CLASS DEFINITION #############################
class Q:
    
    #A large negative constant
    NEGATIVE_BIG_M = -2 ** 10

    ###########################################################################
    # Name:  __init__
    # Notes: class constructor
    def __init__(self):
        #Default/Define reward memory
        self.rewardMap = np.zeros((10, 10))

        #Creates an instance of a reinforcement learning network
        self.network   = NNat_Q([3, 20, 4], 0.1)
    ###########################################################################

    ###########################################################################
    def query(self, inputVector):

        #gets player location from the input vector
        x = inputVector

        #gets the reward of the locations up, down, left, and right of the
        #player location
        potentialReward = []
        
        #If up is in bounds
        if x[1] > 0:
            #Looks to the potential reward up
            potentialReward.append(self.rewardMap[x[1]-1][x[0]])
        #If the up is out of bounds
        else:
            #Assigns a potential reward of negative big M
            potentialReward.append(self.NEGATIVE_BIG_M)
        #If the down is in bounds
        if x[1] < len(self.rewardMap)-1:
            #Looks to the potential reward down
            potentialReward.append(self.rewardMap[x[1]+1][x[0]])
        #If the down is out of bounds
        else:
            #Assigns a potential reward of negative big M
            potentialReward.append(self.NEGATIVE_BIG_M)

        #If the left is in bounds
        if x[0] > 0:
            #Looks to the potential reward to the left
            potentialReward.append(self.rewardMap[x[1]][x[0]-1])
        #If the left is out of bounds
        else:
            #Assigns a potential reward of negative big M
            potentialReward.append(self.NEGATIVE_BIG_M)
        #If the right is in bounds
        if x[0] < len(self.rewardMap[0])-1:
            #Looks to the potential reward to the left
            potentialReward.append(self.rewardMap[x[1]][x[0]+1])
        #If the left is out of bounds
        else:
            #Assigns a potential reward of negative big M
            potentialReward.append(self.NEGATIVE_BIG_M)

        #calculates the delta rewards for all 4 options using the formula
        #deltaReward = newReward - currentReward as a list
        deltaReward = potentialReward - self.rewardMap[x[1]][x[0]]
        
        r = np.argmax(deltaReward)

        y = [0, 0, 0, 0]
        y[r] = 1

        output = r
        for _ in range(100):
            output = self.network.query(x + [np.argmax(output)], y)

        o = []
        for i in output.flatten():
            o.append(int(i*100))

        rand = np.random.randint(sum(o))

        if rand < o[0]:
            choice = 0
        elif rand < o[0]+o[1]:
            choice = 1
        elif rand < o[0]+o[1]+o[2]:
            choice = 2
        else:
            choice = 3

        #returns the argmax of the decision vector
        return choice
    #######################################################################

###############################################################################

#%%###################### ENUMERATES CARDINAL DIRECTIONS ######################
class Dir():
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
###############################################################################

#%%######################## CREATES TILE MAP OF MAZE ##########################
class TileMap:

    #Constants that determine what the classification of the tile is
    EMPTY    = 0
    WALL     = 1
    GOAL     = 2

    #Indexes the goal after map creation, so the location can be retrieved later
    GOAL_IDX = (0, 9)

    #Creates the tile map as a matrix in python
    #TODO: add implementation of loading tile map from a text file
    TILE_MAP = [ 
    [ EMPTY, EMPTY, EMPTY, WALL,  EMPTY, EMPTY, EMPTY, EMPTY, WALL,  EMPTY ],
    [ WALL,  WALL,  WALL,  EMPTY, WALL,  EMPTY, WALL,  WALL,  WALL,  EMPTY ],
    [ WALL,  EMPTY, EMPTY, EMPTY, WALL,  EMPTY, WALL,  EMPTY, EMPTY, EMPTY ],
    [ WALL,  WALL,  EMPTY, EMPTY, WALL,  EMPTY, WALL,  EMPTY, WALL,  WALL  ],
    [ WALL,  EMPTY, WALL,  EMPTY, WALL,  EMPTY, WALL,  EMPTY, WALL,  EMPTY ],
    [ WALL,  EMPTY, EMPTY, EMPTY, WALL,  EMPTY, WALL,  EMPTY, WALL,  WALL  ],
    [ WALL,  WALL,  WALL,  EMPTY, WALL,  WALL,  EMPTY, EMPTY, EMPTY, WALL  ],
    [ EMPTY, EMPTY, WALL,  EMPTY, EMPTY, EMPTY, EMPTY, WALL,  WALL,  WALL  ],
    [ EMPTY, EMPTY, WALL,  EMPTY, WALL,  WALL,  WALL,  WALL,  EMPTY, EMPTY ],
    [ EMPTY, EMPTY, WALL,  EMPTY, WALL,  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY ],
               ]

    #The width and height of each tile in pixels
    TILE_WIDTH  = 64
    TILE_HEIGHT = 64

    ###########################################################################
    # Name:  __init__
    # Notes: class constructor
    def __init__(self):
        #Sets the goal within the tile map
        self.TILE_MAP[self.GOAL_IDX[0]][self.GOAL_IDX[1]] = self.GOAL
    ###########################################################################

    ###########################################################################
    # Name:  draw
    # Param: window - the window that will be drawn to
    # Notes: loops through the tile map and draws each tile based on its 
    #        classification
    def draw(self, window):

        #Loops through the indicies of rows in the tile map
        for row in range(len(self.TILE_MAP)):

            #Loops through each index of column (stored in col) and each tile
            #classification (stored in tile)
            for col, tile in enumerate(self.TILE_MAP[row]):

                #If the tile is not classified as EMPTY (placeholder tile)
                if tile != self.EMPTY:

                    #Calculates the Rect object assocciated with a tile at
                    #that index
                    tileRect = pygame.Rect(col*self.TILE_WIDTH,
                                           row*self.TILE_HEIGHT,
                                               self.TILE_WIDTH,
                                               self.TILE_HEIGHT)

                    #Determines the color of the tile based on its 
                    #classification
                    if tile == self.WALL:
                        tileColor = (000, 000, 000)
                    elif tile == self.GOAL:
                        tileColor = (255, 000, 000)

                    #Draws the tile
                    pygame.draw.rect(window, tileColor, tileRect)
    ###########################################################################

    ###########################################################################
    # Name:   getWdith
    # Return: width - the width of the map in pixels
    # Notes:  returns the width of the map in pixels 
    def getWidth(self):
        width = len(self.TILE_MAP[0])*self.TILE_WIDTH
        return width
    ###########################################################################
    
    ###########################################################################
    # Name:   getHeight
    # Return: height = the height of the map in pixels
    # Notes:  returns the height of the map in pixels
    def getHeight(self):
        height = len(self.TILE_MAP)*self.TILE_HEIGHT
        return height
    ###########################################################################

    ###########################################################################
    # Name:   getGoalLoc
    # Return: self.GOAL_IDX - a tuple of the indicies of the goal tile
    # Notes:  returns the goal location as a tuple
    def getGoalLoc(self):
        return self.GOAL_IDX
    ###########################################################################

###############################################################################

#%%############################## PLAYER CLASS ################################
class Player:

    #Defines the player width and height
    WIDTH  = 64
    HEIGHT = 64

    ###########################################################################
    # Name:  __init__
    # Notes: class constructor
    def __init__(self, i, j):

        #Defines player i and j values
        self.i = i
        self.j = j
    ###########################################################################

    ###########################################################################
    # Name:  move
    # Param: direction - the direction to move the player
    # Notes: moves the player 1 tile in 1 of the 4 cardinal directions
    def move(self, direction):

        #If the direction is up
        if direction ==   Dir.UP:
            #Moves up 1 tile
            self.j -= 1
        #If the direction is down
        elif direction == Dir.DOWN:
            #Moves down 1 tile
            self.j += 1
        #If the direction is left
        elif direction == Dir.LEFT:
            #Moves left 1 tile
            self.i -= 1
        #If the direction is right
        else:
            #Moves right one tile
            self.i += 1
    ###########################################################################

    ###########################################################################
    # Name:  draw
    # Param: window - the window to draw to
    # Notes: creates the player rect and draws the player at its current 
    #        positiion
    def draw(self, window):

        #Creates the player rect
        rect = pygame.Rect(self.i*self.WIDTH, self.j*self.HEIGHT, 
                                  self.WIDTH,        self.HEIGHT)

        #Draws the player at its current position
        pygame.draw.rect(window, (0,  255, 0  ), rect)
    ###########################################################################

    ###########################################################################
    # Name:   get_i
    # Return: self.i - the player's i position instance variable
    # Notes:  returns the player's i position
    def get_i(self):
        return self.i
    ###########################################################################

    ###########################################################################
    # Name:   get_j
    # Return: self.j - the player's j position instance variable
    # Notes:  returns the player's j position
    def get_j(self):
        return self.j
    ###########################################################################

#%%############################ HELPER FUNCTIONS ##############################

###############################################################################
# Name:  draw
# Param: window - the window we'll be drawing to
#        map    - the map currently loaded
#        player - the player currently running
# Notes: draws all objects once per frame
def draw(window, map, player):
    #Resets the window background
    window.fill((255, 255, 255))

    #Draws the map
    map.draw(window)

    #Draws the player
    player.draw(window)

    #Updates the window display
    pygame.display.update()
###############################################################################

###############################################################################
# Name:  update
# Param: map    - the map currently loaded
#        player - the player currently running
# Notes: updates game logic once per frame
def update(map, player, nnat):

    #If the palyer has not reached the goal
    if (player.get_j(), player.get_i()) != map.getGoalLoc():

            #Decides which direction to move to
            direction = nnat.query([player.get_i(), player.get_j()])
            
            #Moves 1 tile
            player.move(direction)
###############################################################################

###############################################################################
# Name:   bounds
# Param:  map      - the map currently loaded
#         player   - the player currently running
# Return: inBounds - whether or not the player is within bounds
# Notes:  returns if the player has violated the bounds
def inBounds(map, player):

    #A flag that defaults the player to be in bounds
    inBounds = True

    #If the player is outside of the bounds of the map horizontally
    if player.get_i() < 0 or player.get_i() > len(map.TILE_MAP[0]) - 1:
        #The player is not in bounds
        inBounds = False
    
    #If the player is outside of the bounds of the map vertically
    elif player.get_j() < 0 or player.get_j() > len(map.TILE_MAP) - 1:
        #The player is not in bounds
        inBounds = False

    #If the map at the player position is a wall
    elif map.TILE_MAP[player.get_j()][player.get_i()] == map.WALL:
        #The player is not in bounds
        inBounds = False

    #Returns if the player is in bounds or not
    return inBounds

###############################################################################


def initReward(goali, goalj):

    for j in range(len(REWARD[0])):
        for i in range(len(REWARD)):
            REWARD[j][i] = reward(i, j, goali, goalj)

#%%############################# MAIN FUNCTION ################################
def main():

    #Initializes pygame 
    pygame.init()

    #Initializes the player position
    defaultPlayerPosn = (3, 9)

    #Initializes a map
    map = TileMap()

    #Initializes a player
    player = Player(defaultPlayerPosn[0], defaultPlayerPosn[1])

    #Defaults the width and height of the window display
    width, height = (map.getWidth(), map.getHeight())

    #Creates the display window
    window = pygame.display.set_mode((width, height))

    #While the 'X' button has not been pressed
    quit = False

    #Initializes the reward matrix
    initReward(map.getGoalLoc()[1], map.getGoalLoc()[0])
    #A small amount of bias so it doesn't get stuck at the first branch
    REWARD[6][3] -= 10

    #Initializes an instance of the AI
    nnat = Q()
    
    #Begins a timer for the program's elapsed time
    startTicks = time.time()

    #While we are not quitting the game
    while not quit:
        
        #Event loop
        for event in pygame.event.get():
            #If the 'X' button has been pressed
            if event.type == pygame.QUIT:
                quit = True

        #Draws to the window
        draw(window, map, player)

        #Updates game logic
        update(map, player, nnat)

        #Updates reqard matrix
        if inBounds(map, player):
            r = REWARD[player.get_j()][player.get_i()]
                        
            nnat.rewardMap[player.get_j()][player.get_i()] = r
        #If the player has violated the bounds
        else:
            r = nnat.NEGATIVE_BIG_M
            try:
                nnat.rewardMap[player.get_j()][player.get_i()] = r
            except:
                pass

            ###########Resets the program##################

            #Initializes a player
            player = Player(defaultPlayerPosn[0], defaultPlayerPosn[1])

        if (player.get_j(), player.get_i()) != map.getGoalLoc():
            #Diagnostic. prints the AI's memory
            print(nnat.rewardMap)

            print('Elapsed Time:', time.time() - startTicks )

###############################################################################

#%%########################### RUNS MAIN FUNCTION #############################
#Runs the program
main()
###############################################################################
print(REWARD)