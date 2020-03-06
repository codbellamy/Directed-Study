###############################################################################
#                                                                             #
#                                FlappyGame.py                                #
#                              using NEAT library                             #
#                                J. Steiner                                   #
#                                                                             #
###############################################################################

#%%############################# LOADS MODULES ################################
import pygame
import neat
import time
import os
import random
###############################################################################

#%%############################## INITIALIZES #################################

#Initializes pygame font loading
pygame.font.init()

#Initializes a global variable that counts the current generation
genCtr = 0

#Sets the window width and height as global constants
WIN_WIDTH = 576
WIN_HEIGHT = 800

#Loads in the bird flying animation as a global array of objects
BIRD_IMGS = [
             pygame.transform.scale2x(pygame.image.load(
                                      os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(
                                      os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(
                                      os.path.join("imgs", "bird3.png")))
            ]

#Loads in the pipe image as a global object
PIPE_IMG  = pygame.transform.scale2x(pygame.image.load(
                                     os.path.join("imgs", "pipe.png")))
#Loads in the base image as a global object
BASE_IMG  = pygame.transform.scale2x(pygame.image.load(
                                     os.path.join("imgs", "base.png")))
#Loads in the background as a global object
BG_IMG    = pygame.transform.scale2x(pygame.image.load(
                                     os.path.join("imgs", "bg.png")))

#Loads in the font we will use to display as a global object
STAT_FONT = pygame.font.SysFont("comicsans", 50)
###############################################################################

#%%############################### BIRD CLASS #################################
class Bird:

    #Sets local class constants

    #Sets the set of local images for each bird object to be equal to the
    #global objects that were loaded
    IMGS = BIRD_IMGS

    #Sets the maximum rotation of the bird images to be 25 degrees
    MAX_ROTATION = 25

    #Sets the velocity of rotation for the bird images to be 20 degrees
    ROT_VEL = 20

    #Sets the animation time to be 5 ticks
    ANIMATION_TIME = 5

    ###########################################################################
    # Name:  __init__
    # Param: x - the initial starting x position of the bird
    #        y - the initial starting y position of the bird
    # Notes: class constructor
    def __init__(self, x, y):

        #Sets the instance variable x equal to the default x
        self.x = x

        #Sets the instance variable y equal to the default y
        self.y = y

        #Initializes the initial bird rotation to 0
        self.tilt = 0

        #Initializes the initial acceleration tick count to 0
        self.tickCount = 0

        #Initializes the bird velocity to 0
        self.vel = 0

        #Initializes the height of last jump for the bird to be its default y
        self.height = self.y

        #Defaults the animation timer to be 0
        self.imgCount = 0

        #Defaults the current frame to be displayed as the first image
        self.img = self.IMGS[0]
    ###########################################################################

    ###########################################################################
    # Name:  jump
    # Notes: Applies an upward force to the bird, resets counters involved with
    #        jumping
    def jump(self):

        #Applies an upward force to the bird by instantly changing its velocity
        self.vel = -10.5

        #Resets the acceleration tick counter of the bird
        self.tickCount = 0

        #Resets the height of last jump of the bird
        self.height = self.y
    ###########################################################################

    ###########################################################################
    # Name:  move
    # Notes: Updates the bird's displacement using a butchering of a baisc 
    #        kinematics formula
    def move(self):

        #Updates the time unit used for acceleration
        self.tickCount += 1
        
        #Updates displacement for this time unit based loosely on kinematics
        d = self.vel * self.tickCount + 1.5 * self.tickCount ** 2

        #Terminal velocity
        if d > 16:
            d = 16

        #Extra boost
        elif d < 0:
            d -= 2

        #Updates bird y position
        self.y += d

        #Caps off maximum rotation for tiliting up and tiliting down
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    ###########################################################################

    ###########################################################################
    # Notes:  draw
    # Param:  win - the window that we'll be drawing the bird to
    def draw(self, win):

        #Increments the animation timer
        self.imgCount += 1

        #Runs through the animation
        if self.imgCount < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.imgCount < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.imgCount < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.imgCount < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.imgCount == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.imgCount = 0

        #Defaults to one image if the bird is falling
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.imgCount = self.ANIMATION_TIME * 2
        
        #Updates rotated image
        rotatedImg = pygame.transform.rotate(self.img, self.tilt)
        newRect = rotatedImg.get_rect(center = self.img.get_rect(
                                      topleft = (self.x, self.y)).center)

        #Draws the image to the window
        win.blit(rotatedImg, newRect.topleft)
    ###########################################################################

    ###########################################################################
    # Name:  get_mask
    # Notes: used for pixel perfect collision detection
    def get_mask(self):
        #Gets the bird images' mask
        return pygame.mask.from_surface(self.img)
    ###########################################################################

###############################################################################

#%%############################### PIPE CLASS #################################
class Pipe:

    #A local constant, the velocity at which the pipes move across the screen
    VEL = 5

    ###########################################################################
    # Name:  __init__
    # Param: x - the initial starting position of the pipe
    # Notes: class constructor
    def __init__(self, x):

        #Initilizes the pipe x value to be the x value passed in
        self.x = x

        #Defaults parameters regarding the pipe
        self.height = 0
        self.gap = 200
        self.top = 0
        self.bottom = 0

        #Loads in the pipe images from global constants
        self.pipeTop = pygame.transform.flip(PIPE_IMG, False, True)
        self.pipeBottom = PIPE_IMG

        #The pipe has not yet been passd by the bird
        self.passed = False

        #Sets the height of the pipe to be a random y value
        self.setHeight()
    ###########################################################################

    ###########################################################################
    # Name:  setHeight
    # Notes: initializes a random height of the pipe
    def setHeight(self):

        #Initializes a random height of the pipe within a reasonable range
        self.height = random.randrange(50, 450)

        #Calculates the y values for the top and bottom pipes
        self.top    = self.height - self.pipeTop.get_height()
        self.bottom = self.height + self.gap
    ###########################################################################

    ###########################################################################
    # Name:  move
    # Notes: moves the pipe by the velocity constant
    def move(self):
        self.x -= self.VEL
    ###########################################################################

    ###########################################################################
    # Name:  draw
    # Param: win - the window that images will be drawn to
    # Notes: draws the pipes to the screen
    def draw(self, win):

        #Draw the top pipe
        win.blit(self.pipeTop, (self.x, self.top))

        #Draws the bottom pipe
        win.blit(self.pipeBottom, (self.x, self.bottom))
    ###########################################################################

    ###########################################################################
    # Name:  collide
    # Param: bird - the bird object that the collision will be checked with
    # Notes: if the pipe collided with the bird
    def collide(self, bird):

        #Gets the masks for the bird, top pipe, and bottom pipe
        bird_mask = bird.get_mask()
        top_mask  = pygame.mask.from_surface(self.pipeTop)
        bottom_mask = pygame.mask.from_surface(self.pipeBottom)

        #Calculates the distances between the pipe and the bird
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        #If the points overlap
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        #If the points overlap
        if t_point or b_point:
            #There was a collision
            return True

        #There was no collision
        return False
    ###########################################################################

###############################################################################

#%%############################### BASE CLASS #################################
class Base:

    #A local velocity constant that is the speed the base will move at
    VEL = 5

    #Gets the width of the base image
    WIDTH = BASE_IMG.get_width()

    #Stores the base image locally
    IMG = BASE_IMG 

    ###########################################################################
    # Name:  __init__
    # Param: y - the starting y value of the base
    # Notes: class constructor
    def __init__(self, y):

        #Initilaizes the values of the base position
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    ###########################################################################

    ###########################################################################
    # Name:  move
    # Notes: moves the base across the screen, bases work like a treadmill to
    #        give the illusion of an infinite background
    def move(self):
        
        #Moves the base
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        #If the first base is off the screen
        if self.x1 + self.WIDTH < 0:
            #Resets it
            self.x1 = self.x2 + self.WIDTH
        #If the second base is off the screen
        elif self.x2 + self.WIDTH < 0:
            #Resets it
            self.x2 = self.x1 + self.WIDTH
    ###########################################################################

    ###########################################################################
    # Name:  draw
    # Param: win - the window that will be drawn to
    # Notes: draws the bases to the screen
    def draw(self, win):

        #Draws two bases, one right after the other
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
    ###########################################################################

###############################################################################

###############################################################################
# Name:  drawWindow
# Param: win     - the window that will be drawn to
#        birds   - a list of bird objects to draw
#        pipes   - a list of pipe object to draw
#        base    - a base object to draw
#        score   - the current bird's score
#        gen     - the current generation number to draw as a diagnostic
#        liveCtr - the current number of birds alive to draw as a diagnostic
# Notes: draws everything to the screen
def drawWindow(win, birds, pipes, base, score, gen, liveCtr):

    #Draws the background image to the screen
    win.blit(BG_IMG, (0,0))

    #For each pipe object in the pipe list
    for pipe in pipes:
        #Draws the pipe to the screen
        pipe.draw(win)

    #Renders score, generation number, and birds left alive as diangostics
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    text = STAT_FONT.render("Birds Alive: " + str(liveCtr), 1, (255, 255, 255))
    win.blit(text, (10, 40))

    #Draws the base to the window
    base.draw(win)

    #For each bird in the bird list
    for bird in birds:
        #Draws the bird
        bird.draw(win)

    #Flips the buffer
    pygame.display.update()
###############################################################################

###############################################################################
# Name:  game
# Param: genomes - the current bird genomes for the current generation
#        config  - the current configuraiton settings for the neat algorithm
# Notes: runs one generation of the neat algoirthm
def game(genomes, config):

    #Increments the current generation coutner
    global genCtr
    genCtr += 1

    #Defaults this generation's networks, genomes, and birds to empty lists
    nets  = []
    ge    = []
    birds = []

    #For each genome in the new litter of genomes
    for _, g in genomes:

        #Gets the genome's network
        net = neat.nn.FeedForwardNetwork.create(g, config)
        #Appends to the netowrk list
        nets.append(net)

        #Append a new bird to the bird list
        birds.append(Bird(230, 350))

        #Initializes the genome's fittness to 0
        g.fitness = 0
        #Appends the genome to the genome list
        ge.append(g)

    #Creates the base object
    base  = Base(730)
    #Initializes one pipe in the pipe list
    pipes = [ Pipe(700) ]

    #Creates the window
    win   = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    
    #Defaults the score to be 0
    score = 0

    #Defaults the game to not have been quit
    quit  = False

    #Creates a pygame clock object
    clock = pygame.time.Clock()

    #While we are not quitting the game
    while not quit:

        #60 FPS
        clock.tick(60)

        #Goes through each event registered
        for event in pygame.event.get():

            #If the 'X' button has been pressed
            if event.type == pygame.QUIT:

                #Quit the while loop
                quit = True

                #Deallocate memory to pygame
                pygame.quit()

        #Defaults the pipe index
        pipe_ind = 0
        #If there are birds left
        if len(birds) > 0:
            #If the pipe has been passed
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipeTop.get_width():
                #Updates the pipe index used for the input neurons
                pipe_ind = 1
        #If there are no birds left
        else:
            #Quit the game loop
            quit = True
            break

        #For each bird in the bird list
        for x, bird in enumerate(birds):

            #Moves the bird
            bird.move()

            #A 10th of a fittness point per frame
            ge[x].fitness += 0.1
            
            #Gets the output acitivations from the bird
            output = nets[x].activate((bird.y, 
                                       abs(bird.y - pipes[pipe_ind].height), 
                                       abs(bird.y - pipes[pipe_ind].bottom)))

            #If the output acitvation is greater than 0.5
            if output[0] > 0.5:
                #The bird jumps
                bird.jump()

        #Defaults an empty list of pipes to remove and pipes to add
        rem = []
        addPipe = False

        #For each pipe int he pipe list
        for pipe in pipes:

            #For each bird in the bird list
            for x, bird in enumerate(birds):

                #If the bird collided with the pipe
                if pipe.collide(bird):

                    #Lowers fitness
                    ge[x].fitness -= 1

                    #Removes the bird
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                
                #If the pipe was passed
                if not pipe.passed and pipe.x < bird.x:
                    #Sets it to be passed and adds a pipe
                    pipe.passed = True
                    addPipe = True
            #If the pipe is off the screen
            if pipe.x + pipe.pipeTop.get_width() < 0:
                #Removes it
                rem.append(pipe)
            
            #Moves the pipe
            pipe.move()

        #If a pipe needs to be added
        if addPipe:
            #Incrmenets score
            score += 1

            #Increments fittness
            for g in ge:
                g.fitness += 5

            #Creates a new pipe
            pipes.append(Pipe(700))
        
        #Removes pipes that need to be removed
        for r in rem:
            pipes.remove(r)

        #For each bird in birds
        for x, bird in enumerate(birds):
            
            #If the bird flew over the screen or hit the ground
            if bird.y + BIRD_IMGS[0].get_height() >= 730 or bird.y < 0:
                #Removes the bird
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        #Moves the base
        base.move()

        #Draws to the window
        drawWindow(win, birds, pipes, base, score, genCtr, len(birds))
###############################################################################

###############################################################################
# Name:  run
# Param: path - the file path of the configuration file
# Notes: runs NEAT algorithm with our game loop
def run(path):

    #Loads in the config file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, path)

    #Creates a new population of birds
    p = neat.Population(config)

    #Creates stats reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    #The fittest bird
    winner = p.run(game,50)

#Loads in the config file
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'flappy_config.txt')

    #Runs the program
    run(config_path)