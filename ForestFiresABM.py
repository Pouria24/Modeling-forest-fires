
# ## Goals of this assignment
# 
# The primary goal of this assignment is to model the spread of a forest fire using an agent-based model (ABM).  In doing so, we will:
# 
# * Use ABM to model forest fires
# * Examine and quantify the concept of a "tipping point" in a model.
#--
# ## Reviewing the motivation for the model
# 
# ### Why model forest fires?
# 
# While this isn't a huge problem in Michigan, the states in the western United States having been suffering a tremendous problem with huge and difficult-to-control forest fires.  This comes from a combination of extended drought conditions, dense woodlands, and forest management policies that suppress small fires and thus ensure that large quantities of dead, dry trees and brush are available to burn when a large fire inevitably starts (typically from lighting strikes, but occasionally from negligent campers).  In recent years, this has been exacerbated by climate change, which has both caused drought conditions to be more severe and allowed tree-killing diseases and insects to flourish, which produces more dead, easily-burned wood.
# 
# These forest fires destroy ecosystems and peoples' homes and other property, and can result in the loss of human and animal life.  A key challenge in forest management is to attempt to contain these huge forest fires once they start, in order to protect human lives, settlements, and infrastructure.  To that end, it is critical to have models of how fire spreads in various conditions; see, for example, the [Open Wildland Fire Modeling group](http://www.openwfm.org/wiki/Open_Wildland_Fire_Modeling_E_community_Wiki).
# 
# More generally, the type of model that we're going to create is an example of a "percolation" model, where one substance (in this case, fire) moves through another substance (in this case, forest).  This type of problem is interesting in a variety of fields, including geology (oil or water percolating through rock, sand, or soil), human behavior (crowd movement in amusement parks), and in physics (understanding how two materials mix together).  
# 
# ### What is a "tipping point"?
# 
# This model also demonstrates the concept of a "critical threshold" or a "tipping point".  This is a phenomenon that occurs when a small change in an input parameter results in a large change in outcome.  This is a phenomenon that shows up in both simple and complex models, and happens in such varied circumstances as forest fires, the spread of disease in populations, and the transfer of information within a population.
# ---
# ## The rules for our model
# 
# In this model, we'll create a two-dimensional square array with sides $N$ cells long that represents the forested region we're simulating.  **The cells in the array can have three values: 0 (empty), 1 (trees), and 2 (on fire).**  At the beginning of the model, a user-specified fraction of the cells $f_\text{trees_start}$ (equivalent to the NetLogo `density` parameter) are randomly filled with trees, and the remaining cells are empty.  
# **One edge of the board (say, the entire leftmost column) is set on fire.**
# 
# Each cell has a "neighborhood" that is composed of its four neighbors to the left, right, above, and below it.  (Note: not the diagonal elements!)  If a cell is along one of the edges of the array, only consider the neighbors that it has, and don't try to go out of the bounds of the array!
# 
# The model takes steps forward in time, where every cell is modified based on the previous step.  The model evolves as follows:
# 
# * If the cell was empty last turn, it stays empty this turn.
# * If the cell is a tree and any of its neighbors were on fire last turn, it catches on fire.
# * If the cell was on fire last turn, the fire has consumed all of the trees and it is now empty.
# 
# The model evolves forward in time until all of the fires have burned out.  After this happens, you can calculate the fraction of the cells that still have trees at the end of the model ($f_\text{trees_end}$) and the fraction of cells that are empty ($f_\text{empty}$).  The fraction of burned cells, $f_\text{burned}$, is just the difference between the fraction of cells that were initially trees and the fraction of cells that are trees at the end of the model; in other words,
# 
# $f_\text{burned} = f_\text{trees_end} - f_\text{trees_start}$
# 

# 
# ## Your mission:
# 
# Your mission is to answer the question: "How does the spread of fire relate to the density of the forest?"  
# More precisely, we're asking "How does $f_\text{burned}$ depend on $f_\text{trees_start}$?"
# 
# To achieve this mission, we will break this down into two stages:
# 
# ---
# ### Stage 1:  Planning your solution
# As a group create pseudocode on the whiteboards that shows how you plan to implement this model for an arbitrary value of $f_\text{trees_start}$.  Make sure that you think about how to set up the initial conditions, how to evolve the model, and how to calculate the fraction of trees and empty cells that remain in the end.  Use the whiteboard and make sure to take a picture of it. **You must turn in the picture along with your notebook.**
# 
# **Important**: Make sure you discuss how you will handle cells that are on the boundaries of your 2D board!
# 
# **Do not spend more than 20 minutes on this part of the activity!**
# ___

# ### Stage 2: Implementing your solution
# 
# Now we're going to work through a combination of provided code and code that you have to write. The goal is to have a functioning forest fire model by the end of class!
# 
# In[1]:
# standard includes
import numpy as np
import numpy.random as rand
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Next we are going to import some specific libraries we will use to get the animation to work cleanly
from IPython.display import display, clear_output
import time  


# ### First important Function: Plotting the grid
# 

# In[2]:


# Function plotgrid() does what??

def plotgrid(myarray):
    
    # create two vectors based on x and y size of the grid
    x_range = np.linspace(0, myarray.shape[1]-1, myarray.shape[1]) 
    y_range = np.linspace(0, myarray.shape[0]-1, myarray.shape[0])
    
    # making two matrix
    x_indices, y_indices = np.meshgrid(x_range, y_range)
    
    # making square and triangle x and y
    tree_x = x_indices[myarray == 1];   
    tree_y = y_indices[myarray == 1]; 
    fire_x = x_indices[myarray == 2];   
    fire_y = y_indices[myarray == 2]; 
    
    # plotting the square and triangles
    plt.plot(tree_x, myarray.shape[0] - tree_y - 1, '^g',markersize=10)   
    plt.plot(fire_x, myarray.shape[0] - fire_y - 1, 'rs',markersize=10)  
    
    # making sure we dont cutoff the shapes
    plt.xlim([-1,myarray.shape[1]])
    plt.ylim([-1,myarray.shape[0]]) 

    # checking if there is fire around it
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)


# ### Initializing the forest
# 
# In[3]:


def set_board(board_size=50,f_trees_start=0.5):
    '''
    Creates the initial game board.

    Inputs:
        board_size: length of one edge of the board
        f_trees_start: probability that a given cell is a tree
                       (effectively the tree density)

    Outputs a 2D numpy array with values set to either 0, 1, or 2
        (empty, tree, or fire)
    '''
    
    # all cells initialized to 'empty' (0) by default
    game_board = np.zeros((board_size,board_size),dtype='int64')
    
    # loop over board and roll the dice; if the random number is less
    # than f_trees_start, make it a tree.
    for i in range(board_size):
        for j in range(board_size):
            if rand.random() <= f_trees_start:
                game_board[i,j] = 1

    # set the whole left edge of the board on fire. We're arsonists!
    game_board[:,0] = 2
    
    return game_board


# In[4]:


fig = plt.figure(figsize=(10,10))
fire=set_board(50,0.6)
plotgrid(fire)


# ## The main event: make the fire spread!
#  
# > Each cell has a "neighborhood" that is composed of its four neighbors to the left, right, above, and below it.  (Note: not the diagonal elements!)  If a cell is along one of the edges of the array, only consider the neighbors that it has, and don't try to go out of the bounds of the array!
# 
# >The model takes steps forward in time, where every cell is modified based on the previous step.  The model evolves as follows:
# 
# >* If the cell was empty last turn, it stays empty this turn.
# >* If the cell is a tree and any of its neighbors were on fire last turn, it catches on fire.
# >* If the cell was on fire last turn, the fire has consumed all of the trees and it is now empty.

# In[5]:


def onBoard(i, j, image):
    if i <= image.shape[0]-1 and i >= 0 and j<=image.shape[1]-1 and j >=0: # You need some more conditions here! 
                                     # We've checked i, but what about j?
            return True
    else:
        return False
def getNeighborValues(i,j, board):
    # The following list contains the indices of the neighbors for a pixel at (i.j)
    neighborhood = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]

    neighbor_values = []
    for neighbor in neighborhood:
                        # FIXME!
        if onBoard(neighbor[0],neighbor[1],board)==True:
            neighbor_values.append(board[neighbor])

    return neighbor_values


# In[6]:


def advance_board(game_board):
    '''
    Advances the game board using the given rules.
    Input: the initial game board.
    Output: the advanced game board
    '''
    
    # create a new array that's just like the original one, but initially set to all zeros (i.e., totally empty)
    new_board = np.zeros_like(game_board)
    
    # loop over each cell in the board and decide what to do.
    for i in range(new_board.shape[0]):
        for j in range(new_board.shape[1]):
    # You'll need two loops here, one nested inside the other.
            shape=new_board.shape[0]
            # Now that we're inside the loops we need to apply our rules
            
            # if the cell was empty last turn, it's still empty.
            if game_board[i,j]==0:
                new_board[i,j]=0
            # if it was on fire last turn, it's now empty.
            if game_board[i,j]==2:
                new_board[i,j]=0
    
            # now, if there is a tree in the cell, we have to decide what to do
            if game_board[i,j]==1:
                
                # initially make the cell a tree in the new board
                new_board[i,j]=1

                # If one of the neighboring cells was on fire last turn, 
                # this cell is now on fire!
                fireneighbor=getNeighborValues(i,j, game_board)
                if 2 in fireneighbor:
                    new_board[i,j]=2
                # (make sure you account for whether or not you're on the edge!)
                

    # return the new board
    return new_board



# In[7]:


# Initialize a new board here
p=advance_board(fire)


# In[12]:

fig = plt.figure(figsize=(10,10))
plotgrid(p)


# ## Analyzing the state of the board
# 
# In[13]:


def calc_stats(game_board):
    '''
    Calculates the fraction of cells on the game board that are 
    a tree or are empty.
    
    Input: a game board
    
    Output: fraction that's empty, fraction that's covered in trees.
    '''
    frac_empty=0
    frac_tree=0
    # use numpy to count up the fraction that are empty
    for i in range(game_board.shape[0]):
        for j in range(game_board.shape[1]):
            if game_board[i,j]==0:
                frac_empty+=1

    # do the same for trees
            if game_board[i,j]==1:
                frac_tree+=1
    sh=game_board.shape[0]
    total=sh*sh
    frac_empty=frac_empty/total
    frac_tree=frac_tree/total
    # return it!
    return frac_empty, frac_tree


# ### Putting it all together!
#
# In[16]:


# This initials the values
f_trees_start=0.3
board_size = 50

#this making the figure size
fig = plt.figure(figsize=(10,10))

# This starts the first board
game_board = set_board(board_size=board_size, f_trees_start=f_trees_start)

# This plots it
plotgrid(game_board)

# This starts the fire
on_fire = True

# This keeps going through the function and keep going to the next frame
while on_fire == True:

    # This makes the new board the normal board
    game_board = advance_board(game_board)
    
    # This in the time between each frame
    plotgrid(game_board)
    time.sleep(0.01)  # 
    clear_output(wait=True)
    display(fig)
    fig.clear()

    # This calculates the fractions
    frac_empty, frac_trees = calc_stats(game_board)

    # This keeps the fire going
    if frac_empty + frac_trees == 1.0:
        on_fire = False

# This closes the fire
plt.close()               



# In[ ]:


board_size = 50
step=np.arange(0.01,1,0.01)
f_tree = []
f_burned = []

for tree_fraction in step:
    
    # Complete this line
    game_board = set_board(board_size=board_size, f_trees_start=f_trees_start)

    on_fire = True
    while on_fire == True:
        # Complete this line
        game_board = advance_board(game_board)
        
        # Complete this line
        frac_empty, frac_trees = calc_stats(game_board)
        if frac_empty + frac_trees == 1.0:
            # Complete this line
            on_fire = False

    f_tree.append(tree_fraction)
    f_burned.append(frac_empty - (1.0-tree_fraction))
    

plt.plot(f_tree, f_burned)
plt.xlabel("tree fraction")
plt.ylabel("burned fraction (normalized)")


