#!/usr/bin/python3

from potential_field_islands import draw_field, make_islands
from numpy.linalg import norm
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np
import pdb

"""
There is not a single convention as to the definition of the state in robmoocpy code. Here, it will be a numpy array x = [x, y, θ]. This would more elegantly bre represented as a vector with respect to the origin. However, we will first implement this in a "yo, it just works this way" kind of way. Aiming for easiest correct implementation, even if it is somewhat cumbersome. 
"""

def straight_line(x):
    ''' toy controller to go in a straight line. For some reason, it also accelerates.'''
    x = x.flatten()
    v = x[2]
    θ = np.pi / 2
    x0 = x[0] + v * np.cos(θ)
    x1 = x[1] + v * np.sin(θ)
    return np.array([[x0, x1, v, θ]]).T


def f(x, u=np.array([[0], [0.3]])):
    ''' this seems to be the defualt controller.'''

    x, u = x.flatten(), u.flatten()
    v, θ = x[2], x[3]
    return np.array([[v * np.cos(θ)], [v * np.sin(θ)], [u[0]], [u[1]]])

def location_to_index(loc, Mx, My):
    ''' Converts location in plot to index for the gradient and potentia matrices.
    Args:
      loc: iterable where loc[0] anx loc[1] are the x and y coordinates of the
         location, respectively.
      Mx and My:
         1-d numpy arrays with linearly increasing values. They are the x and y
         coordinates of the gradient and potential values, respectively.
    Returns:
      index: numpy array of shape (2,) with index corresponding to the gradient
         and potential arrays.
    '''
    # We assume the difference between the first two values in Mx, My, is the
    #   step size for the entire matrix, and extrapolate.
    #   There's still probably an easier way.
    step_size_guess = np.abs(np.array([Mx[0] - Mx[1], My[0] - My[1]]))
    x     = np.array([loc[0], loc[1]]).flatten()
    x_0   = np.array((Mx[0], My[0]))
    index = np.round((x - x_0) / step_size_guess)

    # I feel that these eight lines could be done in two...
    if(index[0] >= Mx.size):
        index[0] = Mx.size - 1
    elif(index[0] < 0):
        index[0] = 0        
    if(index[1] >=  My.size):
        index[1] = My.size - 1
    elif(index[1] < 0):
        index[1] = 0

    return index.astype(int)

def flc_disk(state, gradient, height, desired_height):
    # disks have 360° symmetry.
    unit_g = gradient / norm(gradient)
    
    dx = np.zeros(state.size)
    # dx[0] =  -unit_g[0] + unit_g[1] * (desired_height - height) 
    # dx[1] =   unit_g[1] + unit_g[0] * (desired_height - height)
    # dx[0] =  unit_g[1] * (desired_height - height) 
    # dx[1] =  unit_g[0] * (desired_height - height)
    dx[0] =  -unit_g[0] 
    dx[1] =   unit_g[1] 
    return dx.reshape((state.size, 1))

def flc_tank(state, gradient, height, desired_height):
    # The actual controller doesn't know where the robot is; just the gradient and height.
    v  = state[2]
    dx = np.zeros(state.size) 
    dx[0] = v * gradient[0] * (desired_height - height) 
    dx[1] = v * gradient[1] * (desired_height - height) 

    return dx.reshape((state.size, 1))
    

def follow_level_curve(state, desired_height, Mx, My, VX, VY, V):
    '''
    Wrapper for flc_tank and flc_disk. The state should really be an object, with properties. 
    '''
    # pyplot displays the robot in the coordinate frame, and this is its state.
    # The potential and gradient matrices have integer indices, thus this conversion:
    xi = location_to_index(state, Mx, My)
    height = V[xi[0], xi[1]]; 
    gradient = np.array([VX[xi[0], xi[1]], VY[xi[0], xi[1]]])
    
    if(state.size == 4):
        # Assume state includes θ
        return flc_tank(state, gradient, height, desired_height)
    if(state.size == 3):
        # Assume state does not include θ
        return flc_disk(state, gradient, height, desired_height)
    
    
def runcar(duration, dt=.1):
    # initialize variables
    x = np.array([[1, 1, 1500]]).T  # x,y,v
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    V_0 = 0.025  # desired height, or potential

    # make a controller that includes the gradient
    # the curve following controller does does not localize, but uses these:
    Mx, My, VX, VY, V = make_islands(xmin, xmax, ymin, ymax)
    controller = lambda state: follow_level_curve(state, V_0, Mx, My, VX, VY, V)

    # run the animation
    for t in np.arange(0, duration, dt):
        # Reset matplotlib view
        plt.pause(0.001)
        plt.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # iterate the controller and draw vehicle
        x = x + dt * controller(x)
        if(x.size == 4):
            rl.draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x,y,θ
        elif(x.size == 3):
            rl.draw_disk(x[[0, 1]], 0.2, ax, 'red') # x,y
            
        draw_field(normalize=False, animation_vars=(x, xmin, xmax, ymin, ymax))


def show_tank(x=np.array([[0, 0, 1, np.pi / 2]]).T, col='darkblue', r=1):
    '''
    Runs roblib.draw_tank and displays the result.
    Written to play with and further understand roblib.draw_tank().
    '''
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    rl.draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x,y,θ
    plt.show()

if __name__ == "__main__":
    runcar(25, dt=.5)
