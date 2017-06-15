#!/usr/bin/python3

from potential_field_islands import draw_field, make_islands
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np
import pdb

"""
There is not a single convention as to the definition of the state in robmoocpy code. Here, it will be a numpy array x = [x, y, θ]. This would more elegantly bre represented as a vector with respect to the origin. However, we will first implement this in a "yo, it just works this way" kind of way. Aiming for easiest correct implementation, even if it is somewhat cumbersome. 
"""


def follow_level_curve(height, desired_height, gradient):
    '''
    Simple proportional controller to follow a level
    Args:
      height:   Number
      gradient: 2-d numpy array

    Returns:
      xdot = (x, y) 
    '''
    # This is currently naïve.
    # change in heading
    dθ = (desired_height - height) * 10
    
    return dθ


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

    
def runcar(duration, dt=.1):
    # initialize variables
    x = np.array([[-4, -4, 100000, np.pi / 2]]).T  # x,y,v,θ
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    V_0 = 0.025  # desired height, or potential

    # make a controller that includes the gradient
    # the curve following controller does does not localize, but uses these:
    Mx, My, VX, VY, V = make_islands(xmin, xmax, ymin, ymax)

    def flc(state):
        # The actual controller doesn't know where the robot is; just the gradient and height.        
        xi = location_to_index(state, Mx, My)
        height = V[xi[0], xi[1]]; 
        gradient = np.array([VX[xi[0], xi[1]], VY[xi[0], xi[1]]])
        dθ = follow_level_curve(height, V_0, gradient)
        
        v, θ = state[2], state[3]
        dx   = np.zeros(4)
        
        dx[0] = v * gradient[0] * (V_0 - height) 
        dx[1] = v * gradient[1] * (V_0 - height) 
        # dx[3] = dθ
        print(dx)
        # print("height = ", height)
        # print("gradient:\n",gradient, gradient.shape)
        # print("dθ =", dθ)
        # print("θ  =", θ)
        # return column vector
        return dx.reshape((x.size, 1))

    # run the animation
    for t in np.arange(0, duration, dt):
        # Reset matplotlib view
        plt.pause(0.001)
        plt.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # iterate the controler and draw vehicle
        x = x + dt * flc(x) 
        rl.draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x,y,θ
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
