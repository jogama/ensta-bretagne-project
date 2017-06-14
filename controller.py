#!/usr/bin/python3

from potential_field_islands import draw_field, make_islands
import matplotlib.pyplot as plt 
import roblib as rl
import numpy as np

"""
There is not a single convention as to the definition of the state in robmoocpy code. Here, it will be a numpy array x = [x, y, θ]. This would more elegantly bre represented as a vector with respect to the origin. However, we will first implement this in a "yo, it just works this way" kind of way. Aiming for easiest correct implementation, even if it is somewhat cumbersome. 
"""

def follow_level_curve(height, desired_height, gradient):
    '''
    Simple proportional controller to follow a level
    Args:
      height:   Number
      gradient: Number

    Returns:
      xdot = (x, y) 
    '''
    ω = desired_height - height
    return ω


def straight_line(x):
    ''' toy controller to go in a straight line. For some reason, it also accelerates.'''
    x = x.flatten()
    v = x[2]
    θ = np.pi / 2
    x0 = x[0]  + v * np.cos(θ)
    x1 = x[1]  + v * np.sin(θ)
    return np.array([[x0, x1, v, θ]]).T
    
def f(x,u=np.array([[0],  [0.3]])):
    ''' this seems to be the defualt controller.'''
    
    x,u  = x.flatten(), u.flatten()
    v,θ = x[2],x[3]    
    return np.array([ [ v*np.cos(θ) ],[ v*np.sin(θ) ], [u[0]], [u[1]]])


def run_car(duration, dt=.1):
    # initialize variables
    x    = np.array([[0, 0, 1, np.pi/2]]).T #x,y,v,θ
    fig  = plt.figure(0)
    ax   = fig.add_subplot(111, aspect='equal')
    xmin,xmax,ymin,ymax=-5,5,-5,5
    V_0   = 0.25 # desired height, or potential

    # make a controller that includes the gradient
    # the curve following controller does does not localize, but uses the 
    Mx, My, VX, VY, V = make_islands(xmin, xmax, ymin, ymax)
    
    flc = lambda x: np.array([x[0], x[1], x[2],
                              follow_level_curve(V[x[0], x[1]],
                                                 V_0,
                                                 np.array([VX[x[0], x[1]], VY[x[0], x[1]]]))])
    def flc(x):
        # make a controller that includes the gradient
        # the curve following controller does does not localize, but uses the
        xi = x.astype(int) # this *WILL FAIL*, but it will run.
        height = V[xi[0], xi[1]]
        gradient = np.array([VX[xi[0], xi[1]], VY[xi[0], xi[1]]])
        ω = follow_level_curve(height, V_0, gradient)
        
        return ω
    
    # run the animation
    for t in np.arange(0,duration,dt):
        # Reset matplotlib view
        plt.pause(0.001)
        plt.cla()
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

        # iterate the controler and draw vehicle
        x = x + dt*flc(x)
        rl.draw_tank(x[[0,1,3]],'red',0.2) # x,y,θ
        draw_field(normalize=False
                   , animation_vars=(x, xmin, xmax, ymin, ymax))

if __name__ == "__main__":
    run_car(2)
