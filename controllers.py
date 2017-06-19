#!/usr/bin/python3

import potential_fields as pf
from numpy.linalg import norm
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np
import pdb

"""
There is not a single convention as to the definition of the state in robmoocpy code. Here, it will be a numpy array x = [x, y, θ]. This would more elegantly bre represented as a vector with respect to the origin. However, we will first implement this in a "yo, it just works this way" kind of way. Aiming for easiest correct implementation, even if it is somewhat cumbersome. 
"""




def fp_disk(gradient, height):
    '''Follow Potential as a Disk.
    Args:
      gradient = [VX, VY] # potential
      height = the current height
    Returns:
      (dx, dy)
    Notes: The overall data model does not at all represent the auv,
      and should be rewritten as such. 
    '''
    # normalize gradient
    gradient = gradient / norm(gradient);
    dx = gradient[1]
    dy = gradient[0]
    return np.array([dx, dy])
        
def runcar(duration, dt=.1):
    # initialize variables
    x = np.array([[.2, 0, .1, 0]]).T  # x,y,v,θ
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    V_0 = 1  # desired height, or potential
    Mx, My, VX, VY, V = pf.make_vehicle_field(xmin, xmax, ymin, ymax,
                                              desired_potential=V_0, threshold=.1)

    # run the animation
    for t in np.arange(0, duration, dt):
        # Reset matplotlib view
        plt.pause(0.001)
        plt.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # The controller takes only gradient and height as input:
        xi = pf.location_to_index(x, Mx, My)
        height = V[xi[0], xi[1]]
        gradient = np.array([VX[xi[0], xi[1]], VY[xi[0], xi[1]]])

        # Get the displacements * velocity to iterate the controller:
        print( x.flatten()[2], fp_disk(gradient, height))
        dx, dy = fp_disk(gradient, height) * x.flatten()[2]
        dX = np.array([[dx, dy, 0, 0]]).T
        x = x + dt * dX

        # Draw vehicle and field
        rl.draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x,y,θ        
        pf.draw_field(fig=fig, field=(Mx, My, VX, VY, V, V_0))


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
    runcar(100, dt=1)
