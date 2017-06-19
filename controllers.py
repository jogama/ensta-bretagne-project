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


def flc_disk(gradient, error):
    unit_g = gradient / norm(gradient)
    dx = - unit_g[0] + unit_g[1] * error * np.round(1 / error)
    dy =   unit_g[1] + unit_g[0] * error * np.round(1 / error)

    return np.array([dx, dy])

def flc_disk_pid(gradient, error, error_sum):
    r = 0#np.array([-gradient[0], gradient[1]]) / norm(gradient)
    g = np.array([ gradient[1], gradient[0]]) / norm(gradient)
    g = g * error * np.round(1 / error) # Proportional term
    #    g = g + error_sum # Integral term

    return (r + g).flatten()


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
    print(gradient)
    dx = gradient[1]
    dy = gradient[0]
    print(dx, dy)
    return dx, dy
    

def flc_tank(gradient, error):
    # This scenario assumes we can only change the angle, or heading, of the vehicle.
    # It would be more efficient to pass dx and dy rather than recalculating them
    # in follow_level_curve, but this controller should approximate the real world.
    unit_g = gradient / norm(gradient)
    
    dx = - unit_g[0] + unit_g[1] * error * np.round(1 / error)
    dy =   unit_g[1] + unit_g[0] * error * np.round(1 / error)
    dθ = np.arctan(dy / dx)

    return dθ
    

def follow_level_curve(state, desired_height, Mx, My, VX, VY, V, control='flc_tank'):
    '''
    Wrapper for flc_tank and flc_disk. The state should really be an object, with properties. 
    '''
    # pyplot displays the robot in the coordinate frame, and this is its state.
    # The potential and gradient matrices have integer indices, thus this conversion:
    xi = pf.location_to_index(state, Mx, My)
    height = V[xi[0], xi[1]]
    error = desired_height - height
    gradient = np.array([VX[xi[0], xi[1]], VY[xi[0], xi[1]]])
    velocity = state.flatten()[2]
    if(control == 'tank'):
        # Assume state includes θ
        dθ = flc_tank(gradient, error)
        dx = velocity * np.cos(dθ)
        dy = velocity * np.sin(dθ)
        dX = np.array([[dx, dy, 0, dθ]]).T
        return dX
    if(control == 'disk'):
        # Assume state does not include θ
        dX = np.zeros((3,1))
        dx, dy = flc_disk(gradient, error) 
        dX = np.array([[dx, dy, 0]]).T * velocity
        return dX
    if(control == 'disk_pid' or control == 'pid_disk'):
        error_sum = state[3] + error
        dx, dy = flc_disk_pid(gradient, error, error_sum)
        dx *= velocity
        dy *= velocity
        dX = np.array([[dx, dy, 0, error_sum]]).T
        return dX
    if(control == 'fp_disk'):
        # The expected gradient is NOT the same as in the above function calls.
        dx, dy = fp_disk(gradient, height)
        dx *= velocity
        dy *= velocity
        dX = np.array([[dx, dy, 0, 0]]).T
        #print(dx, dy, state[0], state[1])
        return dX        
    
    
def runcar(duration, dt=.1):
    # initialize variables
    x = np.array([[.2, 0, .1, 0]]).T  # x,y,v,θ
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    V_0 = 1  # desired height, or potential

    # make a controller that includes the gradient
    # the curve following controller does does not localize, but uses these:
    Mx, My, VX, VY, V = pf.make_vehicle_field(xmin, xmax, ymin, ymax, desired_potential=V_0, threshold=.1)
    controller = lambda state: follow_level_curve(state, V_0, Mx, My, VX, VY, V,
                                                  control='fp_disk')
    tank = False

    # run the animation
    for t in np.arange(0, duration, dt):
        # Reset matplotlib view
        plt.pause(0.001)
        plt.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # iterate the controller and draw vehicle
        x = x + dt * controller(x)
        if(tank):
            rl.draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x,y,θ
        else:
            rl.draw_disk(x[[0, 1]], 0.2, ax, 'red') # x,y
            
        pf.draw_field(normalize=True, animation_vars=(V_0, xmin, xmax, ymin, ymax), fig=fig)


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
