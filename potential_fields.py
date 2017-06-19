#!/usr/bin/python3

'''
Author: Jonathan Garcia-Mallen

Demonstrates matplotlib's quiver to produce potential field plots. 3d plotting
as used to debug this, and is kept in the code as another example. The gausians
represent islands, hence the variable names.

Based off of 4.07_ex_potential.py
'''

from scipy.stats import multivariate_normal as mnorm
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np


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


def make_islands(xmin, xmax, ymin, ymax):
    '''
    Simply makes two 'islands' from gaussian distributions, joins them,
    and returns data on the islands. Split off from draw_field, as the
    data is needed by other functions. Could be better named.

    Args:
      the bounds of the region
    Returns:
      a 5-tuple (X, Y, VX, VY, V)
      X, Y:   arrow locations for plt.quiver
      V: "potential" at every point; the height of the sea floor
      VX, VY: The gradient in the X and Y direction, respectively, of V.
    '''
    # x and y coordinates/locations of the arrows
    Mx = np.arange(xmin, xmax, 0.3)
    My = np.arange(ymin, ymax, 0.3)

    # Draw islands
    X1, X2 = np.meshgrid(Mx, My)

    island_center1 = [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2]
    island_center2 = [xmin + (xmax - xmin) * .2, ymin + (ymax - ymin) * .9]
    island_sum = mnorm(island_center1, cov=2).pdf(np.dstack((X1, X2))) / 2
    island_sum += mnorm(island_center2, cov=2).pdf(np.dstack((X1, X2)))
    V = island_sum

    # VX, VY are x and y components
    VX = np.gradient(island_sum)[1]
    VY = np.gradient(island_sum)[0]

    return (Mx, My, VX, VY, V)


def make_vehicle_field(xmin, xmax, ymin, ymax, desired_potential=None, clockwise=True):
    '''makes the potential field that the vehicle would follow'''
    
    Mx, My, VY, VX, V = make_islands(xmin, xmax, ymin, ymax)
    if(desired_potential is None):
        desired_potential = np.mean(V)

    # Make field orthogonal to gradient
    if(clockwise):
        VX = -VX
    else:
        VY = -VY

    # Make field attractive about the desired potential
    
    return Mx, My, VX, VY, V

def draw_field(normalize=True, three_d=False, animation_vars=None):
    if animation_vars != None:
        x, xmin, xmax, ymin, ymax = animation_vars

    Mx, My, VX, VY, V = make_vehicle_field(xmin, xmax, ymin, ymax)
    X1, X2 = np.meshgrid(Mx, My)

    if(normalize):
        R = np.sqrt(VX**2 + VY**2)
        plt.quiver(Mx, My, VX / R, VY / R)
    else:
        plt.quiver(Mx, My, VX, VY)

    if(three_d):
        from mpl_toolkits.mplot3d import axes3d
        fig2 = plt.figure('0')
        ax1 = fig2.add_subplot(111, projection='3d')
        ax1.plot_surface(X1, X2, V)

    return()


def f(x, u):
    x, u = x.flatten(), u.flatten()
    v, θ = x[2], x[3]
    return np.array([[v * np.cos(θ)], [v * np.sin(θ)], [u[0]], [u[1]]])


if __name__ == "__main__":
    x = np.array([[4, -3, 1, 2]]).T  # x,y,v,θ
    dt = 0.2
    fig = plt.figure('Two normalized Island')
    ax = fig.add_subplot(111, aspect='equal')
    # xmin,xmax,ymin,ymax=-5,5,-5,5
    av = (None, -5, 5, -5, 5)
    draw_field(three_d=True, normalize=True, animation_vars=av)
    plt.show()
