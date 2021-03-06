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


def make_islands(xmin, xmax, ymin, ymax):
    '''
    Simply makes 'islands' from gaussian distributions, sums them,
    and returns data on the islands.

    Args:
      the bounds of the region
    Returns:
      a 5-tuple (X, Y, VX, VY, V)
      X, Y:   arrow locations for plt.quiver
      V: "potential" at every point; the height of the sea floor
      VX, VY: The gradient in the X and Y direction, respectively, of V.
    '''
    # x and y coordinates/locations of the arrows
    Mx = np.arange(xmin, xmax, .1)
    My = np.arange(ymin, ymax, .1)
    X1, X2 = np.meshgrid(Mx, My)

    # Draw islands
    island_center0 = [xmin + (xmax - xmin) * .5, ymin + (ymax - ymin) * .5]
    island_center1 = [xmin + (xmax - xmin) * .4, ymin + (ymax - ymin) * .7]
    island_center2 = [xmin + (xmax - xmin) * .3, ymin + (ymax - ymin) * .4]
    island_center3 = [xmin + (xmax - xmin) * .7, ymin + (ymax - ymin) * .5]
    V = mnorm(island_center0, cov=2).pdf(np.dstack((X1, X2))) / 2
    V += mnorm(island_center2, cov=1).pdf(np.dstack((X1, X2))) / 3
    V += mnorm(island_center1, cov=2).pdf(np.dstack((X1, X2))) / 2
    V += mnorm(island_center3, cov=0.5).pdf(np.dstack((X1, X2))) / 10

    # VX, VY are x and y components.
    # I don't know why they must be flipped; it's worrisome
    VX = np.gradient(V, axis=1)
    VY = np.gradient(V, axis=0)
    return (Mx, My, VX, VY, V)

def show_3d_surface(Mx, My, V):
    '''debugging utiilty to view the "ocean floor" in 3D'''
    from mpl_toolkits.mplot3d import axes3d
    X1, X2 = np.meshgrid(Mx, My)
    fig2 = plt.figure('Island 3D')
    ax1 = fig2.add_subplot(111, projection='3d')
    ax1.plot_surface(X1, X2, V)
    return()


def draw_field(normalize=True, three_d=False, animation_vars=None, fig=None, field=None):
    if(fig is None):
        fig = plt.figure('Island Potentials')
    ax = fig.add_subplot(111, aspect='equal')

    if field is None and animation_vars is not None:
        from follow_level import make_vehicle_field
        dp, xmin, xmax, ymin, ymax = animation_vars
        Mx, My, VX, VY, V = make_vehicle_field(
            xmin, xmax, ymin, ymax, desired_potential=dp)
    else:
        Mx, My, VX, VY, V, dp = field

    X1, X2 = np.meshgrid(Mx, My)

    if(normalize):
        R = np.sqrt(VX**2 + VY**2)
        plt.quiver(Mx, My, VX / R, VY / R)
    else:
        plt.quiver(Mx, My, VX, VY)

    # draw contour to be followed
    cs = plt.contour(Mx, My, V, levels=[dp])
    plt.clabel(cs, inline=1, fontsize=10)

    if(three_d):
        from mpl_toolkits.mplot3d import axes3d
        fig2 = plt.figure('Island 3D')
        ax1 = fig2.add_subplot(111, projection='3d')
        ax1.plot_surface(X1, X2, V)

    return()


if __name__ == "__main__":
    x = np.array([[4, -3, 1, 2]]).T  # x,y,v,θ
    dt = 0.2
    # xmin,xmax,ymin,ymax=-5,5,-5,5
    av = (.03, -5, 5, -5, 5)
    draw_field(three_d=True, normalize=True, animation_vars=av)
    plt.show()
