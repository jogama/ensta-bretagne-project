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

def draw_field(normalize=True, three_d=False, animation_vars=None):
    if animation_vars != None:
        x, xmin, xmax, ymin, ymax = animation_vars
    # x and y coordinates/locations of the arrows
    Mx    = np.arange(xmin,xmax,0.3)
    My    = np.arange(ymin,ymax,0.3)

    # Draw islands
    X1, X2 = np.meshgrid(Mx,My)
    
    island_center1 = [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2]
    island_center2 = [xmin + (xmax - xmin) * .2, ymin + (ymax - ymin) * .9]
    island_sum  = mnorm(island_center1, cov=2).pdf(np.dstack((X1, X2))) / 2
    island_sum += mnorm(island_center2, cov=2).pdf(np.dstack((X1, X2))) 
    

    # VX, VY are x and y components
    VX = np.gradient(island_sum)[1]
    VY = np.gradient(island_sum)[0]
    
    if(normalize):
        R=np.sqrt(VX**2+VY**2)
        plt.quiver(Mx,My,VX/R,VY/R)
    else:
        plt.quiver(Mx,My,VX,VY)
        
    if(three_d):
        from mpl_toolkits.mplot3d import axes3d
        fig2 = plt.figure('0')
        ax1 = fig2.add_subplot(111, projection='3d')
        ax1.plot_surface(X1, X2, island_sum)

    return()

def f(x,u):
    x,u  = x.flatten(), u.flatten()
    v,θ = x[2],x[3]    
    return np.array([ [ v*np.cos(θ) ],[ v*np.sin(θ) ], [u[0]], [u[1]]])


if __name__ == "__main__":
    x    = np.array([[4,-3,1,2]]).T #x,y,v,θ
    dt   = 0.2
    fig  = plt.figure('Two normalized Island')
    ax   = fig.add_subplot(111, aspect='equal')
    xmin,xmax,ymin,ymax=-5,5,-5,5

    draw_field(three_d=True, normalize=False)
    plt.show()
