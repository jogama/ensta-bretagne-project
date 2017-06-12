#!/usr/bin/python3

# Author: Jonathan Garcia-Mallen

from scipy.stats import multivariate_normal as mnorm
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np

print("go!")

def draw_field(normalize=True):
    # x and y coordinates/locations of the arrows
    Mx    = np.arange(xmin,xmax,0.3)
    My    = np.arange(ymin,ymax,0.3)

    # Draw islands
    X1, X2 = np.meshgrid(Mx,My)
    
    island_center = [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2]
    island = mnorm(island_center, cov=1).pdf(np.dstack((X1, X2)))

    # VX, VY are x and y components
    VX = np.gradient(island)[1]
    VY = np.gradient(island)[0]
    
    if(normalize):
        R=np.sqrt(VX**2+VY**2)
        plt.quiver(Mx,My,VX/R,VY/R)
    else:
        plt.quiver(Mx,My,VX,VY)
    return()

def f(x,u):
    x,u  = x.flatten(), u.flatten()
    v,θ = x[2],x[3]    
    return np.array([ [ v*np.cos(θ) ],[ v*np.sin(θ) ], [u[0]], [u[1]]])

x    = np.array([[4,-3,1,2]]).T #x,y,v,θ
dt   = 0.2
fig  = plt.figure(0)
ax   = fig.add_subplot(111, aspect='equal')
xmin,xmax,ymin,ymax=-5,5,-5,5

draw_field()
plt.show()
for t in np.arange(0,2,dt):
    plt.pause(0.001)
    plt.cla()
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    phat = np.array([[1],[2]])
    qhat = np.array([[3],[4]])        
    rl.draw_disk(qhat,0.3,ax,"magenta")
    rl.draw_disk(phat,0.2,ax,"green")
    u = np.array([[0],  [0.3]])
    x = x + dt*f(x,u)    
    rl.draw_tank(x[[0,1,3]],'red',0.2) # x,y,θ
    draw_field()
