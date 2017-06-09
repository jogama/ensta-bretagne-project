from roblib import *

def f(x,u):
    x = x.flatten()
    θ = x[2]
    return array([[cos(θ)],[sin(θ)],[u]])

def control(x):
    u=0
    return u
    
    
x   = array([[0],[0],[0.1]])
dt  = 0.1
fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')

for t in arange(0,30,dt):
    pause(0.001)
    cla()
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    u = control(x)
    x = x + dt*f(x,u)    
    draw_tank(x,'red',0.3) 
    