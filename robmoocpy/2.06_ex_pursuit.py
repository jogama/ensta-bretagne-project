from roblib import *

def f(x,u):
    x=x.flatten()
    u=u.flatten()
    return (array([[u[0]*cos(x[2])], [u[0]*sin(x[2])],[u[1]]]))

def control(xa,xb,v):
    u=array([[0],[0]]) #TO DO
    return u    

fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')
dt = 0.1

xa = array([[-10], [-10],[0]])
xb = array([[-5],[-5],[0]])


for t in arange(0,10,dt) :
    pause(0.01)  
    cla()
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    v = array([[3],[sin(0.2*t)]])
    u=control(xa,xb,v)
    draw_tank(xa,'blue')  	
    draw_tank(xb,'red')  	
    xa = xa + dt*f(xa,u)
    xb = xb + dt*f(xb,v)
show()


    
