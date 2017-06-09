from roblib import *

def f(x,u):
    x=x.flatten()
    u=u.flatten()
    return (array([[x[3]*cos(x[2])],     [x[3]*sin(x[2])],  [u[0]],[u[1]]]))



fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')

dt = 0.1
x = array([[0],[1],[pi/3],[1]])
u = array([[1],[1]])


for t in arange(0,10,dt) :
    pause(0.01)  
    cla()
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    draw_tank(x)  	
    x = x + dt*f(x,u)
show()