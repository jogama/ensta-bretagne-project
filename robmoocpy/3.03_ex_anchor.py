from roblib import *

def f(x,u):
    x    = x.flatten()
    return array([[5*cos(x[2])],[5*sin(x[2])],[u]])

        

x    = array([[15],[20],[1]])
dt   = 0.1

fig  = figure(0)
ax   = fig.add_subplot(111, aspect='equal')

for t in arange(0,50,dt):
    pause(0.001)
    cla()
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    draw_disk(array([[0],[0]]),10,ax,'cyan')
    u = 0.5
    draw_tank(x,'red')
    x = x+dt*f(x,u)            
show()

