from roblib import *

def draw_crank(x): 
    θ1=x[0,0]
    θ2=x[1,0]
    z=L1*array([[cos(θ1)],[sin(θ1)]])
    y=z+L2*array([[cos(θ1+θ2)],[sin(θ1+θ2)]])
    plot( [0,z[0,0],y[0,0]],[0,z[1,0],y[1,0]],'magenta', linewidth = 2)   
    draw_disk(c,r,ax,"cyan")


L1,L2 = 4,3
c = array([[1],[2]])
r=4
dt = 0.05

x = array([[-1],[1]])


def f(x):
    θ1=x[0,0]
    θ2=x[1,0]
    dθ1=1
    dθ2=2
    return(array([[dθ1],[dθ2]]))
    

fig = figure(0)    
ax = fig.add_subplot(111, aspect='equal')

for t in arange(0,1,dt) :
    pause(0.01) 
    cla()       
    ax.set_xlim(-4,8)
    ax.set_ylim(-4,8)         
    draw_crank(x)
    x = x + dt*f(x)  
show()  

