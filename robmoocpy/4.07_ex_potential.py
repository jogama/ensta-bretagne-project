from roblib import *

def f(x,u):
    x,u  = x.flatten(), u.flatten()
    v,θ = x[2],x[3]    
    return array([ [ v*cos(θ) ],[ v*sin(θ) ], [u[0]], [u[1]]])
    
def draw_field():
    Mx    = arange(xmin,xmax,0.3)
    My    = arange(ymin,ymax,0.3)
    X1,X2 = meshgrid(Mx,My)
    VX = -X1
    VY = -X2
    R=sqrt(VX**2+VY**2)
    quiver(Mx,My,VX/R,VY/R)
    return()

x    = array([[4,-3,1,2]]).T #x,y,v,θ
dt   = 0.1
fig  = figure(0)
ax   = fig.add_subplot(111, aspect='equal')
xmin,xmax,ymin,ymax=-5,5,-5,5

for t in arange(0,2,dt):
    pause(0.001)
    cla()
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    phat = array([[1],[2]])
    qhat = array([[3],[4]])        
    draw_disk(qhat,0.3,ax,"magenta")
    draw_disk(phat,0.2,ax,"green")
    u = array([[0],  [0.3]])
    x = x + dt*f(x,u)    
    draw_tank(x[[0,1,3]],'red',0.2) # x,y,θ
    draw_field()
    


