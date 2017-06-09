from roblib import *
def f(x,u):
    """equation d'évolution"""
    x,u  = x.flatten(), u.flatten()
    xdot = array([[x[3]*cos(x[4])*cos(x[2])],
                  [x[3]*cos(x[4])*sin(x[2])],
                  [   x[3]*sin(x[4])/3     ],
                  [          u[0]          ],
                  [          u[1]          ]])
    return(xdot)


def draw_field(xmin,xmax,ymin,ymax):
    Mx    = arange(xmin,xmax,2)
    My    = arange(ymin,ymax,2)
    X1,X2 = meshgrid(Mx,My)
    VX    = X2
    VY    = -(0.01*(X1**2)-1)*X2-X1
    VX    = VX/sqrt(VX**2+VY**2)
    VY    = VY/sqrt(VX**2+VY**2)
    quiver(Mx,My,VX,VY)
    return()


x    = array([[0,5,pi/2,30,0.6]]).T
dt   = 0.01
fig  = figure(0)
ax   = fig.add_subplot(111, aspect='equal')
for t in arange(0,5,dt):
    pause(0.001)
    cla()
    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    u    = array([[0],[0]])
    x    = x +dt*f(x,u)
    draw_field(-40,40,-40,40)
    draw_car(x)
show()


