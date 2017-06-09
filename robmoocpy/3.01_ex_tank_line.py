from roblib import *

def f(x,u):
    x     = x.flatten()
    θ = x[2]
    return array([[cos(θ)], [sin(θ)],[u]])

x=array([[-20],[-10],[4]])
u=1
dt= 0.1
a,b = array([[-30],[-4]]), array([[30],[6]])
fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')

for t in arange(0,50,dt):
    pause(0.001)
    cla()
    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    draw_tank(x,'darkblue')
    plot([a[0,0],b[0,0]],[a[1,0],b[1,0]],'r')
    plot(a[0,0],a[1,0],'ro')
    plot(b[0,0],b[1,0],'ro')
    x   = x+dt*f(x,u)
show()
    

