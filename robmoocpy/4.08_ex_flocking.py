from roblib import *

def f(x,u):
    x=x.flatten()
    θ = x[2]
    return array([[cos(θ)], [sin(θ)], [u]])

    
fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')
m   = 20
X   = 20*randn(3,m)
dt  = 0.2

for t in arange(0,10,dt):
    pause(0.001)
    cla()
    ax.set_xlim(-60,60)
    ax.set_ylim(-60,60)
    for i in range(m):
        xi=X[:,i].flatten()
        xi=xi.reshape(3,1)
        draw_tank(xi,'b')
        u=0
        xi=xi+f(xi,u)*dt        
        X[:,i]  = xi.flatten()        



