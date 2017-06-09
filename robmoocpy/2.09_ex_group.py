from roblib import *
from numpy import *


def f(x,u):
    x,u=x.flatten(),u.flatten()
    xdot = array([[x[3]*cos(x[2])],[x[3]*sin(x[2])],[u[0]],[u[1]]])
    return(xdot)

def control(x,w,dw,ddw):
    u=array([[0],[0]]) #TO DO
    return u    
    

fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')
m   = 20
X   = 10*randn(0,1,(4,m))
a,dt = 0.1,0.1

for t in arange(0,3,dt):
    pause(0.001)
    cla()
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    for i in range(m):        
        w = zeros((2,1)) #TO DO
        dw = zeros((2,1))  #TO DO
        ddw = zeros((2,1))#TO DO
        x=X[:,i].reshape(4,1)
        u       = control(x,w,dw,ddw)
        x=X[:,i].reshape(4,1)
        draw_tank(x,'b')
        x=x+f(x,u)*dt        
        X[:,i]  = x.flatten()
        plot([w[0][0]],[w[1][0]],'r+')

show()


