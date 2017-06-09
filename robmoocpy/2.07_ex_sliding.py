
from roblib import *

def f(x,u):
    x=x.flatten()
    u=u.flatten()
    return (array([[x[3]*cos(x[2])],     [x[3]*sin(x[2])],  [u[0]],[u[1]]]))
    
    
    
def control(x,w,dw,ddw):
    u=array([[0],[0]]) #TO DO
    return u    

   


fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')
dt = 0.02
x = array([[10],[0],[1],[1]])
u = array([[1],[1]])
L=10
s = arange(0,2*pi,0.01)
for t in arange(0,15,dt) :
    pause(0.01)  
    cla()
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    plot(L*cos(s), L*sin(3*s),color='magenta')
    draw_tank(x,'red')  
    w=array([[0],[0]])  #TO DO
    dw=array([[0],[0]])  #TO DO
    ddw=array([[0],[0]])  #TO DO
    u=control(x,w,dw,ddw)
    draw_disk(w,0.5,ax,"red")    
    x = x + dt*f(x,u)
show()




    

    



    
    
    

