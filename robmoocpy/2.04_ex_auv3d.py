from roblib import *


fig = figure()
ax = Axes3D(fig)


def translate(R,x):
    return(R + array([[x[0]]*R.shape[1],[x[1]]*R.shape[1],[x[2]]*R.shape[1]]))
      
        

def draw(x):
    x = x.flatten()    
    M = array([ [0.0,0.0,10.0,0.0,0.0,10.0,0.0,0.0],
                [-1.0,1.0,0.0,-1.0,-0.2,0.0,0.2,1.0],
                [0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0]])
    M = eulermat(x[6],x[5],x[4]) @ M
    
    
    M = translate(M,x[0:3])   
    ax.clear()
    ax.set_xlim3d(-20,20)
    ax.set_ylim3d(-20,20)
    ax.set_zlim3d(0,40)
    ax.plot(M[0],M[1],M[2])
    ax.plot(M[0],M[1],0*M[2],color='black')
    ax.scatter(0,0,0,color='magenta')
       

def f(x,u):
    x=x.flatten()
    u=u.flatten()
    v=x[3]; ψ=x[4]; θ=x[5]; φ=x[6];
    return array([[v*cos(θ)*cos(ψ)] ,
                     [v*cos(θ)*sin(ψ)] ,
                     [-v*sin(θ)] ,
                     [u[0]] ,
                     [(sin(φ)/cos(θ)) * v * u[1] + (cos(φ)/cos(θ)) * v * u[2]] ,
                     [cos(φ) * v * u[1] - sin(φ) * v * u[2]] ,
                     [-0.1 * sin(φ) * cos(θ) + tan(θ) * v * (sin(φ) * u[1] + cos(φ) * u[2])]])
    
           
x = array([[0,0,10,15,0,1,0]]).T
u = array([[0,0,0.1]]).T
dt = 0.05
for t in arange(0,5,dt):
    xdot=f(x,u)
    x = x + dt * xdot
    draw(x)
    pause(0.001)