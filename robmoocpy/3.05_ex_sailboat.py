from roblib import *

         
    
def f(x,u):
    x,u=x.flatten(),u.flatten()
    θ=x[2]; v=x[3]; w=x[4]; deltar=u[0]; deltasmax=u[1];
    w_ap = array([[awind*cos(psi-θ) - v],
                    [awind*sin(psi-θ)]])
    psi_ap = np.arctan2(w_ap[1,0], w_ap[0,0])
    a_ap=norm(w_ap)
    sigma = cos(psi_ap) + cos(deltasmax)
    if sigma < 0 :
        deltas = pi + psi_ap
    else :
        deltas = -sign(sin(psi_ap))*deltasmax
    fr = p[4]*v*sin(deltar)
    fs = p[3]*a_ap* sin(deltas - psi_ap)
    dx=v*cos(θ) + p[0]*awind*cos(psi)
    dy=v*sin(θ) + p[0]*awind*sin(psi)
    dv=(fs*sin(deltas)-fr*sin(deltar)-p[1]*v**2)/p[8]
    dw=(fs*(p[5]-p[6]*cos(deltas)) - p[7]*fr*cos(deltar) - p[2]*w*v)/p[9]
    xdot=array([ [dx],[dy],[w],[dv],[dw]])
    return xdot,deltas
    
    
    
    
    
fig = figure(0)
ax = plt.subplot(111, aspect='equal')
p = [0.1,1,6000,1000,2000,1,1,2,300,10000]
x = array([[10,-40,-3,1,0]]).T   #x=(x,y,θ,v,w)

u = array([[0.1,1]]).T
dt = 0.1
awind = 2
psi = -2  
a = np.array([[-50,-100]]).T   
b = np.array([[50,100]]).T
q = 1         
                  
 


for t in arange(0,10,0.1):
    ax.clear()
    cla()
    ax.set_xlim([-100,100])
    ax.set_ylim([-60,60])
    plot([a[0,0],b[0,0]],[a[1,0],b[1,0]],'red')
    u=array([[0],[1]])
    xdot,deltas=f(x,u)
    x = x + dt*xdot
    draw_sailboat(x,deltas,u[0,0],psi,awind)
    pause(0.001)
        


        