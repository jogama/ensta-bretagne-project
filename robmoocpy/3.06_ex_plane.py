from roblib import *

def draw(x,u,ax):
    x,u        = x.flatten(),u.flatten()
    uleft,uright = -u[1]+u[2], -u[1]-u[2]
    plane    = array([[0,  0, 6, 0,  0, 0,   0,   1, 6, 0],
                      [0, -1, 0, 1, -1, 0,   0,   0, 0, 0],
                      [0,  0, 0, 0,  0, 0,   1, 0.2, 0, 0],
                      [1,  1, 1, 1,  1, 1,   1,   1, 1, 1]])
    e        = 0.5
    aileron  = array([[-e,  0, 0, -e, -e],[-e, -e, e,  e, -e],
                      [ 0,  0, 0,  0,  0],[ 1,  1, 1,  1,  1]])
    
    R        = concatenate((
                            eulermat(-x[3],-x[4],x[5]),
                            array([[x[0],x[1],-x[2]]]).T
                            ), axis=1)
    R        = concatenate((R,array([[0, 0, 0, 1]])))
    
    ailLeft  = concatenate( (eulermat(0,uleft,0),array([[0,1-e,0]]).T),axis=1)
    ailLeft  = concatenate((ailLeft,array([[0,0,0,1]])),axis=0)
    ailLeft  = R @ ailLeft @ aileron    
    ailRight = concatenate( (eulermat(0,uright,0),array([[0,e-1,0]]).T),axis=1)
    ailRight = concatenate((ailRight,array([[0,0,0,1]])),axis=0)
    ailRight = R @ ailRight @ aileron    
    plane    = R @ plane
    
    ax.clear()
    ax.set_xlim3d(-100,100)
    ax.set_ylim3d(-100,100)
    ax.set_zlim3d(-10,100)
    ax.plot(plane[0,:],plane[1,:],plane[2,:],'blue')          # drone
    ax.plot(ailLeft[0,:],ailLeft[1,:],ailLeft[2,:],'red')     # aileron gauche
    ax.plot(ailRight[0,:],ailRight[1,:],ailRight[2,:],'red')  # aileron droit
    ax.plot(plane[0,:],plane[1,:],0*plane[2,:],'black')       # ombre du drone
    ax.plot(Cx0,Cy0,Cz0,'green')                              # cercle consigne


def f(x,u):
    v     = x[6:9]
    w     = x[9:12]
    x,u=x.flatten(),u.flatten()
    V     = norm(v)
    α = arctan(x[8]/x[6])
    β  = arcsin(x[7]/V)
    φ,θ,ψ = x[3],x[4],x[5]
    cf,sf,ct,st,tt,ca,sa,cb,sb = cos(φ),sin(φ),cos(θ),sin(θ),tan(θ),cos(α),sin(α),cos(β),sin(β)
    Fa= 0.002*(V**2)*array([[-ca*cb,ca*sb,sa],[sb,cb,0],[-sa*cb,sa*sb,-ca]])  \
            @  \
            array([[4+(-0.3+10*α+10*w[1,0]/V+2*u[2]+0.3*u[1])**2+abs(u[1])+3*abs(u[2])],
                   [-50*β + 10*(w[2,0]-0.3*w[0,0])/V],
                   [10+500*α+400*w[1,0]/V+50*u[2]+10*u[1]]])
    
    return concatenate((
                         eulermat(φ,θ,ψ) @ v,
                         array([[1,tt*sf,tt*cf],[0,cf,-sf],[0,sf/ct,cf/ct]]) @ w,
                         9.81*array([[-st],[ct*sf],[ct*cf]])+Fa+array([[u[0]],[0],[0]]) - cross(w.T,v.T).T,
                         array([ -w[2]*w[1]+0.1*(V**2)*(-β -2*u[2]+(-5*w[0]+w[2])/V),
                                w[2]*w[0]+0.1*(V**2)*(-0.1-2*α+0.2*u[2]-3*u[1]-30*w[1]/V),
                                0.1*w[0]*w[1]+0.1*(V**2)*(β+0.5*u[2]+0.5*(w[0]-2*w[2])/V)])
                         ),axis=0)
    

    
x    = array([[1, 0, 0, 0, 0.1, 0, 20, 0, 0, 0, 10, 0]]).T #[x;y;z;φ;θ;ψ;v;w]
dt   = 0.02
vbar,zbar,rbar = 15,-50,100  
a    = arange(0,2*pi+0.1, 0.1)
Cx0,Cy0,Cz0 = rbar*np.cos(a),rbar*np.sin(a),[-zbar]*len(a) #circle to follow

fig   = figure()
ax    = Axes3D(fig)


for t in arange (0,5,dt):
    u=array([[10],[0],[0]])
    x = x +dt*f(x,u)
    draw(x,u,ax)
    pause(0.01)
show()


