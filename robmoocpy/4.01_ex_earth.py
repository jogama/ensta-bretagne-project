from roblib import *

def T(lx,ly,ρ):
    return ρ*array([
        [cos(ly)*cos(lx)],
        [cos(ly)*sin(lx)],
        [sin(ly)],
    ])

def draw_earth():
    a = pi/10
    Lx = arange(0, 2*pi+a, a)
    Ly = arange(-pi/2, pi/2+a, a)    
    M1 = T(0,-pi/2,ρ)
    for ly1 in Ly:
        for lx1 in Lx:
            M1 = hstack((M1, T(lx1,ly1,ρ)))                                   
    M2 = T(0,-pi/2,ρ)
    for lx1 in Lx:
        for ly1 in Ly:
             M2 = hstack((M2, T(lx1,ly1,ρ)))
    ax.plot(M1[0],M1[1],M1[2],color='gray')
    ax.plot(M2[0],M2[1],M2[2],color='gray')

        
def draw_rob(x):
    x = x.flatten()
    lx,ly,ψ = x[0],x[1],x[2]
    M = array([
        [ 0,  0,  10,  0,   0,   10,   0,   0, ],
        [ -1,  1,  0, -1,  -0.2,  0,  0.2,  1, ],
        [ 0,  0,   0,  0,   1,    0,   1,   0, ] ])
    
    Rlatlong = array([
        [ -sin(lx), -sin(ly)*cos(lx), cos(ly)*cos(lx)],
        [ cos(lx) , -sin(ly)*sin(lx), cos(ly)*sin(lx)],
        [ 0       ,      cos(ly)    ,     sin(ly)    ] ])
    
    M = Rlatlong @ eulermat(0,0,ψ) @ M
    M=translate_motif(M,T(lx,ly,ρ))
    ax.plot(M[0],M[1],M[2])
       
    
def f(x,u):
    x = x.flatten()
    lx,ly,ψ = x[0],x[1],x[2]
    return array([[cos(ψ)/(ρ*cos(ly))], [sin(ψ)/ρ], [u]])

    
fig = figure()
ax = Axes3D(fig)
ρ = 30; 
x   = array([[3],[1],[1]])
dt = 0.1

for t in arange(0,20,dt):
    ax.clear()
    ax.set_xlim3d(-ρ,ρ)
    ax.set_ylim3d(-ρ,ρ)
    ax.set_zlim3d(-ρ,ρ)
    u = 0.1 * randn(1)    
    x = x + dt*f(x,u)    
    draw_rob(x)
    draw_earth()
    pause(0.001)

