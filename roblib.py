#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, pi, cos, sin, sqrt, tan, arctan, arctan2, tanh, arcsin,\
    exp, dot, array, log, inf, eye, zeros, ones, inf, size,\
    arange, reshape, concatenate, vstack, hstack, diag, median, sign, sum, meshgrid, cross, linspace, append
from matplotlib.pyplot import *
from numpy.random import randn, rand
from numpy.linalg import inv, det, norm, eig
from scipy.linalg import sqrtm, expm, norm, block_diag
from scipy.signal import place_poles
from mpl_toolkits.mplot3d import Axes3D
from math import factorial

from matplotlib.patches import Ellipse, Rectangle, Circle, Wedge, Polygon, Arc

from matplotlib.collections import PatchCollection


# Unicode https://en.wikipedia.org/wiki/List_of_Unicode_characters
# for instance to get θ : shift + ctr + U03B8
# U+03B1 α alpha;  U+03B2 β beta; U+03B3;	 Gamma 	0419; U+03B4 δ Delta;
# U+03B5 ε Epsilon;  U+03B6 Zeta; U+03B7 Eta; U+03B8 θ Theta;
# U+03BB Lambda; U+03BC Mu; U+03BD Nu; U+03BE Xi; U+03C0 Pi; U+03C1 ρ Rho;
# U+03C3 Sigma; U+03C4 Tau; U+03C6 φ Phi; U+03C8 ψ Psi; U+03C9 Omega ω
# U+0393 Γ

'''
Origin: ENSTA Bretagne
    RobMOOC
    https://www.ensta-bretagne.fr/jaulin/robmooc.html
Author: Luc Jaulin
Modified by Jonathan Garcia-Mallen
'''


def eulermat(φ, θ, ψ):
    Ad_i = array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ad_j = array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Ad_k = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    M = expm(ψ * Ad_k) @ expm(θ * Ad_j) @ expm(φ * Ad_i)
    return(M)


def angle(x):
    x = x.flatten()
    return arctan2(x[1], x[0])


def move_motif(M, x, y, θ):
    M1 = ones((1, len(M[1, :])))
    M2 = vstack((M, M1))
    R = array([[cos(θ), -sin(θ), x], [sin(θ), cos(θ), y]])
    return(R @ M2)


def translate_motif(R, x):
    return R + x @ ones((1, R.shape[1]))

#    def translate3D(R,x,y,z):
#    return   R+array([[x],[y],[z]])@ones((1,R.shape[1]))

#    (R + array([[x]*R.shape[1],[y]*R.shape[1],[z]*R.shape[1]]))


def draw_tank(x, col='darkblue', r=1):
    x = x.flatten()
    M = r * array([[1, -1, 0, 0, -1, -1, 0, 0, -1, 1, 0, 0, 3, 3, 0], [-2, -2, -2, -1, -1, 1, 1, 2, 2, 2, 2, 1, 0.5, -0.5, -1]])
    M = move_motif(M, x[0], x[1], x[2])
    plot(M[0], M[1], col, 2)


def draw_ellipse(c, Γ, η, ax, col):  # Gaussian confidence ellipse with artist
    # draw_ellipse(array([[1],[2]]),eye(2),0.9,ax,[1,0.8-0.3*i,0.8-0.3*i])
    if (norm(Γ) == 0):
        Γ = Γ + 0.001 * eye(len(Γ[1, :]))
    A = sqrtm(-2 * log(1 - η) * Γ)
    w, v = eig(A)
    v1 = array([[v[0, 0]], [v[1, 0]]])
    v2 = array([[v[0, 1]], [v[1, 1]]])
    f1 = A @ v1
    f2 = A @ v2
    φ = (arctan2(v1[1, 0], v1[0, 0]))
    alpha = φ * 180 / 3.14
    e = Ellipse(xy=c, width=2 * norm(f1), height=2 * norm(f2), angle=alpha)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)
    e.set_facecolor(col)


def draw_disk(c, r, ax, col):
    # draw_disk(array([[1],[2]]),0.5,ax,"blue")
    e = Ellipse(xy=c, width=2 * r, height=2 * r, angle=0)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)
    e.set_facecolor(col)


def draw_box(x1, x2, y1, y2, ax, col):
    c = array([[x1], [y1]])
    rect = Rectangle(c, width=x2 - x1, height=y2 - y1, angle=0)
    rect.set_facecolor(array([0.4, 0.3, 0.6]))
    ax.add_patch(rect)
    rect.set_clip_box(ax.bbox)
    rect.set_alpha(0.7)
    rect.set_facecolor(col)


def draw_polygon(P, ax, col):
    patches = []
    patches.append(Polygon(P, True))
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4, color=col)
    ax.add_collection(p)


def draw_arc(c, a, theta, color):
    s = arange(0, abs(theta), 0.01)
    s = sign(theta) * s
    d = a - c
    r = norm(d)
    alpha = angle(d)
    w = c@ones((1, size(s))) + r * array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])@array([cos(s), sin(s)])
    plot(w[0, :], w[1, :], color, linewidth=3)


def draw_arrow(x, y, θ, L, color):
    e = 0.2
    M1 = L * array([[0, 1, 1 - e, 1, 1 - e], [0, 0, -e, 0, e]])
    M = np.append(M1, [[1, 1, 1, 1, 1]], axis=0)
    R = array([[cos(θ), -sin(θ), x], [sin(θ), cos(θ), y], [0, 0, 1]])
    M = R@M
    plt.plot(M[0, :], M[1, :], color)


def draw_sailboat(x, δs, δr, ψ, awind):
    x = x.flatten()
    θ = x[2]
    hull = array([[-1, 5, 7, 7, 5, -1, -1, -1], [-2, -2, -1, 1, 2, 2, -2, -2], [1, 1, 1, 1, 1, 1, 1, 1]])
    sail = array([[-7, 0], [0, 0], [1, 1]])
    rudder = array([[-1, 1], [0, 0], [1, 1]])
    R = array([[cos(θ), -sin(θ), x[0]], [sin(θ), cos(θ), x[1]], [0, 0, 1]])
    hull = R@hull
    Rs = array([[cos(δs), -sin(δs), 3], [sin(δs), cos(δs), 0], [0, 0, 1]])
    Rr = array([[cos(δr), -sin(δr), -1], [sin(δr), cos(δr), 0], [0, 0, 1]])
    sail = R@Rs@sail
    rudder = R@Rr@rudder
    draw_arrow(x[0] + 5, x[1], ψ, 5 * awind, 'red')
    plot(hull[0, :], hull[1, :], 'black')
    plot(sail[0, :], sail[1, :], 'red')
    plot(rudder[0, :], rudder[1, :], 'red')


def draw_car(x):
    x = x.flatten()
    M = array([[-1, 4, 5, 5, 4, -1, -1, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, 3, 3, 3],
               [-2, -2, -1, 1, 2, 2, -2, -2, -2, -3, -3, -3, -3, 3, 3, 3, 3, 2, 2, 3, -3], ])

    M = move_motif(M, x[0], x[1], x[2])
    plot(M[0], M[1], "blue", 2)
    W = array([[-1, 1], [0, 0]])  # Front Wheel
    Wr = move_motif(W, 3, 3, x[4])
    Wr = move_motif(Wr, x[0], x[1], x[2])
    Wl = move_motif(W, 3, -3, x[4])
    Wl = move_motif(Wl, x[0], x[1], x[2])

    plot(Wr[0, :], Wr[1, :], 'magenta', linewidth=2)
    plot(Wl[0, :], Wl[1, :], 'magenta', linewidth=2)


def tondarray(M):
    if type(M) == float:
        return array([[M]])
    elif type(M) == int:
        return array([[M]])
    else:
        return M


def mvnrnd2(x, G):
    n = len(x)
    x1 = x.reshape(n)
    y = np.random.multivariate_normal(x1, G).reshape(n, 1)
    return(y)


def mvnrnd1(G):
    G = tondarray(G)
    n = len(G)
    x = array([[0]] * n)
    return(mvnrnd2(x, G))


def kalman_predict(xup, Gup, u, Γα, A):
    Γ1 = A @ Gup @ A.T + Γα
    x1 = A @ xup + u
    return(x1, Γ1)


def kalman_correc(x0, Γ0, y, Γβ, C):
    S = C @ Γ0 @ C.T + Γβ
    K = Γ0 @ C.T @ inv(S)
    ytilde = y - C @ x0
    Gup = (eye(len(x0)) - K @ C) @ Γ0
    xup = x0 + K@ytilde
    return(xup, Gup)


def kalman(x0, Γ0, u, y, Γα, Γβ, A, C):
    xup, Gup = kalman_correc(x0, Γ0, y, Γβ, C)
    x1, Γ1 = kalman_predict(xup, Gup, u, Γα, A)
    return(x1, Γ1)


def demo_draw():
    fig = figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    c = array([[5], [0]])
    e = Ellipse(xy=c, width=13.0, height=2.0, angle=45)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.9)
    e.set_facecolor(array([0.7, 0.3, 0.6]))

    rect = Rectangle((1, 1), width=5, height=3)
    rect.set_facecolor(array([0.4, 0.3, 0.6]))
    ax.add_patch(rect)

    pause(0.2)
    draw_tank(array([[-7], [5], [1]]))
    draw_tank(array([[-7], [5], [1]]), 'red', 0.2)

    draw_car(array([[1], [2], [3], [4], [0.5]]))

    c = array([[-2], [-3]])
    G = array([[2, -1], [-1, 4]])
    draw_ellipse(c, G, 0.9, ax, [0.8, 0.8, 1])
    P = array([[5, -3], [9, -10], [7, -4], [7, -6]])
    draw_polygon(P, ax, 'green')

    draw_disk(array([[-8], [-8]]), 2, ax, "blue")

    draw_arc(array([[0], [5]]), array([[4], [6]]), 2, 'red')

    show()  # only at the end. Otherwize, it closes the figure in a terminal mode


def demo_animation():
    fig = figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    for t in arange(0, 5, 0.1):
        pause(0.01)  # needed. Otherwize, draws only at the end
        cla()
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        draw_car(array([[t], [2], [3 + t], [4], [5 + t]]))
        c = array([[-2 + 2 * t], [-3]])
        G = array([[2 + t, -1], [-1, 4 + t]])
        draw_ellipse(c, G, 0.9, ax, [0.8, 0.8, 1])
#        if (t>50)&(k%2000==0):
#            fig.savefig('convoy'+str(k)+'.pdf', dpi=fig.dpi)
    show()


def demo_random():
    N = 1000
    xbar = array([[1], [2]])
    Γx = array([[3, 1], [1, 3]])
    X = randn(2, N)
    Y = rand(2, 3)
    print("Y=", Y)
    X = (xbar @ ones((1, N))) + sqrtm(Γx) @ X
    xbar_ = mean(X, axis=1)
    Xtilde = X - xbar @ ones((1, N))
    Γx_ = (Xtilde @ Xtilde.T) / N
    fig = figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    cla()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    draw_ellipse(xbar, Γx, 0.9, ax, [1, 0.8, 0.8])
    pause(0.5)
    ax.scatter(X[0], X[1])
    pause(0.3)
    plot()


def sawtooth(x):
    return (x + pi) % (2 * pi) - pi   # or equivalently   2*arctan(tan(x/2))


if __name__ == "__main__":

    R = zeros((3, 4))
    x = [[1], [2], [3]]
    R1 = translate_motif(R, x)
    print('R1=', R1)

    demo_draw()


#     demo_animation()
#    demo_random()


#    M = array([ [1, 2], [5, 6], [9, 10]])
#    print(M)
#    x=array([[1], [2]])
#    x2= M@x  #multiplication dans Python 3
#
#    G = [[1, 0], [0, 1]]
#    x3=mvnrnd2(x,G)
#    print(x3)
#
#    x4=mvnrnd1(G)
#    print(x4)
#
#    draw_box(-15,15,-15,15,'blue',4)
#    x=array([[2], [3], [1], [0], [0.5]])
#    draw_car(x)
#    axis ('equal')
#    draw_tank(-2,-3,-1)
#    print(randn())
#
#    A = array([[0,0,1,0],[0,0,0,1],[0,2,0,0],[0,3,0,0]])
#    B = array([[0,0,4,5]]).T
#    poles = [-2,-2.1,-2.2,-2.3]
#    K = place_poles(A,B,poles).gain_matrix
#    print(K)
#
#
