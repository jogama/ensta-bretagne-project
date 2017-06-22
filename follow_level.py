#!/usr/bin/python3

from scipy.stats import multivariate_normal as mnorm
from potential_fields import draw_field
from numpy.linalg import norm
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np
import pdb

"""
There is not a single convention as to the definition of the state in robmoocpy code. Here, it will be a numpy array x = [x, y, θ]. This would more elegantly bre represented as a vector with respect to the origin. However, we will first implement this in a "yo, it just works this way" kind of way. Aiming for easiest correct implementation, even if it is somewhat cumbersome. 
"""
np.set_printoptions(threshold=np.inf)  # debug


def location_to_index(loc, Mx, My):
    ''' Converts location in plot to index for the gradient and potentia matrices.
    Args:
      loc: iterable where loc[0] anx loc[1] are the x and y coordinates of the
         location, respectively.
      Mx and My:
         1-d numpy arrays with linearly increasing values. They are the x and y
         coordinates of the gradient and potential values, respectively.
    Returns:
      index: numpy array of shape (2,) with index corresponding to the gradient
         and potential arrays.
    '''
    # We assume the difference between the first two values in Mx, My, is the
    #   step size for the entire matrix, and extrapolate.
    #   There's still probably an easier way.
    step_size_guess = np.abs(np.array([Mx[0] - Mx[1], My[0] - My[1]]))
    x = np.array([loc[0], loc[1]]).flatten()
    x_0 = np.array([Mx[0], My[0]])
    index = np.round((x - x_0) / step_size_guess)

    # Prevent out-of-bounds errors and return
    index[0] = np.clip(index[0], 0, Mx.size - 1)
    index[1] = np.clip(index[1], 0, My.size - 1)
    return index.astype(int)


def make_islands(xmin, xmax, ymin, ymax):
    '''
    Simply makes two 'islands' from gaussian distributions, joins them,
    and returns data on the islands. Split off from draw_field, as the
    data is needed by other functions. Could be better named.

    Args:
      the bounds of the region
    Returns:
      a 5-tuple (X, Y, VX, VY, V)
      X, Y:   arrow locations for plt.quiver
      V: "potential" at every point; the height of the sea floor
      VX, VY: The gradient in the X and Y direction, respectively, of V.
    '''
    # x and y coordinates/locations of the arrows
    Mx = np.arange(xmin, xmax, .3)
    My = np.arange(ymin, ymax, .3)
    X1, X2 = np.meshgrid(Mx, My)

    # Draw islands
    island_center0 = [xmin + (xmax - xmin) * .5, ymin + (ymax - ymin) * .5]
    V = mnorm(island_center0, cov=2).pdf(np.dstack((X1, X2))) * 50

    # VX, VY are x and y components.
    # I don't know why they must be flipped; it's worrisome
    VX = np.gradient(V)[1]  # (np.arange(Mx.size))#
    VY = np.gradient(V)[0]  # (np.arange(Mx.size))#

    return (Mx, My, VX, VY, V)


def make_vehicle_field(xmin, xmax, ymin, ymax, desired_potential, threshold=.7):
    '''makes the potential field that the vehicle would follow, clockwise'''
    Mx, My, GX, GY, V = make_islands(xmin, xmax, ymin, ymax)
    VX, VY = -GY, GX

    # Make field attractive about the desired potential and within the threshold
    desired_pot_low = desired_potential + desired_potential * threshold
    desired_pot_high = desired_potential - desired_potential * threshold
    less_than_wanted = (V < desired_pot_low).astype(np.int)
    more_than_wanted = (V > desired_pot_high).astype(np.int)
    wanted = ((V > desired_pot_low) & (V < desired_pot_high)).astype(np.int)
    VX = (VX + GX) * less_than_wanted + (VX - GX) * more_than_wanted + VX * wanted
    VY = (VY + GY) * less_than_wanted + (VY - GY) * more_than_wanted + VY * wanted

    return Mx, My, VX, VY, V


def desired_heading(heading_vector, height, height_desired):
    '''
    Should almost certainly be combined with control().
    Args:
      heading_vector: numpy array of shape (2,)
      height: vehicle's distance from seafloor below. Can be thought of as potential.
      height_desired: The target level, or height, to maintain.
    '''
    assert(heading_vector.shape == (2,))
    θ_desired = np.arctan2(heading_vector[1], heading_vector[0])
    return θ_desired


def control(x, θ_desired, θ_previous, dt, a2=1, b2=1):
    '''
    Args:
      x: state vector as a numpy array
    Remarks: See "Mobile Robotics" sec 3.1.2 (L. Jaulin) for
      derivation
    '''
    u = np.zeros((2, 1))
    u[0] = 0  # no acceleration

    # Proportional-Derivative control of heading
    θ = x.flatten()[3]
    u[1] = a2 * rl.sawtooth(θ_desired - θ)  # - b2 * np.sin(θ - θ_previous) / dt
    return u


def f(x, u):
    '''
    evolution function
    Args:
      x, u: state and input vectors, respectively.
        x is a numpy arrays; u is scalar
    Returns:
      xdot: the approximate time derivative of the state.
        of type numpy array with shape (x.size,1).
    '''
    x, u = x.flatten(), u.flatten()
    v, θ = x[2], x[3]

    return np.array([[v * np.cos(θ), v * np.sin(θ), u[0], u[1]]]).T


# Euler's approx
def runcar(duration, dt=.1):
    # initialize variables
    x = np.array([[-4.6, 0, 2, np.pi / 4]]).T  # x,y,v,θ
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    V_0 = 1  # desired height, or potential
    Mx, My, VX, VY, V = make_vehicle_field(xmin, xmax, ymin, ymax,
                                           desired_potential=V_0, threshold=.1)

    # run the animation
    θ_previous = x.flatten()[3]
    for t in np.arange(0, duration, dt):
        # Reset matplotlib view
        plt.pause(0.001)
        plt.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Controller inputs are height and r, the direction the vehicle's field points in
        xi = location_to_index(x, Mx, My)
        height = V[xi[0], xi[1]]
        r = np.array([VX.T[xi[0], xi[1]], VY.T[xi[0], xi[1]]])

        # Get the displacements * velocity to iterate the controller:
        θ_d = desired_heading(r, height, V_0)
        u = control(x, θ_d, θ_previous, dt)
        x = x + dt * f(x, u)

        # Retain previous θ for PD control (consider including this in x)
        θ_previous = x.flatten()[3]

        # for debugging
        c1 = plt.Circle((Mx[xi[0]], My[xi[1]]), .2, color='b')
        ax.add_artist(c1)
        c2 = plt.Circle((x[0], x[1]), .2, color='g')
        ax.add_artist(c2)
        plt.arrow(Mx[xi[0]], My[xi[1]], r[0] / norm(r), r[1] / norm(r),
                  head_width=.25)  # debug
        plt.arrow(-4, -4, np.cos(θ_d), np.sin(θ_d), head_width=.25)
        plt.text(-4, -4.5, str(θ_d / np.pi) + 'π')
        plt.text(-0, -4.5, 'r = ' + str(r))

        # Draw vehicle and field
        rl.draw_tank(x[[0, 1, 3]], 'red', 0.1)  # x,y,θ
        draw_field(fig=fig, field=(Mx, My, VX, VY, V, V_0))


if __name__ == "__main__":
    runcar(7, dt=.1)
