#!/usr/bin/python3

from scipy.stats import multivariate_normal as mnorm
from potential_fields import draw_field, make_islands
from numpy.linalg import norm
import matplotlib.pyplot as plt
import roblib as rl
import numpy as np

"""
Demonstration of a robot following a level curve. 
"""


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




def make_vehicle_field(xmin, xmax, ymin, ymax, desired_potential, threshold=.1):
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


def control(x, θ_desired, θ_previous, dt, a2=1, b2=1):
    '''
    Args:
      x: state vector as a numpy array
      θ_desired: desired vehicle angle in the global frame
      θ_previous: vehicle angle from previous iteration
      a2, b2: coefficients for proportional-derivative control, respectively,
        of the vehicle angle.
    '''
    u = np.zeros((2, 1))
    u[0] = 0 # no acceleration

    # Proportional-Derivative control of heading
    θ = x.flatten()[3]
    u[1] = a2 * rl.sawtooth(θ_desired - θ) + b2 * np.sin(θ - θ_previous) / dt
    print()
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


def runcar(duration, dt=.1):
    # initialize variables
    x = np.array([[0, 0, 2, 0]]).T  # x,y,v,θ
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    V_0 = .04  # desired height, or potential
    Mx, My, VX, VY, V = make_vehicle_field(xmin, xmax, ymin, ymax,
                                           desired_potential=V_0, threshold=.15)

    # run the animation using Euler's method
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
        θ_d = np.arctan2(r[1], r[0])
        u = control(x, θ_d, θ_previous, dt, a2=10, b2=100)
        x = x + dt * f(x, u)

        # Retain previous θ for PD control (consider including this in x)
        θ_previous = x.flatten()[3]

        # Draw vehicle and field
        rl.draw_tank(x[[0, 1, 3]], 'red', 0.2)  # x,y,θ
        draw_field(fig=fig, field=(Mx, My, VX, VY, V, V_0))


if __name__ == "__main__":
    runcar(10, dt=.1)
