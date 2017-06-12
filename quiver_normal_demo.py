'''
========================================================
Demonstration of advanced quiver and quiverkey functions
========================================================

Known problem: the plot autoscaling does not take into account
the arrows, so those on the boundaries are often out of the picture.
This is *not* an easy problem to solve in a perfectly general way.
The workaround is to manually expand the axes.
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal as mnorm
import numpy as np

step = .2
X, Y = np.meshgrid(np.arange(-np.pi, np.pi, step), np.arange(-np.pi, np.pi, step))
island_center = [0, 0]
island = mnorm(island_center, cov=1).pdf(np.dstack((X,Y)))

U = np.gradient(island)[1]
V = np.gradient(island)[0]

plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, island)

plt.show()
