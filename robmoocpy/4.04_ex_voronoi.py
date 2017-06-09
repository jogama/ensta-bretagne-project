from roblib import *

fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-5,15)
ax.set_ylim(-5,15)
m=20
p=10*rand(2,m)
plot(p[0,:],p[1,:],'ob')
pause(1)

