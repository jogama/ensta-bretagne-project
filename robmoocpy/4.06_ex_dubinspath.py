from roblib import *

    
fig = figure(0)
ax = fig.add_subplot(111, aspect='equal')

r=10
a,b,ech = array([[-25,0,pi/2]]).T, array([[25,0,pi/2]]).T, 40      #simu 1
                    
cla()
ax.set_xlim(-ech,ech)
ax.set_ylim(-ech,ech)

draw_tank(a,"black")
draw_tank(b,"blue")

draw_arc(array([[0],[5]]),array([[4],[6]]),r,'red')

pause(1)
