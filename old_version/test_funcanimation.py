import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Create random data
t = np.linspace(0,100,1000)
y = np.sin(t)

# Initialise figure and ax
fig, ax = plt.subplots()
# Setup axis
ax.set(xlim=[0, 100], ylim=[-1, 1], xlabel='Time [s]', ylabel='y')

# Attach an artist to the ax - this is stored in the 'line' variable
line = ax.scatter(t[0], y[0])
print(line, type(line))


ax.set(xlim=[0, 100], ylim=[-1, 1], xlabel='Time [s]', ylabel='y')


# Create animation function that runs through data
def draw_sin(timestep):
    # Access data
    global t
    global y
    global ax
    # Produce new data 
    a = t[timestep]
    b = y[timestep]
    data = np.stack([a,b]).T
    # Set attributes of an existing 'artist'
    line.set_offsets(data)
    # return (line,)

ani = FuncAnimation(fig=fig, func=draw_sin, frames=len(t), interval=30)
plt.show()
