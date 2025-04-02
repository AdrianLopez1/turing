import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animate_2d_array(u, interval=100):
    """
    Animate a 2D array u[i,j] as a GIF where each frame is a plot of u[:,j] 
    for a particular timestep j.

    Parameters:
    u (numpy.ndarray): 2D array where rows are spatial points and columns are timesteps.
    interval (int): Time interval (milliseconds) between frames.
    """
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', lw=2)

    def init():
        ax.set_xlim(0, u.shape[0])  # Set x-axis based on spatial dimension
        ax.set_ylim(np.min(u), np.max(u))  # Set y-axis to cover all values in u
        line.set_data([], [])
        return line,

    def update(j):
        x = np.arange(u.shape[0])  # Spatial index
        y = u[:, j]  # Values at timestep j
        line.set_data(x, y)
        ax.set_title(f"Timestep {j}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=u.shape[1], init_func=init,
                                  blit=True, interval=interval, repeat=True)

    plt.close(fig)
    return HTML(ani.to_jshtml())