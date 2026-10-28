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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Video

def animate_3d_function_small(u, extent=None, interval=100, cmap="gray", fps=10, filename="animation.mp4", n_skip=10):
    """
    Animate a 3D function u(x, y, t) using plt.imshow(), optimized for smaller file size.

    Parameters:
    u (numpy.ndarray): 3D array where u[:,:,t] represents the function at time t.
    extent (tuple): Optional (xmin, xmax, ymin, ymax) for axis labeling.
    interval (int): Time interval (milliseconds) between frames.
    cmap (str): Colormap for visualization (use "gray" for smaller file size).
    fps (int): Frames per second for the saved animation.
    filename (str): Output filename for MP4.
    """
    # Set color scale limits globally
    vmin, vmax = np.min(u), np.max(u)
    
    # Plot image
    fig, ax = plt.subplots()
    img = ax.imshow(u[:, :, 0], cmap=cmap, origin='lower', 
                    extent=extent if extent else None, animated=True,
                    vmin=vmin, vmax=vmax) 
    fig.colorbar(img, ax=ax)

    def update(t):
        img.set_array(u[:, :, t])
        ax.set_title(f"Timestep {t}")
        return img,

    ani = animation.FuncAnimation(fig, update, frames=range(0, u.shape[2], n_skip),  # Skip every nth frame
                                  blit=False, interval=interval, repeat=False)

    # Save animation as MP4 (without extra_args)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer)
    
    plt.close(fig)  # Prevents static image from showing

    return Video(filename)