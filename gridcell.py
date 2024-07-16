import numpy as np
import matplotlib.pyplot as plt

def grid_cell_firing_pattern(position, initial_position, wavelength, direction):
    """
    Compute the grid cell firing pattern for a given position.
    
    Parameters:
    - position: Current position (x, y)
    - initial_position: Initial position (x0, y0)
    - wavelength: Spatial wavelength (lambda)
    - direction: Movement direction (w_dir)
    
    Returns:
    - Firing pattern value at the given position
    """
    x, y = position
    x0, y0 = initial_position
    k = [
        np.array([np.cos(direction), np.sin(direction)]),
        np.array([np.cos(direction + np.pi / 3), np.sin(direction + np.pi / 3)]),
        np.array([np.cos(direction + 2 * np.pi / 3), np.sin(direction + 2 * np.pi / 3)])
    ]
    
    firing_pattern =(2 / 3) * ((1 / 3) * sum(
        np.cos(4 * np.pi * np.dot(k_i, [x - x0, y - y0]) / wavelength * np.sqrt(3)) for k_i in k
    ) + 0.5)
    
    return firing_pattern

def plot_gridcell(env_size, wavelength=20, direction=0, initial_position=(0, 0)):

    # Create a grid of positions
    x_range = np.linspace(0, env_size, env_size * 10)
    y_range = np.linspace(0, env_size, env_size * 10)
    firing_patterns = np.zeros((len(x_range), len(y_range)))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            firing_patterns[i, j] = grid_cell_firing_pattern((x, y), initial_position, wavelength, direction)

    # Plot the firing pattern
    plt.figure(figsize=(8, 8))
    plt.contourf(x_range, y_range, firing_patterns.T, cmap='viridis')
    plt.colorbar(label='Firing Rate')
    plt.title('Grid Cell Firing Pattern')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()


# plot_gridcell(50, 20, 0, (0, 0))