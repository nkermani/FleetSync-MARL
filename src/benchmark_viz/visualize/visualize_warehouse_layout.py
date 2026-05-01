# src/benchmark_viz/visualize/visualize_warehouse_layout.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def visualize_warehouse_layout(env, title="Warehouse Layout"):
    """Visualize the warehouse with robots, obstacles, and task zones."""
    fig, ax = plt.subplots(figsize=(10, 10))

    grid = env.env.grid
    size = grid.size

    grid_display = np.zeros((size, size))

    for x, y in grid.obstacles:
        grid_display[x, y] = 1
    for x, y in grid.pickup_zones:
        grid_display[x, y] = 2
    for x, y in grid.delivery_zones:
        grid_display[x, y] = 3

    cmap = ListedColormap(['white', 'gray', 'green', 'red'])
    ax.imshow(grid_display.T, cmap=cmap, origin='lower', extent=[0, size, 0, size])

    for i, robot in enumerate(env.env.robots):
        x, y = robot.get_position_int()
        color = plt.cm.tab10(i % 10)
        ax.scatter(x, y, s=200, c=[color], marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'R{i}', (x + 0.3, y + 0.3), fontsize=8, fontweight='bold')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    legend_elements = [
        mpatches.Patch(color='gray', label='Obstacle'),
        mpatches.Patch(color='green', label='Pickup Zone'),
        mpatches.Patch(color='red', label='Delivery Zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return fig, ax
