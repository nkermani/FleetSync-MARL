# src/benchmark_viz/visualize/visualize_robot_observations.py

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_robot_observations(env, robot_idx=0):
    """Visualize what a single robot observes."""
    obs = env.reset()
    agent_obs = obs['agents'][robot_idx]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    feature_names = [
        'Position X', 'Position Y', 'Has Load',
        'Task Target X', 'Task Target Y',
        'Nearby Robots', 'Nearby Obstacles',
        'Nearest Task X', 'Nearest Task Y', 'Task Distance',
        'Reserved 1', 'Reserved 2', 'Reserved 3',
        'Zone Direction X', 'Zone Direction Y',
        'Reserved 4', 'Reserved 5'
    ]

    values = agent_obs[:17]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 17))
    axes[0, 0].barh(range(17), values, color=colors)
    axes[0, 0].set_yticks(range(17))
    axes[0, 0].set_yticklabels(feature_names)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_title(f"Robot {robot_idx} Observation")
    axes[0, 0].set_xlabel("Normalized Value")

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
    axes[0, 1].imshow(grid_display.T, cmap=cmap, origin='lower', extent=[0, size, 0, size])
    rx, ry = env.env.robots[robot_idx].get_position_int()
    axes[0, 1].scatter(rx, ry, s=200, c='blue', marker='o', edgecolors='black', linewidths=2, zorder=5)
    axes[0, 1].set_title("Robot Position")

    axes[1, 0].hist(agent_obs, bins=20, edgecolor='black')
    axes[1, 0].set_title("Observation Distribution")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].pie(
        [abs(v) for v in values[:5]],
        labels=feature_names[:5],
        autopct='%1.1f%%'
    )
    axes[1, 1].set_title("Feature Importance (Top 5)")

    plt.tight_layout()
    return fig
