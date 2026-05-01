# src/benchmark_viz/visualize/visualize_episode.py

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_episode(env, model, max_steps=100, title_prefix="Episode"):
    """Visualize a full episode with robot movement."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    obs = env.reset()

    positions_history = [[] for _ in range(env.num_agents)]
    rewards_history = []

    for step in range(max_steps):
        actions = model.get_actions(obs['agents'], None, epsilon=0.1)
        result = env.step(actions)

        next_obs = result[0]
        reward = result[2]
        done = result[3] or result[4]
        info = result[5]

        for i, robot in enumerate(env.env.robots):
            positions_history[i].append(robot.get_position_int())

        rewards_history.append(reward)

        obs = next_obs
        if done:
            break

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

    axes[0].imshow(grid_display.T, cmap=cmap, origin='lower', extent=[0, size, 0, size])

    colors = plt.cm.tab10(np.linspace(0, 1, env.num_agents))
    for i in range(env.num_agents):
        if positions_history[i]:
            xs = [p[0] for p in positions_history[i]]
            ys = [p[1] for p in positions_history[i]]
            axes[0].plot(xs, ys, c=colors[i], linewidth=2, alpha=0.7)
            axes[0].scatter(xs[0], ys[0], c=[colors[i]], marker='o', s=100, edgecolors='black')
            axes[0].scatter(xs[-1], ys[-1], c=[colors[i]], marker='s', s=100, edgecolors='black')

    axes[0].set_title(f"{title_prefix} - Robot Trajectories")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].bar(range(env.num_agents), [len(set(p)) for p in positions_history])
    axes[1].set_title("Unique Positions Visited")
    axes[1].set_xlabel("Robot ID")
    axes[1].set_ylabel("Positions")

    axes[2].plot(rewards_history, linewidth=2)
    axes[2].fill_between(range(len(rewards_history)), rewards_history, alpha=0.3)
    axes[2].set_title("Episode Rewards")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Reward")

    plt.tight_layout()
    return fig
