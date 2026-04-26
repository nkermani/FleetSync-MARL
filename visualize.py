import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from src.env.environment import SimpleWarehouseEnv, WarehouseEnv
from src.model.qmix import SimpleQMIX
from src.train import ReplayBuffer


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


def visualize_training_progress(buffer_sizes, losses, rewards):
    """Visualize training progress over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(buffer_sizes, linewidth=2)
    axes[0].set_title("Buffer Size")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Samples")

    axes[1].plot(losses, linewidth=2, color='orange')
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("MSE Loss")

    axes[2].plot(rewards, linewidth=2, color='green')
    axes[2].set_title("Episode Reward")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Total Reward")

    plt.tight_layout()
    return fig


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


def run_demo(num_episodes=5):
    """Run a demo visualization."""
    env = SimpleWarehouseEnv(num_agents=5, grid_size=15, seed=42)
    model = SimpleQMIX(num_agents=5, obs_dim=17, n_actions=5, hidden_dim=32)

    print("=" * 50)
    print("FleetSync-MARL Visualization Demo")
    print("=" * 50)

    obs = env.reset()
    fig, ax = visualize_warehouse_layout(env, title="Initial Warehouse Layout")
    plt.savefig("visualizations/01_warehouse_layout.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/01_warehouse_layout.png")
    plt.close()

    print("\nRunning episode visualization...")
    fig = visualize_episode(env, model, max_steps=50, title_prefix="Demo Episode")
    plt.savefig("visualizations/02_episode_visualization.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/02_episode_visualization.png")
    plt.close()

    print("\nVisualizing robot observations...")
    fig = visualize_robot_observations(env, robot_idx=0)
    plt.savefig("visualizations/03_robot_observations.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/03_robot_observations.png")
    plt.close()

    buffer_sizes = []
    losses = []
    rewards = []

    print("\nRunning training visualization...")
    for ep in range(num_episodes):
        buffer = ReplayBuffer(capacity=1000, num_agents=5, obs_dim=17, state_dim=10)
        obs = env.reset()
        ep_reward = 0

        for step in range(50):
            actions = model.get_actions(obs['agents'], None, epsilon=0.5)
            result = env.step(actions)
            next_obs = result[0]
            reward = result[2]
            done = result[3] or result[4]
            buffer.push(obs['agents'], env.env._get_global_state(), actions, reward, next_obs['agents'], env.env._get_global_state(), done)
            ep_reward += reward
            obs = next_obs
            if done:
                obs = env.reset()

        if len(buffer) >= 32:
            batch = buffer.sample(32)
            loss = model.update(batch)
            losses.append(loss)

        buffer_sizes.append(len(buffer))
        rewards.append(ep_reward)
        print(f"  Episode {ep}: Buffer={len(buffer)}, Loss={loss:.4f}, Reward={ep_reward:.1f}")

    fig = visualize_training_progress(buffer_sizes, losses, rewards)
    plt.savefig("visualizations/04_training_progress.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/04_training_progress.png")
    plt.close()

    print("\n" + "=" * 50)
    print("Visualization complete!")
    print("=" * 50)


if __name__ == "__main__":
    import os
    os.makedirs("visualizations", exist_ok=True)
    run_demo(num_episodes=20)