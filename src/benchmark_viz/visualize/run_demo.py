# src/benchmark_viz/visualize/run_demo.py

from src.model.qmix import QMIXAgent as QMIXAgentModel
from src.train import ReplayBuffer
from src.env.environment import SimpleWarehouseEnv
import matplotlib.pyplot as plt

def run_demo(num_episodes=5):
    """Run a demo visualization."""
    env = SimpleWarehouseEnv(num_agents=5, grid_size=15, seed=42)
    model = QMIXAgentModel(num_agents=5, obs_dim=17, n_actions=5, hidden_dim=32, state_dim=10)

    print("=" * 50)
    print("FleetSync-MARL Visualization Demo")
    print("=" * 50)

    from .visualize_warehouse_layout import visualize_warehouse_layout
    obs = env.reset()
    fig, ax = visualize_warehouse_layout(env, title="Initial Warehouse Layout")
    plt.savefig("visualizations/01_warehouse_layout.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/01_warehouse_layout.png")
    plt.close()

    print("\nRunning episode visualization...")
    from .visualize_episode import visualize_episode
    fig = visualize_episode(env, model, max_steps=50, title_prefix="Demo Episode")
    plt.savefig("visualizations/02_episode_visualization.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/02_episode_visualization.png")
    plt.close()

    print("\nVisualizing robot observations...")
    from .visualize_robot_observations import visualize_robot_observations
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
            actions = model.get_actions(obs['agents'], env.env._get_global_state(), epsilon=0.5)
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

    from .visualize_training_progress import visualize_training_progress
    fig = visualize_training_progress(buffer_sizes, losses, rewards)
    plt.savefig("visualizations/04_training_progress.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/04_training_progress.png")
    plt.close()

    print("\n" + "=" * 50)
    print("Visualization complete!")
    print("=" * 50)
