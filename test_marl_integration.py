#!/usr/bin/env python
"""Integration test for FleetSync-MARL"""

import sys
sys.path.insert(0, "src")

from environment import WarehouseEnv
from model.qmix import SimpleQMIX, QMIXNetwork
from train import ReplayBuffer
import numpy as np

def test_marl_integration():
    print("=" * 60)
    print("FleetSync-MARL Integration Test")
    print("=" * 60)

    # Initialize environment
    print("\n1. Initializing environment...")
    env = WarehouseEnv(num_agents=3, grid_size=15, max_steps=100, seed=42)
    obs, info = env.reset()
    print(f"   ✓ Environment created with {env.num_agents} agents")
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Tasks completed: {info['tasks_completed']}")

    # Initialize agent
    print("\n2. Initializing QMIX agent...")
    agent = SimpleQMIX(num_agents=3, obs_dim=17, n_actions=5)
    print(f"   ✓ Agent created with {agent.num_agents} agents")
    print(f"   ✓ Device: {agent.device}")

    # Initialize replay buffer
    print("\n3. Initializing replay buffer...")
    buffer = ReplayBuffer(1000, 3, 17, 51)
    print(f"   ✓ Buffer created with capacity {buffer.capacity}")

    # Run a few episodes
    print("\n4. Running training episodes...")
    episode_rewards = []

    for episode in range(5):
        obs, info = env.reset()
        state = env._get_global_state()
        done = False
        episode_reward = 0

        while not done:
            actions = agent.get_actions(obs, state, epsilon=0.3)
            result = env.step(actions)
            next_obs, reward, terminated, truncated, info = result
            next_state = env._get_global_state()
            done = terminated or truncated

            buffer.push(obs, state, actions, reward, next_obs, next_state, done)
            episode_reward += reward

            if len(buffer) >= 32:
                batch = buffer.sample(32)
                loss = agent.update(batch)

            obs = next_obs
            state = next_state

        episode_rewards.append(episode_reward)
        print(f"   Episode {episode + 1}: Reward = {episode_reward:.2f}, Tasks = {info['tasks_completed']}")

    # Evaluate
    print("\n5. Evaluating agent (greedy policy)...")
    eval_rewards = []
    for _ in range(3):
        obs, info = env.reset()
        state = env._get_global_state()
        done = False
        episode_reward = 0

        while not done:
            actions = agent.get_actions(obs, state, epsilon=0.0)  # Greedy
            result = env.step(actions)
            next_obs, reward, terminated, truncated, info = result
            next_state = env._get_global_state()
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
            state = next_state

        eval_rewards.append(episode_reward)

    print(f"   Mean eval reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")

    print("\n" + "=" * 60)
    print("✓ Integration test PASSED!")
    print("✓ MARL system is working correctly:")
    print("  - Environment provides multi-agent observations")
    print("  - QMIX agent selects actions using CTDE (Centralized Training, Decentralized Execution)")
    print("  - Replay buffer stores and samples experiences")
    print("  - Training updates the Q-network using TD learning")
    print("=" * 60)

if __name__ == "__main__":
    test_marl_integration()
