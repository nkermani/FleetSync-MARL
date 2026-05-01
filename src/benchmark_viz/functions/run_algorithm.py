# src/benchmark_viz/functions/run_algorithm.py

import numpy as np
from src.benchmark_viz.random_agent import RandomAgent
from src.benchmark_viz.iql_agent import IQLAgent

def run_algorithm(env_class, algo_class, num_episodes=150, num_agents=5, obs_dim=17, n_actions=5,
                 hidden_dim=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99,
                 name="Algorithm"):
    print(f"\n{'='*40}")
    print(f"Running {name}")
    print(f"{'='*40}")

    if algo_class == RandomAgent:
        agents = [RandomAgent(n_actions) for _ in range(num_agents)]
    elif algo_class == IQLAgent:
        agents = [IQLAgent(obs_dim, n_actions, hidden_dim) for _ in range(num_agents)]
    else:
        agents = algo_class(num_agents, obs_dim, n_actions, hidden_dim)

    rewards_history = []
    tasks_history = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        env = env_class(num_agents=num_agents, grid_size=10, seed=episode)
        obs = env.reset()

        episode_reward = 0
        episode_tasks = 0
        done = False
        step = 0

        while not done and step < 50:
            if algo_class in [IQLAgent, RandomAgent]:
                actions = [agent.get_action(obs['agents'][i], epsilon) for i, agent in enumerate(agents)]
            else:
                actions = agents.get_actions(obs['agents'], epsilon)

            result = env.step(actions)
            next_obs = result[0]
            reward = result[2]
            done = result[3] or result[4]
            info = result[5]

            episode_tasks += info.get("tasks_completed", 0)
            episode_reward += reward

            if algo_class == IQLAgent:
                for i, agent in enumerate(agents):
                    agent.update(obs['agents'][i], actions[i], reward, next_obs['agents'][i], done)
            elif algo_class == RandomAgent:
                pass
            else:
                buffer = {"obs": obs['agents'], "next_obs": next_obs['agents'], "reward": np.array([reward]), "done": np.array([done])}
                agents.update(buffer)

            obs = next_obs
            step += 1

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        rewards_history.append(episode_reward)
        tasks_history.append(episode_tasks)

        if episode % 30 == 0:
            print(f"  Ep {episode:3d}: Reward={episode_reward:6.1f}, Tasks={episode_tasks}, Eps={epsilon:.3f}")

    return {"name": name, "rewards": rewards_history, "tasks": tasks_history}
