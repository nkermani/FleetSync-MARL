# src/train/trainer/methods/evaluate.py

import numpy as np
import torch

def evaluate(self, env, num_episodes=10, epsilon=0.0):
    self.agent.model.eval()
    total_rewards = []
    tasks_completed = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        state = env.env._get_global_state()
        done = False
        episode_reward = 0
        tasks = 0

        while not done:
            actions = self.agent.get_actions(obs, state, epsilon)
            next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            if "tasks_completed" in info:
                tasks += info["tasks_completed"]
            episode_reward += reward
            obs = next_obs
            state = next_state

        total_rewards.append(episode_reward)
        tasks_completed.append(tasks)

    self.agent.model.train()
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_tasks": np.mean(tasks_completed),
    }
