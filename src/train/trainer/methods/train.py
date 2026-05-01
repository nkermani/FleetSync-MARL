# src/train/trainer/methods/train.py

import numpy as np

def train(self, env, num_episodes, buffer, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, batch_size=32, log_interval=100):
    for episode in range(num_episodes):
        episode_reward = self._run_episode(env, epsilon, buffer, batch_size)
        self.training_history.append(episode_reward)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % log_interval == 0:
            avg_reward = np.mean(self.training_history[-log_interval:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {epsilon:.3f}")

    return self.training_history
