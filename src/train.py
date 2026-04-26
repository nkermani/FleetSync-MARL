import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, state_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        self.obs_buffer = [deque(maxlen=capacity) for _ in range(num_agents)]
        self.state_buffer = deque(maxlen=capacity)
        self.actions_buffer = deque(maxlen=capacity)
        self.rewards_buffer = deque(maxlen=capacity)
        self.next_obs_buffer = [deque(maxlen=capacity) for _ in range(num_agents)]
        self.next_state_buffer = deque(maxlen=capacity)
        self.done_buffer = deque(maxlen=capacity)

    def push(self, obs, state, actions, reward, next_obs, next_state, done):
        for i in range(self.num_agents):
            self.obs_buffer[i].append(obs[i])
            self.next_obs_buffer[i].append(next_obs[i])

        self.state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        self.actions_buffer.append(actions)
        self.rewards_buffer.append(reward)
        self.done_buffer.append(done)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)

        obs_batch = []
        for i in range(self.num_agents):
            agent_obs = [self.obs_buffer[i][idx] for idx in indices]
            obs_batch.append(np.array(agent_obs))

        next_obs_batch = []
        for i in range(self.num_agents):
            agent_next_obs = [self.next_obs_buffer[i][idx] for idx in indices]
            next_obs_batch.append(np.array(agent_next_obs))

        return {
            "obs": np.array(obs_batch),
            "state": np.array([self.state_buffer[idx] for idx in indices]),
            "actions": np.array([self.actions_buffer[idx] for idx in indices]),
            "reward": np.array([self.rewards_buffer[idx] for idx in indices]),
            "next_obs": np.array(next_obs_batch),
            "next_state": np.array([self.next_state_buffer[idx] for idx in indices]),
            "done": np.array([self.done_buffer[idx] for idx in indices]),
        }

    def __len__(self):
        return len(self.state_buffer)


class Trainer:
    def __init__(
        self,
        model,
        num_agents,
        obs_dim,
        state_dim,
        learning_rate=1e-4,
        gamma=0.99,
        target_update_freq=200,
    ):
        self.model = model
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)

        self.update_count = 0
        self.training_history = []

    def update(self, batch):
        obs_batch = torch.FloatTensor(batch["obs"])
        state_batch = torch.FloatTensor(batch["state"])
        actions_batch = torch.LongTensor(batch["actions"])
        rewards_batch = torch.FloatTensor(batch["reward"])
        next_obs_batch = torch.FloatTensor(batch["next_obs"])
        next_state_batch = torch.FloatTensor(batch["next_state"])
        dones_batch = torch.FloatTensor(batch["done"])

        q_tot, individual_qs = self.model.model(obs_batch, state_batch)

        action_idx = actions_batch.long()

        chosen_q_vals = []
        for i in range(self.num_agents):
            agent_q_vals = individual_qs[:, i, :]
            agent_actions = action_idx[:, i]
            chosen_q_vals.append(agent_q_vals.gather(1, agent_actions.unsqueeze(1)))

        chosen_q_vals = torch.cat(chosen_q_vals, dim=1)
        mixed_q = self.model.model.mixer(chosen_q_vals, state_batch)

        with torch.no_grad():
            next_q_tot, _ = self.model.target_model(next_obs_batch, next_state_batch)
            target_q = rewards_batch + self.gamma * (1 - dones_batch) * next_q_tot

        loss = F.mse_loss(mixed_q.squeeze(-1), target_q.squeeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 10)
        self.optimizer.step()

        self.update_count += 1

        if self.update_count % self.target_update_freq == 0:
            self.model.update_target()

        return loss.item()

    def train(
        self,
        env,
        num_episodes,
        buffer,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        batch_size=32,
        log_interval=100,
    ):
        self.model.model.train()

        for episode in range(num_episodes):
            obs, info = env.reset()
            state = env.env._get_global_state()
            done = False
            episode_reward = 0

            while not done:
                actions = self.model.get_actions(
                    torch.FloatTensor(obs),
                    torch.FloatTensor(state),
                    epsilon
                )

                (
                    next_obs,
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(actions)

                done = terminated or truncated

                buffer.push(
                    obs,
                    state,
                    actions,
                    reward,
                    next_obs,
                    next_state,
                    done
                )

                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    loss = self.update(batch)

                obs = next_obs
                state = next_state
                episode_reward += reward

            self.training_history.append(episode_reward)

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if episode % log_interval == 0:
                avg_reward = np.mean(self.training_history[-log_interval:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {epsilon:.3f}")

        return self.training_history

    def evaluate(self, env, num_episodes=10, epsilon=0.0):
        self.model.model.eval()

        total_rewards = []
        tasks_completed = []

        for _ in range(num_episodes):
            obs, info = env.reset()
            state = env.env._get_global_state()
            done = False
            episode_reward = 0
            tasks = 0

            while not done:
                actions = self.model.get_actions(
                    torch.FloatTensor(obs),
                    torch.FloatTensor(state),
                    epsilon
                )

                (
                    next_obs,
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(actions)

                done = terminated or truncated

                if "tasks_completed" in info:
                    tasks += info["tasks_completed"]

                episode_reward += reward
                obs = next_obs
                state = next_state

            total_rewards.append(episode_reward)
            tasks_completed.append(tasks)

        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_tasks": np.mean(tasks_completed),
        }