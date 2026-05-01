# src/benchmark_viz/qmix_agent_benchmark/methods/update.py

import torch
import torch.nn.functional as F

def update(self, batch, gamma=0.99):
    obs_batch = torch.FloatTensor(batch["obs"])
    rewards_batch = torch.FloatTensor(batch["reward"])
    next_obs_batch = torch.FloatTensor(batch["next_obs"])
    dones_batch = torch.FloatTensor(batch["done"])

    total_q, _ = self.forward(obs_batch)

    with torch.no_grad():
        next_total_q, _ = self.forward(next_obs_batch)
        target = rewards_batch + gamma * (1 - dones_batch) * next_total_q

    loss = F.mse_loss(total_q.mean(), target.mean())

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()
