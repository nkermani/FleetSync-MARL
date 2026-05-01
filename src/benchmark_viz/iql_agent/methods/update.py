# src/benchmark_viz/iql_agent/methods/update.py

import torch
import torch.nn.functional as F

def update(self, obs, action, reward, next_obs, done, gamma=0.99):
    q_val = self.q_net(torch.FloatTensor(obs).unsqueeze(0)).gather(1, torch.LongTensor([action]).unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q = self.q_net(torch.FloatTensor(next_obs).unsqueeze(0)).max(1)[0]
        target = reward + gamma * (1 - done) * next_q
    loss = F.mse_loss(q_val, target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()
