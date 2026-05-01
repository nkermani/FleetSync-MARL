# src/benchmark_viz/vdn/methods/forward.py

import numpy as np
import torch

def forward(self, obs):
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs)

    if obs.dim() == 2 and obs.size(0) == self.num_agents:
        obs = obs.unsqueeze(0)

    if obs.size(-1) != self.obs_dim:
        obs = obs.view(-1, self.num_agents, self.obs_dim)

    q_vals = []
    for i, actor in enumerate(self.actors):
        agent_obs = obs[:, i]
        q = actor(agent_obs)
        q_vals.append(q)

    q_vals = torch.stack(q_vals, dim=1)
    total_q = q_vals.sum()

    return total_q.unsqueeze(0), q_vals
