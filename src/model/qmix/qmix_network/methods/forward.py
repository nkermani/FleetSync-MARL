# src/model/qmix/qmix_network/methods/forward.py

import numpy as np
import torch

def forward(self, obs, state=None):
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs)

    if obs.dim() == 3:
        if obs.size(0) == self.num_agents:
            obs = obs.transpose(0, 1)
        elif obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)
    elif obs.dim() == 2 and obs.size(0) == self.num_agents:
        obs = obs.transpose(0, 1).unsqueeze(0)
    elif obs.dim() == 1:
        obs = obs.unsqueeze(0).unsqueeze(0)

    if obs.size(-1) != self.obs_dim:
        obs = obs.view(-1, self.num_agents, self.obs_dim)

    q_vals = []
    for i, actor in enumerate(self.actors):
        agent_obs = obs[:, i]
        q = actor(agent_obs)
        q_vals.append(q)
    q_vals = torch.stack(q_vals, dim=1)

    return q_vals
