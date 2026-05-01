# src/model/qmix/qmix_network/methods/get_actions.py

import numpy as np
import torch

def get_actions(self, obs, state, epsilon=0.1):
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs)
    if isinstance(state, np.ndarray):
        state = torch.FloatTensor(state)

    if obs.dim() == 2 and obs.size(0) == self.num_agents:
        obs = obs.unsqueeze(0)
    elif obs.dim() == 1:
        obs = obs.unsqueeze(0)

    if obs.size(-1) != self.obs_dim:
        obs = obs.view(-1, self.num_agents, self.obs_dim)

    actions = []
    for i, actor in enumerate(self.actors):
        agent_obs = obs[:, i]
        with torch.no_grad():
            q = actor(agent_obs)
        if np.random.rand() < epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = q.argmax(dim=-1).item()
        actions.append(a)
    return actions
