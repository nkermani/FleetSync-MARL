# src/benchmark_viz/vdn/constructor.py

import torch.nn as nn
import torch.nn.functional as F

def init(self, num_agents, obs_dim, n_actions, hidden_dim=64):
    nn.Module.__init__(self)
    self.num_agents = num_agents
    self.obs_dim = obs_dim
    self.n_actions = n_actions

    self.actors = nn.ModuleList([
        nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        for _ in range(num_agents)
    ])

    self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
