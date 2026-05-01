# src/benchmark_viz/iql_agent/constructor.py

import torch.nn as nn
import torch.nn.functional as F

def init(self, obs_dim, n_actions, hidden_dim=64):
    nn.Module.__init__(self)
    self.obs_dim = obs_dim
    self.n_actions = n_actions
    self.q_net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_actions),
    )
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
