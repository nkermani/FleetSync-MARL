# src/model/qmix/robot_actor/constructor.py

import torch.nn as nn

def init(self, obs_dim, n_actions, hidden_dim=64):
    nn.Module.__init__(self)
    self.obs_dim = obs_dim
    self.n_actions = n_actions
    self.hidden_dim = hidden_dim
    self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_actions),
    )
