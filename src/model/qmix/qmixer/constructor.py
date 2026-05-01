# src/model/qmix/qmixer/constructor.py

import torch.nn as nn

def init(self, num_agents, state_dim, mixer_hidden_dim=64):
    nn.Module.__init__(self)
    self.num_agents = num_agents
    self.state_dim = state_dim
    self.mixer_hidden_dim = mixer_hidden_dim

    self.hyper_w1 = nn.Linear(state_dim, num_agents * mixer_hidden_dim)
    self.hyper_b1 = nn.Linear(state_dim, mixer_hidden_dim)
    self.hyper_w2 = nn.Linear(state_dim, mixer_hidden_dim)
    self.hyper_b2 = nn.Linear(state_dim, 1)
