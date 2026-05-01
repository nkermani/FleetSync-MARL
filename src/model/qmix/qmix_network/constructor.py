# src/model/qmix/qmix_network/constructor.py

import torch.nn as nn
from ..robot_actor import RobotActor
from ..qmixer import QMixer

def init(self, num_agents, obs_dim, n_actions, state_dim, hidden_dim=64, mixer_hidden_dim=64):
    nn.Module.__init__(self)
    self.num_agents = num_agents
    self.obs_dim = obs_dim
    self.n_actions = n_actions
    self.hidden_dim = hidden_dim
    self.state_dim = state_dim

    self.actors = nn.ModuleList([
        RobotActor(obs_dim, n_actions, hidden_dim) for _ in range(num_agents)
    ])
    self.mixer = QMixer(num_agents, state_dim, mixer_hidden_dim)
