# src/model/qmix/qmix_agent/constructor.py

import torch
from ..qmix_network import QMIXNetwork

def init(self, num_agents, obs_dim, n_actions, state_dim, hidden_dim=64, mixer_hidden_dim=64, device="cpu"):
    self.num_agents = num_agents
    self.obs_dim = obs_dim
    self.n_actions = n_actions
    self.state_dim = state_dim
    self.device = device

    self.model = QMIXNetwork(
        num_agents=num_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        mixer_hidden_dim=mixer_hidden_dim
    ).to(device)

    self.target_model = QMIXNetwork(
        num_agents=num_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        mixer_hidden_dim=mixer_hidden_dim
    ).to(device)

    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
