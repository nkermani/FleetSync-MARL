# src/model/qmix/qmix_agent/methods/get_actions.py

import torch

def get_actions(self, obs, state, epsilon=0.1):
    obs_tensor = torch.FloatTensor(obs).to(self.device)
    state_tensor = torch.FloatTensor(state).to(self.device) if state is not None else None
    return self.model.get_actions(obs_tensor, state_tensor, epsilon)
