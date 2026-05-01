# src/benchmark_viz/iql_agent/methods/get_action.py

import numpy as np
import torch

def get_action(self, obs, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(self.n_actions)
    with torch.no_grad():
        q_vals = self.q_net(torch.FloatTensor(obs))
        return q_vals.argmax().item()
