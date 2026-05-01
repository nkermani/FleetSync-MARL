# src/benchmark_viz/random_agent/methods/get_action.py

import numpy as np

def get_action(self, obs=None, epsilon=0.0):
    return np.random.randint(self.n_actions)
