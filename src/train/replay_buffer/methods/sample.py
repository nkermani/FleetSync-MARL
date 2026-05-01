# src/train/replay_buffer/methods/sample.py

import numpy as np

def sample(self, batch_size):
    indices = np.random.choice(len(self), batch_size, replace=False)

    obs_batch = []
    for i in range(self.num_agents):
        agent_obs = [self.obs_buffer[i][idx] for idx in indices]
        obs_batch.append(np.array(agent_obs))

    next_obs_batch = []
    for i in range(self.num_agents):
        agent_next_obs = [self.next_obs_buffer[i][idx] for idx in indices]
        next_obs_batch.append(np.array(agent_next_obs))

    return {
        "obs": np.array(obs_batch),
        "state": np.array([self.state_buffer[idx] for idx in indices]),
        "actions": np.array([self.actions_buffer[idx] for idx in indices]),
        "reward": np.array([self.rewards_buffer[idx] for idx in indices]),
        "next_obs": np.array(next_obs_batch),
        "next_state": np.array([self.next_state_buffer[idx] for idx in indices]),
        "done": np.array([self.done_buffer[idx] for idx in indices]),
    }
