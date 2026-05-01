# src/train/replay_buffer/constructor.py

from collections import deque

def init(self, capacity, num_agents, obs_dim, state_dim):
    self.capacity = capacity
    self.num_agents = num_agents
    self.obs_dim = obs_dim
    self.state_dim = state_dim

    self.obs_buffer = [deque(maxlen=capacity) for _ in range(num_agents)]
    self.state_buffer = deque(maxlen=capacity)
    self.actions_buffer = deque(maxlen=capacity)
    self.rewards_buffer = deque(maxlen=capacity)
    self.next_obs_buffer = [deque(maxlen=capacity) for _ in range(num_agents)]
    self.next_state_buffer = deque(maxlen=capacity)
    self.done_buffer = deque(maxlen=capacity)
