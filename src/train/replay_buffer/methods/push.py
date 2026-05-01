# src/train/replay_buffer/methods/push.py

def push(self, obs, state, actions, reward, next_obs, next_state, done):
    for i in range(self.num_agents):
        self.obs_buffer[i].append(obs[i])
        self.next_obs_buffer[i].append(next_obs[i])

    self.state_buffer.append(state)
    self.next_state_buffer.append(next_state)
    self.actions_buffer.append(actions)
    self.rewards_buffer.append(reward)
    self.done_buffer.append(done)
