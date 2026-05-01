# src/train/trainer/methods/_run_episode.py

def _run_episode(self, env, epsilon, buffer, batch_size):
    obs, info = env.reset()
    state = env.env._get_global_state()
    done = False
    episode_reward = 0

    while not done:
        actions = self.agent.get_actions(obs, state, epsilon)
        next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        buffer.push(obs, state, actions, reward, next_obs, next_state, done)

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            self.agent.update(batch)
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.agent.update_target()

        obs = next_obs
        state = next_state
        episode_reward += reward

    return episode_reward
