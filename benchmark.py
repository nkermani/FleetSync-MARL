import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
np.random.seed(42)
torch.manual_seed(42)


class RandomAgent:
    """Random baseline."""
    def __init__(self, n_actions):
        self.n_actions = n_actions
    
    def get_action(self, obs=None, epsilon=0.0):
        return np.random.randint(self.n_actions)


class IQLAgent(nn.Module):
    """Independent Q-Learning - each agent learns separately."""
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
    
    def get_action(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(obs))
            return q_vals.argmax().item()
    
    def update(self, obs, action, reward, next_obs, done, gamma=0.99):
        q_val = self.q_net(torch.FloatTensor(obs).unsqueeze(0)).gather(1, torch.LongTensor([action]).unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.q_net(torch.FloatTensor(next_obs).unsqueeze(0)).max(1)[0]
            target = reward + gamma * (1 - done) * next_q
        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class VDN(nn.Module):
    """Value Decomposition Network."""
    def __init__(self, num_agents, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )
            for _ in range(num_agents)
        ])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        if obs.dim() == 2 and obs.size(0) == self.num_agents:
            obs = obs.unsqueeze(0)
        
        if obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)
        
        q_vals = []
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i]
            q = actor(agent_obs)
            q_vals.append(q)
        
        q_vals = torch.stack(q_vals, dim=1)
        total_q = q_vals.sum()
        
        return total_q.unsqueeze(0), q_vals
    
    def get_actions(self, obs, epsilon=0.1):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        if obs.dim() == 2 and obs.size(0) == self.num_agents:
            obs = obs.unsqueeze(0)
        
        if obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)
        
        actions = []
        for i, actor in enumerate(self.actors):
            with torch.no_grad():
                q = actor(obs[:, i])
            if np.random.rand() < epsilon:
                a = np.random.randint(self.n_actions)
            else:
                a = q.argmax(dim=-1).item()
            actions.append(a)
        
        return actions
    
    def update(self, batch, gamma=0.99):
        obs_batch = torch.FloatTensor(batch["obs"])
        rewards_batch = torch.FloatTensor(batch["reward"])
        next_obs_batch = torch.FloatTensor(batch["next_obs"])
        dones_batch = torch.FloatTensor(batch["done"])
        
        total_q, _ = self.forward(obs_batch)
        
        with torch.no_grad():
            next_total_q, _ = self.forward(next_obs_batch)
            target = rewards_batch + gamma * (1 - dones_batch) * next_total_q
        
        loss = F.mse_loss(total_q.mean(), target.mean())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class QMIXAgent(nn.Module):
    """QMIX - value decomposition with state conditioning."""
    def __init__(self, num_agents, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )
            for _ in range(num_agents)
        ])
        
        self.state_encoder = nn.Linear(num_agents * obs_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        if obs.dim() == 2 and obs.size(0) == self.num_agents:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        if obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)
        
        q_vals = []
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i]
            q = actor(agent_obs)
            q_vals.append(q)
        
        q_vals = torch.stack(q_vals, dim=1)
        total_q = q_vals.mean()
        
        return total_q.unsqueeze(0), q_vals
    
    def get_actions(self, obs, epsilon=0.1):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        if obs.dim() == 2 and obs.size(0) == self.num_agents:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        if obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)
        
        actions = []
        for i, actor in enumerate(self.actors):
            with torch.no_grad():
                q = actor(obs[:, i])
            if np.random.rand() < epsilon:
                a = np.random.randint(self.n_actions)
            else:
                a = q.argmax(dim=-1).item()
            actions.append(a)
        
        return actions
    
    def update(self, batch, gamma=0.99):
        obs_batch = torch.FloatTensor(batch["obs"])
        rewards_batch = torch.FloatTensor(batch["reward"])
        next_obs_batch = torch.FloatTensor(batch["next_obs"])
        dones_batch = torch.FloatTensor(batch["done"])
        
        total_q, _ = self.forward(obs_batch)
        
        with torch.no_grad():
            next_total_q, _ = self.forward(next_obs_batch)
            target = rewards_batch + gamma * (1 - dones_batch) * next_total_q
        
        loss = F.mse_loss(total_q.mean(), target.mean())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def run_algorithm(env_class, algo_class, num_episodes=150, num_agents=5, obs_dim=17, n_actions=5, 
                hidden_dim=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99,
                name="Algorithm"):
    print(f"\n{'='*40}")
    print(f"Running {name}")
    print(f"{'='*40}")
    
    if algo_class == RandomAgent:
        agents = [RandomAgent(n_actions) for _ in range(num_agents)]
    elif algo_class == IQLAgent:
        agents = [IQLAgent(obs_dim, n_actions, hidden_dim) for _ in range(num_agents)]
    else:
        agents = algo_class(num_agents, obs_dim, n_actions, hidden_dim)
    
    rewards_history = []
    tasks_history = []
    
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        env = env_class(num_agents=num_agents, grid_size=10, seed=episode)
        obs = env.reset()
        
        episode_reward = 0
        episode_tasks = 0
        done = False
        step = 0
        
        while not done and step < 50:
            if algo_class in [IQLAgent, RandomAgent]:
                actions = [agent.get_action(obs['agents'][i], epsilon) for i, agent in enumerate(agents)]
            else:
                actions = agents.get_actions(obs['agents'], epsilon)
            
            result = env.step(actions)
            next_obs = result[0]
            reward = result[2]
            done = result[3] or result[4]
            info = result[5]
            
            episode_tasks += info.get("tasks_completed", 0)
            episode_reward += reward
            
            if algo_class == IQLAgent:
                for i, agent in enumerate(agents):
                    agent.update(obs['agents'][i], actions[i], reward, next_obs['agents'][i], done)
            elif algo_class == RandomAgent:
                pass
            else:
                buffer = {"obs": obs['agents'], "next_obs": next_obs['agents'], "reward": np.array([reward]), "done": np.array([done])}
                agents.update(buffer)
            
            obs = next_obs
            step += 1
            
            if done:
                break
        
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        rewards_history.append(episode_reward)
        tasks_history.append(episode_tasks)
        
        if episode % 30 == 0:
            print(f"  Ep {episode:3d}: Reward={episode_reward:6.1f}, Tasks={episode_tasks}, Eps={epsilon:.3f}")
    
    return {"name": name, "rewards": rewards_history, "tasks": tasks_history}


def visualize_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = ['gray', 'blue', 'green', 'orange']
    
    for idx, (name, res) in enumerate(results.items()):
        rewards = res["rewards"]
        smoothed = np.convolve(rewards, np.ones(5)/5, mode='valid')
        axes[0, 0].plot(smoothed, label=name, color=colors[idx], linewidth=2)
    
    axes[0, 0].legend()
    axes[0, 0].set_title("Training Reward (smoothed)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    
    for idx, (name, res) in enumerate(results.items()):
        tasks = res["tasks"]
        smoothed = np.convolve(tasks, np.ones(5)/5, mode='valid')
        axes[0, 1].plot(smoothed, label=name, color=colors[idx], linewidth=2)
    
    axes[0, 1].legend()
    axes[0, 1].set_title("Tasks Completed (smoothed)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Tasks")
    axes[0, 1].grid(True, alpha=0.3)
    
    final_rewards = [np.mean(r["rewards"][-20:]) for r in results.values()]
    final_tasks = [np.mean(r["tasks"][-20:]) for r in results.values()]
    
    axes[1, 0].bar(range(len(results)), final_rewards, color=colors[:len(results)], edgecolor='black')
    axes[1, 0].set_xticks(range(len(results)))
    axes[1, 0].set_xticklabels(list(results.keys()), rotation=30, ha='right')
    axes[1, 0].set_title("Final Mean Reward")
    axes[1, 0].set_ylabel("Reward")
    
    axes[1, 1].bar(range(len(results)), final_tasks, color=colors[:len(results)], edgecolor='black')
    axes[1, 1].set_xticks(range(len(results)))
    axes[1, 1].set_xticklabels(list(results.keys()), rotation=30, ha='right')
    axes[1, 1].set_title("Final Tasks Completed")
    axes[1, 1].set_ylabel("Tasks")
    
    plt.tight_layout()
    plt.savefig("visualizations/05_algorithm_comparison.png", dpi=150, bbox_inches='tight')
    print("\nSaved: visualizations/05_algorithm_comparison.png")
    plt.close()


def main():
    print("="*50)
    print("FleetSync-MARL Benchmark: Algorithm Comparison")
    print("="*50)
    
    from src.env.environment import SimpleWarehouseEnv
    
    results = {}
    
    results["Random"] = run_algorithm(SimpleWarehouseEnv, RandomAgent, num_episodes=150, name="Random")
    results["IQL"] = run_algorithm(SimpleWarehouseEnv, IQLAgent, num_episodes=150, name="IQL")
    results["VDN"] = run_algorithm(SimpleWarehouseEnv, VDN, num_episodes=150, name="VDN")
    results["QMIX"] = run_algorithm(SimpleWarehouseEnv, QMIXAgent, num_episodes=150, name="QMIX")
    
    visualize_comparison(results)
    
    print("\n" + "="*50)
    print("Summary (last 20 episodes):")
    print("="*50)
    for name, res in results.items():
        final_reward = np.mean(res["rewards"][-20:])
        final_tasks = np.mean(res["tasks"][-20:])
        print(f"  {name:8s}: Reward={final_reward:7.1f}, Tasks={final_tasks:.1f}")


if __name__ == "__main__":
    import os
    os.makedirs("visualizations", exist_ok=True)
    main()