import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobotActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class QMIXNetwork(nn.Module):
    def __init__(
        self,
        num_agents,
        obs_dim,
        n_actions,
        hidden_dim=64,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.actors = nn.ModuleList([
            RobotActor(obs_dim, n_actions, hidden_dim)
            for _ in range(num_agents)
        ])

        self.state_encoder = nn.Linear(num_agents * obs_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, state=None):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if state is not None and isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # Handle (num_agents, batch, obs_dim) -> (batch, num_agents, obs_dim)
        if obs.dim() == 3:
            if obs.size(0) == self.num_agents:
                obs = obs.transpose(0, 1)
            elif obs.size(-1) == self.obs_dim:
                pass
            else:
                obs = obs.view(-1, self.num_agents, self.obs_dim)
        elif obs.dim() == 2 and obs.size(0) == self.num_agents:
            obs = obs.transpose(0, 1).unsqueeze(0)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)

        # Now obs is (batch, num_agents, obs_dim)
        if obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)

        q_vals = []
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i]
            q = actor(agent_obs)
            q_vals.append(q)

        q_vals = torch.stack(q_vals, dim=0)  # (num_agents, batch, n_actions)
        q_vals = q_vals.transpose(0, 1)  # (batch, num_agents, n_actions)

        total_q = q_vals.mean(dim=(1, 2))  # (batch,)

        return total_q, q_vals

    def get_actions(self, obs, state, epsilon=0.1):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        if obs.dim() == 2 and obs.size(0) == self.num_agents:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if obs.size(-1) != self.obs_dim:
            obs = obs.view(-1, self.num_agents, self.obs_dim)

        actions = []
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i]
            with torch.no_grad():
                q = actor(agent_obs)
            if np.random.rand() < epsilon:
                a = np.random.randint(self.n_actions)
            else:
                a = q.argmax(dim=-1).item()
            actions.append(a)

        return actions


class SimpleQMIX:
    def __init__(
        self,
        num_agents,
        obs_dim,
        n_actions,
        hidden_dim=64,
        device="cpu",
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

        self.model = QMIXNetwork(
            num_agents=num_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(device)

        self.target_model = QMIXNetwork(
            num_agents=num_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def update(self, batch, gamma=0.99):
        obs_batch = torch.FloatTensor(batch["obs"]).to(self.device)
        rewards_batch = torch.FloatTensor(batch["reward"]).to(self.device)
        next_obs_batch = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones_batch = torch.FloatTensor(batch["done"]).to(self.device)

        q_tot, _ = self.model(obs_batch, None)

        with torch.no_grad():
            next_q_tot, _ = self.target_model(next_obs_batch, None)
            target = rewards_batch + gamma * (1 - dones_batch) * next_q_tot

        loss = F.mse_loss(q_tot.mean(), target.mean())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_actions(self, obs, state, epsilon=0.1):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        state_tensor = torch.FloatTensor(state).to(self.device) if state is not None else None

        return self.model.get_actions(obs_tensor, state_tensor, epsilon)