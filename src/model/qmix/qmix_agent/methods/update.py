# src/model/qmix/qmix_agent/methods/update.py

import torch
import torch.nn.functional as F

def update(self, batch, gamma=0.99):
    obs = torch.FloatTensor(batch["obs"]).to(self.device)
    state = torch.FloatTensor(batch["state"]).to(self.device)
    actions = torch.LongTensor(batch["actions"]).to(self.device)
    rewards = torch.FloatTensor(batch["reward"]).to(self.device)
    next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
    next_state = torch.FloatTensor(batch["next_state"]).to(self.device)
    dones = torch.FloatTensor(batch["done"]).to(self.device)

    individual_qs = self.model(obs)
    selected_q = individual_qs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    total_q = self.model.mixer(selected_q, state)

    with torch.no_grad():
        next_individual_qs = self.target_model(next_obs)
        next_actions = next_individual_qs.argmax(dim=-1)
        next_selected_q = next_individual_qs.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
        next_total_q = self.target_model.mixer(next_selected_q, next_state)
        target_q = rewards + gamma * (1 - dones) * next_total_q

    loss = F.mse_loss(total_q, target_q)

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
    self.optimizer.step()

    return loss.item()
