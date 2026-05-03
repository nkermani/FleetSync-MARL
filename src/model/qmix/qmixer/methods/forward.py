# src/model/qmix/qmixer/methods/forward.py

import torch
import torch.nn.functional as F

def forward(self, q_selected, state):
    batch_size = q_selected.size(0)

    w1 = torch.relu(self.hyper_w1(state))
    w1 = w1.reshape(batch_size, self.num_agents, self.mixer_hidden_dim)
    b1 = torch.relu(self.hyper_b1(state)).unsqueeze(1)

    q_expanded = q_selected.unsqueeze(1)
    hidden = F.relu(torch.bmm(q_expanded, w1) + b1)

    w2 = torch.relu(self.hyper_w2(state))
    w2 = w2.reshape(batch_size, self.mixer_hidden_dim, 1)
    b2 = self.hyper_b2(state).unsqueeze(1)

    total_q = torch.bmm(hidden, w2) + b2
    return total_q.squeeze(-1).squeeze(-1)
