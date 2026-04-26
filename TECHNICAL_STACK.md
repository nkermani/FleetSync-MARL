# FleetSync-MARL Technical Stack

## Overview

This project showcases proficiency with Multi-Agent Reinforcement Learning, Graph Neural Networks, and distributed systems coordination.

---

## Technology Categories

### Reinforcement Learning

| Technology | Usage | Level |
|------------|-------|-------|
| **PyTorch** | Neural network framework | Advanced |
| **PyTorch Geometric** | Graph neural networks for spatial relationships | Advanced |
| **Stable Baselines 3** | RL algorithms (reference) | Intermediate |
| **OpenAI Gym** | Environment interface | Advanced |
| **Gymnasium** | Modern Gym API | Intermediate |

### Multi-Agent Systems

| Technology | Usage | Level |
|------------|-------|-------|
| **QMIX** | Value decomposition for cooperative MARL | Advanced |
| **Centralized Training** | Joint Q-value learning | Advanced |
| **Decentralized Execution** | Independent inference | Advanced |
| **Replay Buffer** | Experience replay | Advanced |

### Scientific Computing

| Technology | Usage | Level |
|------------|-------|-------|
| **NumPy** | Array operations | Advanced |
| **Pandas** | Data analysis | Intermediate |
| **Matplotlib** | Visualization | Advanced |

### Development

| Technology | Usage | Level |
|------------|-------|-------|
| **pytest** | Unit testing | Intermediate |
| **git** | Version control | Advanced |
| **Jupyter** | Interactive notebooks | Intermediate |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CENTRALIZED TRAINING                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JOINT STATE (S)                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Robot 1 │  │ Robot 2 │  │ Robot 3 │  │ ...     │            │
│  │ pos, v  │  │ pos, v  │  │ pos, v  │  │         │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │             │             │             │                    │
│       ▼             ▼             ▼             ▼                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              GNN ENCODER (Optional)                         │  │
│  │            Aggregate spatial relationships                │  │
│  └────────────────────────┬────────────────────────────────┘  │
└────────────────────────────┼───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INDIVIDUAL Q-VALUES                           │
│                                                                 │
│   Q_1(o_1, a_1)  Q_2(o_2, a_2)  ...  Q_n(o_n, a_n)          │
│        │               │                      │                        │
│        └──────────────┼──────────────────┘                        │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  MIXING NETWORK                          │   │
│  │     Q_tot = f(Q_1, Q_2, ..., Q_n, state)                │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    JOINT Q-VALUE                          │
│  │            Q_tot(s, a_1, ..., a_n)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DECENTRALIZED EXECUTION                         │
│                                                                 │
│   At test time, only individual Q-values are available:           │
│                                                                 │
│      Robot 1:  argmax Q_1(o_1, a)                             │
│      Robot 2:  argmax Q_2(o_2, a)                             │
│      ...                                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example

```python
# 1. Environment Reset (Warehouse)
obs = env.reset()
# Output: dict with 'agents': (5, 17), 'tasks': (20, 4)

# 2. Get Individual Observations
agent_obs = [obs['agents'][i] for i in range(num_agents)]
# Output: list of 5 arrays, each (17,)

# 3. Individual Q-Value Computation
q_values = [actor(agent_obs[i]) for i in range(num_agents)]
# Output: list of 5 arrays, each (5,)

# 4. Joint Q-Value Mixing
joint_q = mixer(q_values, obs['global_state'])
# Output: (1,) scalar Q-value

# 5. Action Selection (Epsilon-Greedy)
actions = [np.argmax(q) if random.random() > epsilon
           else random.randint(n_actions) for q in q_values]
# Output: list of 5 actions

# 6. Environment Step
next_obs, rewards, done, info = env.step(actions)
# Output: rewards list, done boolean
```

---

## Key Implementation Details

### CTDE (Centralized Training, Decentralized Execution)

```python
class QMIXNetwork(nn.Module):
    def forward(self, obs, state):
        # Training: Use both local and global info
        individual_qs = [actor(obs[i]) for actor in self.actors]

        # Mixing requires global state
        joint_q = self.mixer(individual_qs, state)

        # Loss computed on joint Q
        return joint_q

    def get_actions(self, obs, epsilon=0.1):
        # Execution: Only local observations
        actions = []
        for i, actor in enumerate(self.actors):
            q_vals = actor(obs[i])
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                actions.append(np.argmax(q_vals))
        return actions
```

### Experience Replay Buffer

```python
class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents):
        self.obs = [deque(maxlen=capacity) for _ in range(num_agents)]
        self.actions = [deque(maxlen=capacity) for _ in range(num_agents)]
        self.rewards = deque(maxlen=capacity)
        self.next_obs = [deque(maxlen=capacity) for _ in range(num_agents)]
        self.dones = deque(maxlen=capacity)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        return {k: [v[i] for i in indices] for k, v in ...}
```

### GNN for Spatial Relationships

```python
class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)

    def forward(self, node_features, edge_index):
        # node_features: (num_agents, feature_dim)
        # edge_index: (2, num_edges) - connectivity between nearby agents
        embeddings = self.gnn(node_features, edge_index)
        return embeddings
```

---

## Testing Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| WarehouseEnv | 15 tests | ✅ |
| Robot Agent | 8 tests | ✅ |
| QMIXNetwork | 10 tests | ✅ |
| MixingNetwork | 6 tests | ✅ |
| ReplayBuffer | 5 tests | ✅ |

Run with: `pytest tests/ -v`

---

## Reproducibility

```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
```

---

*Technologies demonstrate readiness for Multi-Agent RL and robotics research positions*