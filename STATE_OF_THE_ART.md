# FleetSync-MARL: State of the Art Comparison

This document benchmarks FleetSync-MARL against other Multi-Agent Reinforcement Learning (MARL) algorithms.

---

## MARL Algorithm Categories

### Value-Based Methods

| Algorithm | Centralized Training | Decentralized Execution | Key Feature |
|-----------|-------------------|----------------------|-------------|
| **IQL** (Independent Q-Learning) | Per-agent | Per-agent | Simple, no coordination |
| **QMIX** (our approach) | Joint Q-function | Per-agent | Value decomposition, monotonicity |
| **VDN** (Value Decomposition Network) | Joint Q-function | Per-agent | Additive decomposition |

### Policy Gradient Methods

| Algorithm | Centralized Training | Decentralized Execution | Key Feature |
|-----------|-------------------|----------------------|-------------|
| **MADDPG** | Centralized critic | Decentralized actors | CTDE, continuous actions |
| **MAPPO** | Centralized critic | Decentralized actors | On-policy, PPO |
| **MAVEN** | Hierarchical latent | Multi-agent variable | Temporal abstraction |

### Communication Methods

| Algorithm | Communication | Bandwidth | Key Feature |
|-----------|--------------|----------|------------|
| **CommNet** | Continuous | Limited | Learn to communicate |
| **TarMAC** | Selective | Attention-based | Who to talk to |
| **RMA** | Differentiable | Full | Differentiable comms |

---

## Algorithm Implementations

### 1. Independent Q-Learning (IQL)

Simple baseline: each agent learns independently using DQN.

```python
class IQLAgent:
    def __init__(self, obs_dim, n_actions):
        self.q_network = DQN(obs_dim, n_actions)
    
    def update(self, obs, action, reward, next_obs, done):
        # Standard DQN update
        pass
    
    def get_action(self, obs, epsilon):
        # Epsilon-greedy
        pass
```

**Pros:** Simple, scales well
**Cons:** Non-stationarity issue, no explicit coordination

### 2. QMIX (Value Mixer)

Centralized training with value decomposition. Our approach.

```python
class QMIX:
    def forward(self, obs, state):
        # Individual Q-values
        individual_qs = [actor_i(obs_i) for actor_i in self.actors]
        
        # Mixing network (monotonic)
        total_q = self.mixer(individual_qs, state)
        
        return total_q
```

**Pros:** Ensures IGM principle, efficient coordination
**Cons:** Requires state information

### 3. VDN (Value Decomposition Network)

Simpler than QMIX: additive value decomposition.

```python
class VDN:
    def forward(self, obs):
        individual_qs = [actor_i(obs_i) for actor_i in self.actors]
        total_q = sum(individual_qs)  # Additive
        return total_q
```

**Pros:** Simple, interpretable
**Cons:** Less expressive than QMIX

### 4. MADDPG (Multi-Agent DDPG)

Policy gradient for continuous actions.

```python
class MADDPG:
    def update(self, obs, actions, rewards, next_obs):
        # Centralized critic for all agents
        critic_loss = self.critic(obs, actions).mean()
        
        # Decentralized policy gradient
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        
        return critic_loss + actor_loss
```

**Pros:** Continuous actions, stable
**Cons:** Requires centralized critic at test time

### 5. CommNet (Communication Network)

Learn communication protocol between agents.

```python
class CommNet:
    def forward(self, obs):
        # Encode observations
        h = self.encoder(obs)
        
        # Communication round
        messages = []
        for i in range(self.comm_rounds):
            m = self.comm_layer(h[i], h[j])  # Messages from others
            h[i] = h[i] + m  # Residual connection
        
        return self.policy(h)
```

**Pros:** Explicit communication
**Cons:** Real-world deployment challenges

---

## Experimental Comparison

### Benchmark Settings

| Setting | Value |
|--------|-------|
| Warehouse Size | 15x15 |
| Num Agents | 5 |
| Num Tasks | 20 |
| Max Steps | 500 |
| Obstacle Density | 10% |
| Episodes | 500 |
| Eval Episodes | 50 |

### Expected Results

| Algorithm | Mean Reward | Tasks Completed | Training Time |
|-----------|------------|----------------|--------------|
| Random | -200 to -150 | 0-1 | N/A |
| IQL | -80 to -50 | 3-5 | 5 min |
| VDN | -60 to -40 | 5-8 | 8 min |
| **QMIX** | **-40 to -30** | **8-12** | **10 min** |
| MADDPG | -50 to -35 | 6-10 | 15 min |
| CommNet | -45 to -30 | 7-11 | 20 min |

### Training Curves

```
Reward
  ^
-20|         ___________
-30|        /           \________
-40|       /                    \_____
-50| _____/                         \___
-60|/
-70|/
-80|/
  +-------------------------> Episodes
  0    100   200   300   400   500
```

---

## Key Metrics Explained

### 1. Mean Episode Reward

Average total reward per episode. Higher is better.

### 2. Tasks Completed

Number of pick-up/delivery tasks completed per episode.

### 3. Collision Rate

Percentage of steps with robot-robot collisions.

### 4. Training Time

Time to reach convergence (reward > -50).

### 5. Sample Efficiency

Episodes needed to reach 90% of final performance.

---

## Implementation Notes

### Why QMIX Wins

1. **CTDE**: Centralized training, decentralized execution
2. **Value Decomposition**: Solves credit assignment
3. **Monotonicity**: Ensures IGM principle

### Why IQL Loses

1. **Non-stationarity**: Other agents' policies change
2. **Environment non-stationarity**: Rewards shift
3. **No explicit coordination**: Emerges implicitly

### Why Communication Helps

1. **Explicit coordination**: Removes ambiguity
2. **Information sharing**: Partial observability solved
3. **Attention**: Focus on relevant agents

---

## References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| QMIX (Rashid et al.) | 2020 | Value decomposition for cooperative MARL |
| VDN (Sunehag et al.) | 2018 | Additive value decomposition |
| MADDPG (Lowe et al.) | 2017 | Multi-agent DDPG with centralized critic |
| CommNet (Foerster et al.) | 2016 | Differentiable communication |
| MAPPO (Kuba et al.) | 2022 | Multi-agent PPO with centralized critic |

---

## Running Benchmarks

```bash
python benchmark.py --algorithms iql qmix maddpg vdn --episodes 500
```

This generates comparison plots for all algorithms.

---

*Last Updated: April 2026*
*Author: Nathan Kermani*