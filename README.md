# FleetSync-MARL

*Multi-Agent Reinforcement Learning for Robot Fleet Coordination*

## Overview

FleetSync-MARL is a research-oriented project that uses Multi-Agent Reinforcement Learning (MARL) to solve coordinated routing problems for robot fleets in warehouse environments. It addresses the challenge of orchestrating multiple robots for pick-up and delivery services while avoiding collisions and optimizing throughput.

## Installation

```bash
git clone https://github.com/nkermani/FleetSync-MARL.git
cd FleetSync-MARL
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- NumPy, Pandas, Matplotlib
- OpenAI Gym
- Stable Baselines 3

## Quick Start

### 1. Create Warehouse Environment

```python
from src.env import WarehouseEnv

env = WarehouseEnv(
    num_agents=5,          # Number of robots
    num_tasks=20,          # Pending pick-up/delivery tasks
    grid_size=20,         # Warehouse dimensions
    max_steps=500,          # Episode length
    collision_penalty=-10 # Penalty for robot-robot collision
)

obs = env.reset()
print(f"Observation shape: {obs['agents'].shape}")
```

### 2. Define Multi-Agent Network

```python
from src.model import QMIX, RobotActor

num_agents = 5
obs_dim = 17         # Agent observation dimension
n_actions = 5       # Actions: up, down, left, right, wait

# Individual actor networks
actors = [RobotActor(obs_dim, n_actions) for _ in range(num_agents)]

# QMIX mixer for joint Q-values
mixer = QMIX(
    state_dim=num_agents * obs_dim,
    num_agents=num_agents,
    mixing_embed_dim=32
)

model = QMIXNetwork(actors, mixer)
```

### 3. Train the MARL Agent

```python
from src.train import Trainer, ReplayBuffer

buffer = ReplayBuffer(capacity=100000)
trainer = Trainer(model, learning_rate=1e-4, gamma=0.99)

episodes = 1000
for episode in range(episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        actions = model.step(obs)  # Epsilon-greedy action selection
        next_obs, reward, done, info = env.step(actions)

        buffer.push(obs, actions, reward, next_obs, done)
        episode_reward += reward

        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            loss = trainer.update(batch)
        obs = next_obs

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

### 4. Evaluate Performance

```python
import numpy as np

eval_episodes = 100
total_rewards = []

for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    tasks_completed = 0

    while not done:
        actions = model.predict(obs)  # Greedy (no exploration)
        obs, reward, done, info = env.step(actions)
        episode_reward += reward
        tasks_completed += info.get('tasks_completed', 0)

    total_rewards.append(episode_reward)

print(f"Mean Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
```

## Core Components

### `WarehouseEnv`

Multi-agent warehouse environment with:
- **Grid-based warehouse**: Obstacles, pick-up zones, delivery zones
- **Collision detection**: Robot-robot and robot-obstacle collisions
- **Task system**: Pending tasks queue with rewards
- **Partially observable**: Agents see local neighborhood

```python
# Methods
obs = env.reset()                    # Initialize environment
obs, reward, done, info = env.step(actions)  # Take action
env.render()                      # Visualize (requires matplotlib)
```

### `QMIXNetwork`

QMIX-style value decomposition:
- **Individual Q-networks**: Each agent has its own Q-function
- **Mixing network**: Combines Q-values into joint Q-function
- **CTDE**: Centralized training, decentralized execution

```python
model = QMIXNetwork(
    obs_dim=17,
    n_actions=5,
    num_agents=5,
    hidden_dim=64
)
```

### `RobotActor`

Individual agent policy network:
- **Shared encoder**: Graph neural network for spatial relationships
- **Action head**: Q-value predictions per action

```python
actor = RobotActor(
    obs_dim=17,
    n_actions=5,
    use_gnn=True
)
```

### `ReplayBuffer`

Multi-agent experience replay buffer:
- ** stores**: (obs, actions, reward, next_obs, done) for all agents
- **Sampling**: Random batch sampling for training

```python
buffer = ReplayBuffer(capacity=100000)
buffer.push(obs, actions, reward, next_obs, done)
batch = buffer.sample(batch_size)
```

## Training Pipeline

```python
from src.env import WarehouseEnv
from src.model import QMIXNetwork
from src.train import Trainer, ReplayBuffer

env = WarehouseEnv(num_agents=5, grid_size=20)
model = QMIXNetwork(obs_dim=17, n_actions=5, num_agents=5)
buffer = ReplayBuffer(capacity=100000)
trainer = Trainer(model, lr=1e-4)

for episode in range(5000):
    obs = env.reset()
    done = False

    while not done:
        actions = model.get_exploratory_actions(obs, epsilon=0.1)
        next_obs, rewards, done, infos = env.step(actions)

        total_reward = sum(rewards)
        buffer.push(obs, actions, total_reward, next_obs, done)

        if len(buffer) >= 32:
            batch = buffer.sample(32)
            loss = trainer.update(batch)

        obs = next_obs

    if episode % 500 == 0:
        eval_reward = evaluate(env, model, num_episodes=10)
        print(f"Episode {episode}: Eval Reward = {eval_reward:.2f}")
```

## Project Structure

```
FleetSync-MARL/
├── data/                   # Dataset storage
├── src/
│   ├── env/
│   │   ├── warehouse.py   # Warehouse environment
│   │   ├── robot.py       # Robot agent class
│   │   └── task.py       # Task definition
│   ├── model/
│   │   ├── qmix.py       # QMIX network
│   │   ├── actor.py      # Actor network
│   │   └── mixer.py     # Value mixing network
│   └── train.py         # Training loop
├── tests/
│   ├── test_env.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Visualization

Generate visualizations to understand agent behavior:

```bash
python visualize.py
```

This creates:
- Warehouse layout with robot positions
- Task completion over time
- Collision avoidance heatmap
- Reward curves during training

## Running Tests

```bash
pytest tests/ -v
```

## Citation

If you use FleetSync-MARL in your research:

> Kermani, N. (2024). FleetSync-MARL: Multi-Agent Reinforcement Learning for Robot Fleet Coordination. Learning-Based Optimization Research Track.

## Connection to N-Drill-Master-RL

This project is part of a two-part exploration of learning-augmented robotics:
- **FleetSync-MARL**: Multi-agent coordination (this project)
- **N-Drill-Master-RL**: Single-agent drilling/optimization