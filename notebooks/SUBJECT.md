# FleetSync-MARL 🤖

*Multi-Agent Reinforcement Learning for Robot Fleet Coordination*

## 📌 Overview

FleetSync-MARL is a research-oriented project exploring the intersection of Multi-Agent Reinforcement Learning (MARL) and Neural Combinatorial Optimization.

## 🚀 The Challenge: Beyond Single-Agent RL

Traditional Reinforcement Learning (RL) handles sequential decision-making for a single agent. However, real-world robotics services involve **multiple agents coordinating simultaneously**:

- **Multi-Agent Coordination**: Multiple robots must avoid collisions while optimizing delivery paths
- **Partial Observability**: Each robot observes only local environment state
- **Non-Stationarity**: Other agents' policies change during learning, breaking standard RL assumptions
- **Scalability**: Fleet size can range from 3-5 robots to 50+ in warehouses

FleetSync-MARL handles this by implementing centralized training with decentralized execution (CTDE), allowing agents to learn coordinated policies that work in real-time.

## 🔬 Research Context

> *"With the growing development of robotics services, the problem of orchestrating a fleet of robots (or autonomous agents) under various constraints has recently become a major design bottleneck, especially when seeking to optimise service operations."*

### Two Traditions Addressed

| Approach | Strengths | Weaknesses |
|----------|----------|------------|
| **Reinforcement Learning** | Learns from experience, adapts to uncertainty | May not find global optimum |
| **Multi-Agent Planning** | Guarantees optimal plans with perfect model | Doesn't scale to uncertainty, brittle |

FleetSync-MARL explores the **intersection** of both traditions, using learned policies augmented by planning heuristics.

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| RL Framework | PyTorch, Stable Baselines 3 |
| Graph Neural Networks | PyTorch Geometric |
| Environment | Custom Multi-Agent Gym (OpenAI Gym wrapper) |
| Simulation | NumPy-based warehouse environment |

### Architecture

- **Centralized Critic**: Learns joint Q-values considering all agents
- ** Decentralized Actors**: Each robot acts based on local observations
- **GNN Encoder**: Encodes spatial relationships between robots/waypoints

## 🧠 How It Works

1. **Environment**: Warehouse with robot fleet, pick-up and delivery zones
2. **State Space**: Joint positions, velocities, task assignments
3. **Action Space**: Discrete navigation actions per robot
4. **Reward**: Tasks completed - collision penalties - time penalties
5. **MARL Algorithm**: QMIX-style value decomposition for coordination

## 📈 Key Research Sell

> "This project demonstrates the ability to scale single-agent RL to multi-agent domains. By leveraging Graph Neural Networks to encode spatial relationships and QMIX-style value decomposition for coordination, FleetSync-MARL moves from isolated optimization to fleet-level orchestration—directly addressing the design bottlenecks of multi-robot service operations."

## Connection to N-Drill-Master-RL

FleetSync-MARL and N-Drill-Master-RL form a complementary duo:

- **FleetSync-MARL**: Focuses on **coordination** - multiple robots working together
- **N-Drill-Master-RL**: Focuses on **drilling/optimization** - single-agent task optimization under uncertainty

Both explore different facets of the same core challenge: using learning to augment combinatorial optimization in robotics.

## 📂 Project Structure

```
├── data/               # Generated warehouse environments
├── src/
│   ├── env/            # Multi-agent environment
│   ├── model/          # Neural network architectures
│   └── train.py        # Training loops
├── tests/              # Unit tests
└── README.md
```

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/nkermani/FleetSync-MARL.git

# Install dependencies
pip install torch torch-geometric numpy gym stable-baselines3
```

1. **Multi-Agent RL**: Training multiple agents to coordinate
2. **Learning-Augmented Optimization**: Using neural networks to predict heuristics
3. **Real-Time Decision Making**: Decentralized execution
4. **Scalability**: Handling varying fleet sizes
