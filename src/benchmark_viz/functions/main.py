# src/benchmark_viz/functions/main.py

import numpy as np
from src.env.environment import SimpleWarehouseEnv
from src.benchmark_viz.random_agent import RandomAgent
from src.benchmark_viz.iql_agent import IQLAgent
from src.benchmark_viz.vdn import VDN
from src.benchmark_viz.qmix_agent_benchmark import QMIXAgent as QMIXAgentBenchmark
from .visualize_comparison import visualize_comparison
from .run_algorithm import run_algorithm

def main():
    print("="*50)
    print("FleetSync-MARL Benchmark: Algorithm Comparison")
    print("="*50)

    results = {}

    results["Random"] = run_algorithm(SimpleWarehouseEnv, RandomAgent, num_episodes=150, name="Random")
    results["IQL"] = run_algorithm(SimpleWarehouseEnv, IQLAgent, num_episodes=150, name="IQL")
    results["VDN"] = run_algorithm(SimpleWarehouseEnv, VDN, num_episodes=150, name="VDN")
    results["QMIX"] = run_algorithm(SimpleWarehouseEnv, QMIXAgentBenchmark, num_episodes=150, name="QMIX")

    visualize_comparison(results)

    print("\n" + "="*50)
    print("Summary (last 20 episodes):")
    print("="*50)
    for name, res in results.items():
        final_reward = np.mean(res["rewards"][-20:])
        final_tasks = np.mean(res["tasks"][-20:])
        print(f"  {name:8s}: Reward={final_reward:7.1f}, Tasks={final_tasks:.1f}")
