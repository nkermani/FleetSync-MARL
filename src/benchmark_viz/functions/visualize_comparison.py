# src/benchmark_viz/functions/visualize_comparison.py

import numpy as np
import matplotlib.pyplot as plt

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
