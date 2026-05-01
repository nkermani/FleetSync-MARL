# src/benchmark_viz/visualize/visualize_training_progress.py

import matplotlib.pyplot as plt

def visualize_training_progress(buffer_sizes, losses, rewards):
    """Visualize training progress over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(buffer_sizes, linewidth=2)
    axes[0].set_title("Buffer Size")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Samples")

    axes[1].plot(losses, linewidth=2, color='orange')
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("MSE Loss")

    axes[2].plot(rewards, linewidth=2, color='green')
    axes[2].set_title("Episode Reward")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Total Reward")

    plt.tight_layout()
    return fig
