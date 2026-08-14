"""Minimal matplotlib visualization helpers (no seaborn dependency)."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trials(env, n_trials: int = 2, save_path=None):
    """Plot ``n_trials`` sampled trials as a trials x timesteps image grid."""
    env.reset()
    observations, ground_truths = [], []
    for _ in range(n_trials):
        env.new_trial()
        observations.append(env.ob.copy())
        ground_truths.append(env.gt.copy())

    n_steps = observations[0].shape[0]
    fig, axes = plt.subplots(
        n_trials, n_steps,
        squeeze=False,
        figsize=(1.4 * n_steps, 1.7 * n_trials),
    )
    for row in range(n_trials):
        for col in range(n_steps):
            ax = axes[row][col]
            ax.imshow(observations[row][col], cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"t={col * env.dt}ms\ngt={int(ground_truths[row][col])}",
                fontsize=7,
            )
            ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history: dict, save_path=None):
    """Plot loss and accuracy from a ``train()`` history dict."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, history["accuracy"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
