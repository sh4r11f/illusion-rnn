"""Verification checkpoint 1: the generator carries no frame-2 label cue.

Prints per-family maximum per-pixel mutual information between frame 2 and the
direction label, and writes a sample-trial figure. The balanced family must be
at the noise floor; the classic family must be well above it, or it is not
diagnosing anything.

Run: .venv/bin/python scripts/verify_generator.py
"""
import matplotlib.pyplot as plt
import numpy as np

from illusion_rnn.generate import pixel_label_mi, render_trial, sampler_for

N_TRIALS = 10_000


def collect(family, split="train"):
    sample = sampler_for(split, family=family)
    rng = np.random.default_rng(0)
    trials = [render_trial(sample(rng)) for _ in range(N_TRIALS)]
    labels = np.array([t.label == "right" for t in trials])
    return trials, labels


def main():
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, family in enumerate(("balanced", "classic")):
        trials, labels = collect(family)
        for col, frame_name in enumerate(("frame1", "frame2")):
            frames = np.array([getattr(t, frame_name) for t in trials])
            mi = pixel_label_mi(frames, labels)
            print(f"{family:9s} {frame_name}: max per-pixel MI = {mi.max():.4f} bits")
        # sample trials: one leftward, one rightward
        for col, want in enumerate(("left", "right")):
            t = next(t for t in trials if t.label == want)
            axes[row, col * 2].imshow(t.frame1, cmap="gray_r", vmin=0, vmax=1)
            axes[row, col * 2].set_title(f"{family} {want}\nframe 1", fontsize=8)
            axes[row, col * 2 + 1].imshow(t.frame2, cmap="gray_r", vmin=0, vmax=1)
            axes[row, col * 2 + 1].set_title("frame 2", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("figures/correspondence_trials.png", dpi=150)
    print("wrote figures/correspondence_trials.png")


if __name__ == "__main__":
    main()
