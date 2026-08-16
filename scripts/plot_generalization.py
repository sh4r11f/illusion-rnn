#!/usr/bin/env python3
"""Regenerate figures/generalization.png from corrected variant numbers.

Evaluates the shipped pixel RNN on the *standard* (training-distribution) and
*outline* (genuinely held-out) TAM variants, shape-by-shape, and plots a
grouped bar chart. `basic` is deliberately excluded: all 24 of its images
are byte-identical to `TAM_task` images, so it is not a held-out variant --
see tests/test_stimuli_integrity.py and the "Provenance" note in
checkpoints/MANIFEST.md.

Usage:
    uv run python scripts/plot_generalization.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display needed to write the PNG

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from illusion_rnn import SHAPES, evaluate, load_rnn, make_env  # noqa: E402

CHECKPOINT = Path("checkpoints/rnn-pixel_h2048_tam-horiz.pt")
VARIANTS = ("standard", "outline")
N_TRIALS = 100
SEED = 0
OUT = Path("figures/generalization.png")


def main():
    if not CHECKPOINT.exists() or CHECKPOINT.stat().st_size <= 1024:
        msg = (
            f"{CHECKPOINT} is missing or an LFS pointer; run "
            "'git lfs install --local && git lfs checkout' first"
        )
        raise SystemExit(msg)

    model = load_rnn(CHECKPOINT)

    # results[variant][shape] = accuracy over N_TRIALS trials
    results = {}
    for variant in VARIANTS:
        accs = []
        for shape in SHAPES:
            env = make_env(
                "tam", box_shape=shape, variant=variant,
                stim_ori="horizontal", img_size=64,
            )
            env.seed(SEED)
            r = evaluate(model, env, n_trials=N_TRIALS, device="cpu")
            accs.append(r.accuracy)
        results[variant] = accs
        print(f"{variant:9s}", {s: round(a, 3) for s, a in zip(SHAPES, accs)})

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(VARIANTS))
    width = 0.25
    for i, shape in enumerate(SHAPES):
        ax.bar(
            x + (i - 1) * width,
            [results[v][i] for v in VARIANTS],
            width, label=shape,
        )
    ax.axhline(1 / 3, color="gray", ls="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Accuracy ({N_TRIALS} trials/bar)")
    ax.set_title("Reference RNN: TAM variant generalization (horizontal)")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
