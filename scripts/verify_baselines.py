"""Verification checkpoint 2: every baseline lands where theory says.

Trains each model once at 64px and prints accuracy with the abstention split.
Expected, on the balanced/growth condition:

    FF2Only  ~0.50   provable ceiling
    FF1Only  ~0.818  measured Bayes-optimal ceiling for the `train` split's
                      bar_centre_range=(0.0, 0.6) -- narrowing that band
                      makes shape position MORE diagnostic of direction, not
                      less (see the comment on SPLITS["train"] in
                      illusion_rnn/generate.py for the full explanation and
                      the by-band numbers). This is NOT a smooth neural
                      classifier beating a per-pixel heuristic; a static
                      frame-1-only Bayes-optimal classifier already lands
                      here. Do not be alarmed if FF1Only lands well above the
                      spec's original "~55%" design-time guess -- that guess
                      predates this measurement and has been superseded by it.
    FFStack  high    has both frames, no recurrence
    GRUNet   high    recurrence-type control for RNNNet
    RNNNet   high    the binding model

Run: .venv/bin/python scripts/verify_baselines.py
"""
import numpy as np
import torch

from illusion_rnn import evaluate, make_dataset, make_env, train
from illusion_rnn.models import FFStack, FrameOnlyNet, GRUNet, RNNNet

IMG, HIDDEN, EPOCHS = 64, 256, 1500


def build(name, env, n_steps):
    size = IMG * IMG
    if name == "RNNNet":
        return RNNNet(size, HIDDEN, 6, dt=50)
    if name == "GRUNet":
        return GRUNet(size, HIDDEN, 6)
    if name == "FFStack":
        return FFStack(size, HIDDEN, 6, n_steps=n_steps)
    if name == "FF1Only":
        return FrameOnlyNet(size, HIDDEN, 6, n_steps=n_steps,
                            frame_index=env.frame1_index)
    if name == "FF2Only":
        return FrameOnlyNet(size, HIDDEN, 6, n_steps=n_steps,
                            frame_index=env.frame2_indices[0])
    raise ValueError(f"Unknown model {name!r}")


def main():
    torch.manual_seed(0)
    env = make_env("correspondence", split="train", img_size=IMG)
    env.seed(0)
    # env.ob does not exist as an attribute until a trial has actually run
    # (it is set by new_trial/reset, not by the constructor), so it must be
    # read after new_trial() rather than guarded with `is not None`.
    env.new_trial()
    n_steps = env.ob.shape[0]

    for name in ("FF2Only", "FF1Only", "FFStack", "GRUNet", "RNNNet"):
        model = build(name, env, n_steps)
        # seq_len is deliberately the length of exactly ONE trial (n_steps),
        # not a multiple of it. ngym.Dataset concatenates trials to fill
        # whatever seq_len it is given, but FrameOnlyNet and FFStack both
        # fix their first linear layer's input width to n_steps at
        # construction and raise/shape-error on any other T. Using
        # seq_len=n_steps keeps every architecture -- fixed-window
        # (FF1Only/FF2Only/FFStack) and free-running recurrent (GRUNet/
        # RNNNet) alike -- on the same one-trial-per-batch-row training
        # regime, so the comparison across architectures stays apples-to-
        # apples.
        dataset = make_dataset(make_env("correspondence", split="train",
                                        img_size=IMG), batch_size=64,
                               seq_len=n_steps)
        train(model, dataset, n_epochs=EPOCHS, log_every=0)
        eval_env = make_env("correspondence", split="train", img_size=IMG)
        eval_env.seed(1)
        r = evaluate(model, eval_env, n_trials=500)
        print(f"{name:9s} acc={r.accuracy:.3f}  abstain={r.abstention_rate:.3f}  "
              f"committed={r.committed_accuracy:.3f}")


if __name__ == "__main__":
    main()
