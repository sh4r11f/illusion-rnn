"""Multi-seed grid runner for the correspondence experiment.

One cell is an (architecture, family, hidden_size, seed) tuple. Each cell trains
one model on the `train` split and evaluates it on all four splits, producing
one record per split. `aggregate` reduces records to mean and 95% CI across
seeds.

Seeds control weight initialisation and trial sampling together, so a cell is
reproducible end to end from an integer.
"""

import numpy as np
import torch

from illusion_rnn.generate import SPLITS
from illusion_rnn.models import FFStack, FrameOnlyNet, GRUNet, RNNNet, shuffle_frames
from illusion_rnn.training import evaluate, make_dataset, make_env, train

ARCHITECTURES = (
    "RNNNet", "GRUNet", "FFStack", "FF1Only", "FF2Only", "RNNNet-shuffled",
)


def build_model(name, *, input_size, hidden_size, output_size, n_steps, env):
    """Construct one architecture by name.

    ``RNNNet-shuffled`` is an ``RNNNet``; the shuffling is applied to its inputs
    during training and evaluation, not baked into the module.
    """
    if name in ("RNNNet", "RNNNet-shuffled"):
        return RNNNet(input_size, hidden_size, output_size, dt=50)
    if name == "GRUNet":
        return GRUNet(input_size, hidden_size, output_size)
    if name == "FFStack":
        return FFStack(input_size, hidden_size, output_size, n_steps=n_steps)
    if name == "FF1Only":
        return FrameOnlyNet(input_size, hidden_size, output_size,
                            n_steps=n_steps, frame_index=env.frame1_index)
    if name == "FF2Only":
        return FrameOnlyNet(input_size, hidden_size, output_size,
                            n_steps=n_steps, frame_index=env.frame2_indices[0])
    msg = f"Unknown architecture {name!r}; expected one of {ARCHITECTURES}"
    raise ValueError(msg)


def run_cell(*, architecture, family, hidden_size, seed, img_size=64,
             transform="growth", n_epochs=1500, batch_size=64,
             n_eval_trials=500, device=None):
    """Train one model and evaluate it on every split. Returns one record per
    split."""
    torch.manual_seed(seed)
    rng_seed = seed

    def env_for(split, eval_seed):
        env = make_env("correspondence", split=split, img_size=img_size,
                       family=family, transform=transform)
        env.seed(eval_seed)
        env.new_trial()
        return env

    train_env = env_for("train", rng_seed)
    n_steps = train_env.ob.shape[0]
    model = build_model(
        architecture, input_size=img_size * img_size, hidden_size=hidden_size,
        output_size=6, n_steps=n_steps, env=train_env,
    )

    shuffled = architecture.endswith("-shuffled")
    frame_indices = (train_env.frame1_index, *train_env.frame2_indices)
    gen = torch.Generator().manual_seed(seed) if shuffled else None

    def maybe_shuffle(x):
        return shuffle_frames(x, frame_indices, gen) if shuffled else x

    # seq_len must equal a single trial's length exactly, not a multiple of
    # it: ngym.Dataset concatenates trials to fill seq_len, but FrameOnlyNet
    # and FFStack both fix n_steps at construction and raise on any other T
    # (see tests/test_sweep.py's frame-2-only gate test for the same
    # convention). run_cell must work uniformly across every architecture in
    # ARCHITECTURES, so seq_len=n_steps is the only choice compatible with
    # all six -- a multiple would work for RNNNet/GRUNet but break FF1Only,
    # FF2Only and FFStack immediately.
    dataset = make_dataset(env_for("train", rng_seed), batch_size=batch_size,
                           seq_len=n_steps)
    train(model, dataset, n_epochs=n_epochs, device=device, log_every=0,
          encoder=None, input_transform=maybe_shuffle)

    records = []
    for split in SPLITS:
        result = evaluate(model, env_for(split, 10_000 + rng_seed),
                          n_trials=n_eval_trials, device=device,
                          input_transform=maybe_shuffle)
        records.append({
            "architecture": architecture, "family": family,
            "hidden_size": hidden_size, "seed": seed, "split": split,
            "transform": transform,
            "accuracy": result.accuracy,
            "abstention_rate": result.abstention_rate,
            "committed_accuracy": result.committed_accuracy,
        })
    return records


def run_grid(*, architectures, families, hidden_sizes, seeds, **kwargs):
    """Run every (architecture, family, hidden_size, seed) combination."""
    out = []
    for architecture in architectures:
        for family in families:
            for hidden_size in hidden_sizes:
                for seed in seeds:
                    out.extend(run_cell(
                        architecture=architecture, family=family,
                        hidden_size=hidden_size, seed=seed, **kwargs,
                    ))
    return out


def aggregate(records, metric="accuracy"):
    """Mean and 95% CI of ``metric`` across seeds, per cell.

    The CI is the normal-approximation interval on the seed mean. With one seed
    the interval is NaN rather than zero -- a single run has no interval and
    must not be drawn as if it had a tight one.
    """
    cells = {}
    for r in records:
        key = (r["architecture"], r["family"], r["hidden_size"], r["split"])
        cells.setdefault(key, []).append(r[metric])

    out = []
    for (architecture, family, hidden_size, split), values in cells.items():
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        mean = float(np.nanmean(arr))
        if n < 2:
            low = high = float("nan")
        else:
            sem = float(np.nanstd(arr, ddof=1) / np.sqrt(n))
            low, high = mean - 1.96 * sem, mean + 1.96 * sem
        out.append({
            "architecture": architecture, "family": family,
            "hidden_size": hidden_size, "split": split, "n_seeds": n,
            f"{metric}_mean": mean, f"{metric}_ci_low": low,
            f"{metric}_ci_high": high,
        })
    return out
