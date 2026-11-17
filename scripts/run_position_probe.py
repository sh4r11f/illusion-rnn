"""Decode blank-frame positions from a frozen apparent-motion RNN's final state.

Reproduce the main condition with:
    uv run python scripts/run_apparent_motion.py --save-dir checkpoints/apparent
    uv run python scripts/run_position_probe.py --checkpoints checkpoints/apparent
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn.analysis import ridge_predict

FAMILIAR = ("square", "circle", "triangle")
NOVEL = "cross"
SIDE, FRAMES = 32, 8
TILE = (11, 21)
THRESHOLD, MARGIN = 0.25, 1
ALPHA = 1.0


def inside_tile(centres):
    return ((centres >= TILE[0]) & (centres < TILE[1])).all(axis=-1)


@dataclass
class Trials:
    endpoints: np.ndarray
    centres: np.ndarray
    shapes: np.ndarray


def sample_trials(n, seed, *, fit, seen):
    """Independent movies; do not reuse even a pixel-identical pair of endpoints."""
    rng = np.random.default_rng(seed)
    endpoints, centres, shapes = [], [], []
    while len(centres) < n:
        batch = ap.sample_motion(rng, 128, n_frames=FRAMES, img_size=SIDE, p_static=0,
                                 shapes=FAMILIAR if fit else (*FAMILIAR, NOVEL))
        for movie, path, shape in zip(batch.frames, batch.centres, batch.shapes):
            if fit and inside_tile(path).any():
                continue
            ends = movie[[0, -1]].astype(bool)
            key = np.packbits(ends).tobytes()
            if key in seen:
                continue
            seen.add(key)
            endpoints.append(ends)
            centres.append(path)
            shapes.append(shape)
            if len(centres) == n:
                break
    return Trials(np.stack(endpoints), np.stack(centres), np.array(shapes))


@torch.inference_mode()
def extract(model, endpoints, *, patches=False):
    """Only the two flashed images enter the RNN. Never pass a target frame."""
    model.eval()
    states, detected = [], []
    for start in range(0, len(endpoints), 64):
        ends = endpoints[start:start + 64]
        x = torch.zeros(FRAMES, len(ends), SIDE**2)
        x[0], x[-1] = torch.from_numpy(ends.reshape(len(ends), 2, -1)).unbind(dim=1)
        if not patches:
            activity, _ = model.rnn(x)
        else:
            logits, activity = model(x)
            probs = logits[1:-1].sigmoid().transpose(0, 1).reshape(-1, FRAMES-2, SIDE, SIDE)
            xy = np.full((len(ends), FRAMES-2, 2), np.nan)
            for i, movie in enumerate(probs.numpy()):
                keep = ~ap._dilate(ends[i, 0], MARGIN)
                for t, image in enumerate(movie):
                    centre = ap._largest_patch(image * keep, THRESHOLD)
                    if centre is not None:
                        xy[i, t] = centre
            detected.append(xy)
        states.append(activity[-1].numpy())
    return np.concatenate(states), np.concatenate(detected) if detected else None


def score(test, predictions, patch_xy):
    truth = test.centres[:, 1:-1]
    novel = (test.shapes == NOVEL)[:, None]
    tile = inside_tile(truth)
    found = np.isfinite(patch_xy).all(axis=-1)
    errors = {k: np.linalg.norm(v - truth, axis=-1) for k, v in predictions.items()}
    errors["patch"] = np.linalg.norm(patch_xy - truth, axis=-1)
    scores = {}
    for label, sel in (("overall", np.ones_like(tile)), ("familiar", ~novel & ~tile),
                       ("shape", novel & ~tile), ("location", ~novel & tile),
                       ("both", novel & tile)):
        scores[label] = {}
        for group, mask in (("all", sel), ("found", sel & found), ("missing", sel & ~found)):
            cell = {"n_frames": int(mask.sum()), "n_trials": int(mask.any(axis=1).sum())}
            for name, err in errors.items():
                values = err[mask]
                valid = values[np.isfinite(values)]
                cell[name] = {
                    "mean_px": float(valid.mean()) if len(valid) else None,
                    # A missing patch counts as a miss, including in the 'all' denominator.
                    "within_2px": float((values <= 2).mean()) if len(values) else None,
                }
            scores[label][group] = cell
    return scores


def run_seed(seed, directory, fit, test):
    weights = torch.load(directory / f"held-out_seed{seed}.pt", map_location="cpu",
                         weights_only=True)
    torch.manual_seed(seed)
    model = ap.ImputationNet(SIDE**2, weights["fc.weight"].shape[1], FRAMES)
    truth = fit.centres[:, 1:-1]

    def probe(patches=False):
        x, _ = extract(model, fit.endpoints)
        test_x, xy = extract(model, test.endpoints, patches=patches)
        return ridge_predict(x, truth, test_x, alpha=ALPHA), xy

    # The random control is the same architecture at its original initialization.
    random, _ = probe()
    model.load_state_dict(weights)
    trained, patch_xy = probe(patches=True)
    a, b = test.centres[:, :1], test.centres[:, -1:]
    time = np.arange(1, FRAMES - 1)[None, :, None] / (FRAMES - 1)
    return score(test, {
        "probe": trained, "untrained": random, "constant": truth.mean(0),
        "endpoint_line": a + time * (b - a),
    }, patch_xy)


def plot(records, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.6]}, sharey=True)
    colours = {"probe": "#0072B2", "untrained": "#888888", "endpoint_line": "#D55E00"}
    labels = {"probe": "Trained RNN", "untrained": "Untrained RNN",
              "endpoint_line": "Straight line between flashes"}
    panels = [
        ([("overall", "found"), ("overall", "missing")],
         ["Detected", "Not detected"], "Was a non-start patch detected?"),
        ([(s, "missing") for s in ("familiar", "shape", "location", "both")],
         ["Neither", "Shape", "Location", "Both"], "No patch: what was held out of probe fitting?"),
    ]
    for ax, (cells, names, title) in zip(axes, panels):
        for offset, (key, colour) in zip((-.16, 0, .16), colours.items()):
            x = np.arange(len(cells)) + offset
            vals = [[r["scores"][s][g][key]["mean_px"] for s, g in cells] for r in records]
            for j, series in enumerate(np.array(vals, dtype=float).T):
                valid = series[np.isfinite(series)]
                if len(valid):
                    ax.vlines(x[j], valid.min(), valid.max(), color=colour)
                    ax.scatter(x[j], valid.mean(), color=colour, s=40,
                               label=labels[key] if not j else None)
        ax.set(xticks=np.arange(len(names)), xticklabels=names, title=title,
               xlim=(-.5, len(names)-.5), ylim=(0, None))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean position error (pixels, lower is better)")
    axes[0].legend(fontsize=8, loc="center", frameon=False)
    fig.suptitle(f"Reading position from the final state  ·  mean and range of {len(records)} seeds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fit-trials", type=int, default=4096)
    parser.add_argument("--test-trials", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/position_probe.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/position_probe.png"))
    args = parser.parse_args()
    if min(args.seeds, args.fit_trials, args.test_trials) < 1:
        parser.error("trial and seed counts must be positive")
    torch.set_num_threads(2)
    seen = set()
    fit = sample_trials(args.fit_trials, 41001, fit=True, seen=seen)
    test = sample_trials(args.test_trials, 41002, fit=False, seen=seen)
    records = []
    for seed in range(args.seeds):
        records.append({"seed": seed, "scores": run_seed(seed, args.checkpoints, fit, test)})
    config = {"checkpoint_condition": "held-out", "fit_trials": len(fit.shapes),
              "test_trials": len(test.shapes), "fit_seed": 41001, "test_seed": 41002,
              "novel_shape": NOVEL, "held_out_xy_interval": TILE, "ridge_alpha": ALPHA,
              "patch_threshold": THRESHOLD, "start_margin_px": MARGIN}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": config, "records": records}, indent=2,
                                   allow_nan=False) + "\n")
    plot(records, args.figure)


if __name__ == "__main__":
    main()
