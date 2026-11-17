"""Forecast a blank suffix without an endpoint; no updates to recurrent or pixel weights."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn import trajectories as tr
from illusion_rnn.analysis import ridge_predict

P, H = tr.PREFIX, tr.FRAMES - tr.PREFIX
FIT_SEED, TEST_SEED = 71001, 71002


@dataclass
class Movies:
    prefix: np.ndarray  # group, accel, path, visible time, y, x
    xy: np.ndarray      # group, accel, path, full time, coordinate


def sample(rng, n):
    tracks, sizes, shapes = tr.forecast_tracks(rng, n)
    batch = tr.render_tracks(tracks.reshape(-1, tr.FRAMES, 2),
                             np.repeat(sizes, 9), np.repeat(shapes, 9))
    return Movies(batch.frames[:, :P].astype(bool).reshape(n, 3, 3, P, tr.SIDE, tr.SIDE),
                  batch.centres.reshape(n, 3, 3, tr.FRAMES, 2))


def keys(data):
    # Acceleration conditions within a path have identical last-two-flash pixels.
    pairs = data.prefix[:, 0, :, -2:].reshape(len(data.xy), 3, -1)
    return [[p.tobytes() for p in group] for group in np.packbits(pairs, axis=-1)]


def dataset(n, seed, exclude=()):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for _ in range(4*n):
        m = sample(rng, min(64, n))
        keep = [i for i, ks in enumerate(keys(m)) if not any(k in exclude for k in ks)][:n]
        if keep:
            parts.append(m.prefix[keep])
            labels.append(m.xy[keep])
            n -= len(keep)
        if n == 0:
            return Movies(np.concatenate(parts), np.concatenate(labels))
    raise ValueError("could not draw enough groups distinct from the excluded flash pairs")


def masked_input(prefix):
    x = torch.zeros((tr.FRAMES, len(prefix), tr.SIDE**2))
    x[:P] = torch.from_numpy(prefix.reshape(len(prefix), P, -1)).transpose(0, 1)
    return x


def kinematics(visible):
    """Only the four rounded, visible ink centroids enter these predictors."""
    h = np.arange(1, H+1)[:, None]
    last = visible[..., -1:, :]
    v = last - visible[..., -2:-1, :]
    t = np.arange(1-P, 1)
    weights = np.vander(h[:, 0], 3) @ np.linalg.pinv(np.vander(t, 3))
    return {"last_velocity": last + h*v, "prefix_quadratic": weights @ visible}


@torch.inference_mode()
def extract(model, prefixes, decode):
    states, pixels, patches = [], [], []
    for start in range(0, len(prefixes), 64):
        chunk = prefixes[start:start+64]
        x = masked_input(chunk)
        if decode:
            logits, activity = model(x)
            probs = logits[P:].sigmoid().transpose(0, 1).reshape(-1, H, tr.SIDE, tr.SIDE).numpy()
            yy, xx = np.mgrid[:tr.SIDE, :tr.SIDE]
            mass = probs.sum((-2, -1))
            pixels.append(np.stack([(probs * grid).sum((-2, -1)) / mass.clip(min=1e-20)
                                    for grid in (xx, yy)], axis=-1))
            xy = np.full((len(probs), H, 2), np.nan)
            for i, movie in enumerate(probs):
                keep = ~ap._dilate(chunk[i, 0], 1)
                for t, image in enumerate(movie):
                    point = ap._largest_patch(image * keep, .25)
                    if point is not None:
                        xy[i, t] = point
            patches.append(xy)
        else:
            activity, _ = model.rnn(x)
        states.append(activity[-1].numpy())
    return (np.concatenate(states), np.concatenate(pixels) if decode else None,
            np.concatenate(patches) if decode else None)


def mean_valid(a):
    valid = a[np.isfinite(a)]
    return float(valid.mean()) if valid.size else None


def scores(pred, xy):
    """Paired acceleration response: 0 for identical forecasts, 1 for true separation."""
    truth = xy[..., P:, :]
    cv = kinematics(xy[..., :P, :])["last_velocity"]
    result = {}
    for j, path in enumerate(tr.PATH):
        y, p = truth[:, :, j], pred[:, :, j]  # group, acceleration, horizon, xy
        err = np.linalg.norm(p - y, axis=-1)
        expected = y[:, 1] - y[:, 2]
        observed = p[:, 1] - p[:, 2]
        horizons = []
        for h in range(H):
            d, q = expected[:, h], observed[:, h]
            both = np.isfinite(q).all(1)
            denom = np.square(d[both]).sum()
            # In each speed-changing movie, compare with its matched steady continuation.
            real = err[:, 1:, h]

            def preference(other):
                alt = np.linalg.norm(p[:, 1:, h] - other, axis=-1)
                return mean_valid(np.where(np.isfinite(real), (real < alt) + .5*(real == alt),
                                           np.nan))
            horizons.append({
                "mean_error_px": mean_valid(err[:, :, h]),
                "error_px_by_accel": [mean_valid(e) for e in err[:, :, h].T],
                "within_2px_all": float((err[:, :, h] <= 2).mean()),
                "coverage": float(np.isfinite(err[:, :, h]).mean()),
                "pair_coverage": float(both.mean()),
                "separation_gain": float((q[both]*d[both]).sum()/denom) if denom else None,
                "closer_to_true_than_steady": preference(y[:, :1, h]),
                "closer_to_true_than_last_velocity": preference(cv[:, 1:, j, h]),
            })
        result[path] = horizons
    return result


def network_scores(model, fit, test, decode=True):
    def flat(m):
        return m.prefix.reshape(-1, P, tr.SIDE, tr.SIDE)
    model.eval()
    x, _, _ = extract(model, flat(fit), False)
    xx, ink, patch = extract(model, flat(test), decode)
    truth = test.xy[..., P:, :]
    pred = ridge_predict(x, fit.xy[..., P:, :].reshape(-1, H, 2), xx)
    out = {"probe": scores(pred.reshape(truth.shape), test.xy)}
    if decode:
        out["ink_centre"] = scores(ink.reshape(truth.shape), test.xy)
        out["patch"] = scores(patch.reshape(truth.shape), test.xy)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="adapted trajectory checkpoints")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fit-groups", type=int, default=512)
    parser.add_argument("--test-groups", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("results/prefix_forecast.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/prefix_forecast.png"))
    args = parser.parse_args()
    if min(args.seeds, args.fit_groups, args.test_groups) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(2)
    test = dataset(args.test_groups, TEST_SEED)
    fit = dataset(args.fit_groups, FIT_SEED, {k for row in keys(test) for k in row})
    baselines = {k: scores(pred, test.xy) for k, pred in kinematics(test.xy[..., :P, :]).items()}
    records = []
    for seed in range(args.seeds):
        weights = torch.load(args.checkpoints / f"trajectory_seed{seed}.pt",
                             weights_only=True, map_location="cpu")["model"]
        torch.manual_seed(seed)
        model = ap.ImputationNet(tr.SIDE**2, weights["fc.weight"].shape[1], tr.FRAMES)
        random = network_scores(model, fit, test, decode=False)
        model.load_state_dict(weights)
        records.append({"seed": seed, "untrained_probe": random["probe"],
                        **network_scores(model, fit, test)})
    result = {"fit_groups": args.fit_groups, "test_groups": args.test_groups,
              "fit_seed": FIT_SEED, "test_seed": TEST_SEED, "acceleration": tr.ACCEL,
              "prefix_frames": P, "blank_frames": H, "baselines": baselines, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(test, result, args.figure)


def plot(test, result, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4), layout="constrained")
    ax = axes[0]
    example = test.xy[0, :, 0]
    step = example[0, P-1] - example[0, P-2]
    if not step.any():
        step = example[0, -1] - example[0, 0]
    disp = (example - example[0, P-1]) @ (step / np.linalg.norm(step))
    for d, label, color in zip(disp, ("Steady", "Speeds up", "Slows down"),
                               ("#777777", "#0072B2", "#D55E00")):
        ax.plot(range(1, P+1), d[:P], color=color, marker="o", ms=3, label=label)
        ax.plot(range(P, tr.FRAMES+1), d[P-1:], color=color, ls="--", marker="o", ms=3)
    ax.axvspan(P+.5, tr.FRAMES+.25, color="#eeeeee", zorder=-1)
    ax.set(xlabel="Frame (shaded frames are hidden)", ylabel="Position along motion (px)",
           title="One matched set · straight path", xticks=range(1, tr.FRAMES+1))
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for key, name, col in (("ink_centre", "RNN image centre", "#009E73"),
                           ("probe", "Position probe", "#0072B2"),
                           ("last_velocity", "Last velocity", "#555555"),
                           ("prefix_quadratic", "Quadratic fit", "#D55E00")):
        runs = ([result["baselines"][key]] if key in result["baselines"]
                else [r[key] for r in result["records"]])
        vals = np.array([[m["separation_gain"] for m in r["straight"]] for r in runs])
        h = np.arange(1, H+1)
        ax.plot(h, vals.mean(0), "o-", ms=4, label=name, color=col)
        ax.fill_between(h, vals.min(0), vals.max(0), color=col, alpha=.16)
    ax.axhline(1, color="#999999", ls=":", label="True separation")
    ax.set(xlabel="Frames after last flash", ylabel="Fraction of true separation",
           xticks=range(1, H+1), title="Does the forecast separate fast from slow?")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
