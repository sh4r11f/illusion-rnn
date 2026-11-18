"""Test frozen models and fixed position readouts on matched stimuli of different sizes."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap, trajectories as tr
from illusion_rnn.analysis import fit_ridge
from scripts import run_prefix_forecast as fore
from scripts import run_trajectory_experiment as interp


STIMULI = ("square", "cross", "outline_square")
SIZES = (3, 7, 11)
INTERP_SEED, FORECAST_SEED = 81001, 81002
BOOTSTRAP_SEED = 81003


def geometry(n):
    """Integer ink-centre tracks [group, acceleration, path, frame, xy]."""
    largest = max(SIZES)
    sizes = (largest, largest)
    chords = tr.sample_chords(np.random.default_rng(INTERP_SEED), n, size_range=sizes)
    endpoints = np.array([[tr.trajectory(chords, a, p) for p in tr.PATH] for a in tr.ACCEL])
    prefix, _, _ = tr.forecast_tracks(np.random.default_rng(FORECAST_SEED), n, size_range=sizes)
    return tuple(np.rint(x).astype(int) + largest//2
                 for x in (endpoints.transpose(2, 0, 1, 3, 4), prefix))


def render(centres, stimulus, size):
    if stimulus not in STIMULI or size not in SIZES:
        raise ValueError("unknown stimulus or size")
    tracks = centres.reshape(-1, tr.FRAMES, 2) - size//2
    outline = stimulus == "outline_square"
    return tr.render_tracks(tracks, np.full(len(tracks), size),
                            np.repeat("square" if outline else stimulus, len(tracks)),
                            outline=outline)


def prefixes(centres, stimulus, size):
    batch = render(centres, stimulus, size)
    return fore.Movies(batch.frames[:, :fore.P].astype(bool).reshape(
        *centres.shape[:-2], fore.P, tr.SIDE, tr.SIDE), centres)


def excluded_pairs(endpoints, forecasts):
    """No probe-fitting movie may repeat a test endpoint pair / last visible pair."""
    ep, last = set(), set()
    for stimulus, size in product(STIMULI, SIZES):
        ep.update(interp.endpoint_keys(render(endpoints[:, 0, 0], stimulus, size)))
        # The three acceleration variants share their last two visible frames.
        data = prefixes(forecasts[:, :1], stimulus, size)
        last.update(k for row in fore.keys(data) for k in row)
    return ep, last


@torch.inference_mode()
def classic_scores(model, centres, stimulus, size):
    batch = render(centres, stimulus, size)
    x = interp.inputs(batch, False)
    logits = torch.cat([model(chunk)[0].sigmoid() for chunk in x.split(64, 1)], dim=1)
    images = logits.transpose(0, 1).reshape(batch.frames.shape).numpy()
    return ap.patch_position_metrics(images, batch, ap.endpoint_mask(len(images), tr.FRAMES))


@torch.inference_mode()
def cue_scores(model, probe, centres, stimulus, size):
    truth = centres.transpose(1, 2, 0, 3, 4)[..., interp.SCORE, :]
    out = {}
    for cue in (False, True):
        predictions, lines = [], []
        for a, p in product(range(len(tr.ACCEL)), range(len(tr.PATH))):
            if not cue and predictions:
                predictions.append(predictions[0])
                lines.append(lines[0])
                continue
            batch = render(centres[:, a, p], stimulus, size)
            x = interp.inputs(batch, cue)
            states = torch.cat([model.rnn(chunk)[0][-1] for chunk in x.split(64, 1)])
            predictions.append(probe(states.numpy()))
            lines.append(interp.visible_line(batch.centres, cue))
        out["plus_cue" if cue else "endpoints"] = {
            k: interp.factor_scores(np.reshape(v, truth.shape), truth)
            for k, v in (("probe", predictions), ("visible_line", lines))}
    return out


def forecast_scores(model, probe, centres, stimulus, size):
    visible = prefixes(centres, stimulus, size).prefix.reshape(-1, fore.P, tr.SIDE, tr.SIDE)
    states, ink, patch = fore.extract(model, visible, True)
    shape = (*centres.shape[:-2], fore.H, 2)
    preds = {**fore.kinematics(centres[..., :fore.P, :]), "probe": probe(states).reshape(shape),
             "ink_centre": ink.reshape(shape), "patch": patch.reshape(shape)}
    scores = {k: {p: horizons[-1] for p, horizons in fore.scores(v, centres).items()}
              for k, v in preds.items()}
    # One observation per matched group, not three independent acceleration trials.
    def straight_error(x):
        return np.linalg.norm(x[:, :, 0, -1] - centres[:, :, 0, -1], axis=-1).mean(axis=1)
    return scores, straight_error(preds["ink_centre"]) - straight_error(preds["last_velocity"])


def paired_intervals(differences, draws=4000):
    """Seed-averaged ink minus velocity error, resampling test groups, not RNNs."""
    delta = np.mean(differences, axis=0)  # condition, group
    n = delta.shape[1]
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(draws, n))
    lo, hi = np.quantile(delta[:, indices].mean(-1), [.025, .975], axis=1)
    return [{"stimulus": s, "size": z, "mean_px": float(row.mean()),
             "interval_95_px": [float(l), float(h)]}
            for (s, z), row, l, h in zip(product(STIMULI, SIZES), delta, lo, hi)]


def evaluate_seed(base, model, endpoints, forecasts, ep_exclude, fit_prefix, n_fit):
    base.eval()
    model.eval()
    cue_probe = fit_ridge(*interp.fit_examples(model, n_fit, ep_exclude))
    states, _, _ = fore.extract(model, fit_prefix.prefix.reshape(-1, fore.P, tr.SIDE, tr.SIDE),
                                False)
    future_probe = fit_ridge(states, fit_prefix.xy[..., fore.P:, :].reshape(-1, fore.H, 2))
    records, deltas = [], []
    for stimulus, size in product(STIMULI, SIZES):
        classic = classic_scores(base, endpoints[:, 0, 0], stimulus, size)
        cue = cue_scores(model, cue_probe, endpoints, stimulus, size)
        future, delta = forecast_scores(model, future_probe, forecasts, stimulus, size)
        deltas.append(delta)
        records.append({"stimulus": stimulus, "size": size, "straight_fill_in": classic,
                        "trajectory_choice": cue, "last_hidden_frame": future})
    return records, deltas


def plot(result, dest):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    panels = (
        ("Endpoints only", "Blank frames with a patch within 2 px (%)",
         lambda r: r["straight_fill_in"]["patch_within_2px_all"]),
        ("Endpoints + one flash", "Correct acceleration choice (%)",
         lambda r: r["trajectory_choice"]["plus_cue"]["probe"]["accel_choice"]),
        ("Endpoint hidden", "Final fast–slow separation recovered (%)",
         lambda r: r["last_hidden_frame"]["ink_centre"]["straight"]["separation_gain"]),
    )
    for ax, (title, label, metric) in zip(axes, panels):
        for s, color, marker in zip(STIMULI, ("#0072B2", "#D55E00", "#009E73"), ("s", "P", "s")):
            data = 100*np.array([[metric(r) for r in run["conditions"] if r["stimulus"] == s]
                                 for run in result["runs"]])
            ax.plot(SIZES, data.mean(0), color=color, marker=marker,
                    markerfacecolor="white" if s == "outline_square" else color,
                    label=s.replace("_", " "))
            ax.fill_between(SIZES, data.min(0), data.max(0), color=color, alpha=.13)
        ax.set(title=title, ylabel=label, xlabel="Stimulus width (px)", xticks=SIZES,
               ylim=(min(-5, ax.dataLim.ymin-3), max(105, ax.dataLim.ymax+3)))
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(100/3, color=".45", ls=":", label="Chance")
    axes[2].axhline(100, color=".45", ls=":", label="True separation")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="straight-motion checkpoints")
    parser.add_argument("--checkpoints", type=Path, required=True, help="adapted checkpoints")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--test-groups", type=int, default=192)
    parser.add_argument("--cue-fit-movies", type=int, default=4096)
    parser.add_argument("--forecast-fit-groups", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("results/stimulus_robustness.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/stimulus_robustness.png"))
    args = parser.parse_args()
    if min(args.seeds, args.test_groups, args.cue_fit_movies, args.forecast_fit_groups) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(2)
    endpoints, forecasts = geometry(args.test_groups)
    ep_exclude, last_exclude = excluded_pairs(endpoints, forecasts)
    fit_prefix = fore.dataset(args.forecast_fit_groups, fore.FIT_SEED, last_exclude)
    runs, deltas = [], []
    result = {"stimuli": STIMULI, "sizes_px": SIZES, "test_groups_per_task": args.test_groups,
              "acceleration": tr.ACCEL, "paths": tr.PATH, "cue_fit_movies": args.cue_fit_movies,
              "forecast_fit_groups": args.forecast_fit_groups, "interpolation_test_seed": INTERP_SEED,
              "forecast_test_seed": FORECAST_SEED, "cue_fit_seed": interp.FIT_SEED,
              "forecast_fit_seed": fore.FIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "runs": runs}
    for seed in range(args.seeds):
        base = interp.load(torch.load(args.base / f"held-out_seed{seed}.pt",
                                      map_location="cpu", weights_only=True))
        adapted = torch.load(args.checkpoints / f"trajectory_seed{seed}.pt",
                             map_location="cpu", weights_only=True)
        rows, diff = evaluate_seed(base, interp.load(adapted["model"]), endpoints, forecasts,
                                   ep_exclude, fit_prefix, args.cue_fit_movies)
        runs.append({"seed": seed, "conditions": rows})
        deltas.append(diff)
        print(f"Finished seed {seed}", flush=True)
    result["straight_final_ink_minus_velocity"] = paired_intervals(deltas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.figure)


if __name__ == "__main__":
    main()
