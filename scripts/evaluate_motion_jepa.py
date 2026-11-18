"""Score a JEPA's predicted embeddings and shared context on the matched motion test."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import trajectories as tr
from illusion_rnn.analysis import fit_ridge
from illusion_rnn.jepa import MotionJEPA
from scripts import run_prefix_forecast as fore, run_stimulus_robustness as suite
from scripts import run_trajectory_experiment as interp
from scripts.train_motion_jepa import MODELS

END = (0, tr.FRAMES-1)
CUED = (0, interp.CUE, tr.FRAMES-1)
PREFIX = tuple(range(fore.P))
DIAG_SEED = 91001


@torch.inference_mode()
def features(model, movies, shown):
    """Copy only visible images; a prefix need not even contain the future frames."""
    h, z = [], []
    for i in range(0, len(movies), 128):
        batch = movies[i:i+128]
        frames = torch.zeros(len(batch), tr.FRAMES, tr.SIDE, tr.SIDE)
        visible = torch.zeros(len(batch), tr.FRAMES, dtype=torch.bool)
        frames[:, shown] = torch.from_numpy(batch[:, shown]).float()
        visible[:, shown] = True
        context, predicted, _ = model(frames, visible)
        h.append(context.numpy())
        z.append(predicted.numpy())
    return np.concatenate(h), np.concatenate(z)


@torch.inference_mode()
def fit_probes(model, n_fit, excluded, prefix_fit):
    contexts, targets, static_x, static_y = [], [], [], []
    for batch, take in interp.fitting_batches(n_fit, excluded):
        movies = batch.frames[take]
        for shown in (END, CUED):
            contexts.append(features(model, movies, shown)[0])
            targets.append(batch.centres[take][:, interp.SCORE])
        if isinstance(model, MotionJEPA):
            # Calibrate a point in a *single image*, never a trajectory or a future label.
            static_x.append(model.target(torch.from_numpy(movies)).flatten(0, 1).numpy())
            static_y.append(batch.centres[take].reshape(-1, 2))
    cue = fit_ridge(np.concatenate(contexts), np.concatenate(targets))
    h, _ = features(model, prefix_fit.prefix.reshape(-1, fore.P, tr.SIDE, tr.SIDE), PREFIX)
    future = fit_ridge(h, prefix_fit.xy[..., fore.P:, :].reshape(-1, fore.H, 2))
    static = fit_ridge(np.concatenate(static_x), np.concatenate(static_y)) if static_x else None
    return cue, future, static


def decode_frames(probe, z):
    return probe(z.reshape(-1, z.shape[-1])).reshape(*z.shape[:-1], 2)


@torch.inference_mode()
def native_position(model, z, static):
    if static is not None:
        return "predicted_embedding", decode_frames(static, z)
    axis = torch.arange(tr.SIDE)
    parts = []
    for chunk in torch.from_numpy(z).split(128):
        images = model.pixels(chunk).sigmoid().unflatten(-1, (tr.SIDE, tr.SIDE))
        mass = images.sum((-2, -1)).clamp_min(1e-20)
        xy = torch.stack([(images.sum(k)*axis).sum(-1)/mass for k in (-2, -1)], dim=-1)
        parts.append(xy.numpy())
    return "ink_centre", np.concatenate(parts)


def interpolation(model, tracks, stimulus, size, cue_probe, static):
    # Uncued movies for a group have identical pixels, so evaluate them only once.
    truth = tracks.transpose(1, 2, 0, 3, 4)[..., interp.SCORE, :]
    out = {}
    for name, shown in (("endpoints", END), ("plus_cue", CUED)):
        observed = tracks if shown == CUED else tracks[:, :1, :1]
        batch = suite.render(observed, stimulus, size)
        h, z = features(model, batch.frames, shown)
        preds = {"context_probe": cue_probe(h),
                 "visible_line": interp.visible_line(batch.centres, shown == CUED)}
        native, pos = native_position(model, z[:, interp.SCORE], static)
        preds[native] = pos
        out[name] = {}
        for key, xy in preds.items():
            xy = xy.reshape(*observed.shape[:3], len(interp.SCORE), 2).transpose(1, 2, 0, 3, 4)
            xy = np.broadcast_to(xy, truth.shape)
            s = interp.factor_scores(xy, truth)
            # The steady, straight subset is the classic apparent-motion condition.
            s["straight_steady_error_px"] = float(np.linalg.norm(xy[0, 0]-truth[0, 0], axis=-1).mean())
            out[name][key] = s
    return out


def forecast(model, tracks, stimulus, size, future_probe, static):
    prefix = suite.prefixes(tracks, stimulus, size).prefix.reshape(-1, fore.P, tr.SIDE, tr.SIDE)
    h, z = features(model, prefix, PREFIX)
    shape = (*tracks.shape[:3], fore.H, 2)
    preds = {"context_probe": future_probe(h).reshape(shape),
             **fore.kinematics(tracks[..., :fore.P, :])}
    native, pos = native_position(model, z[:, fore.P:], static)
    preds[native] = pos.reshape(shape)
    return {k: {path: [{m: r[m] for m in ("mean_error_px", "separation_gain", "within_2px_all")}
                       for r in horizons] for path, horizons in fore.scores(v, tracks).items()}
            for k, v in preds.items()}


@torch.inference_mode()
def diagnostics(model, static):
    """Held-out familiar movies: check targets vary and matching beats an unrelated movie."""
    batch = tr.sample_trajectories(np.random.default_rng(DIAG_SEED), 512)
    truth = model.target(torch.from_numpy(batch.frames)).numpy()
    eig = np.linalg.eigvalsh(np.cov(truth.reshape(-1, truth.shape[-1]).T)).clip(min=0)
    report = {"target_mean_channel_sd": float(truth.std(axis=(0, 1)).mean()),
              "target_participation_rank": float(eig.sum()**2 / max(np.square(eig).sum(), 1e-15)),
              "true_frame_position_error_px": float(np.linalg.norm(
                  decode_frames(static, truth)-batch.centres, axis=-1).mean())}
    for key, shown, score in (("plus_cue", CUED, interp.SCORE),
                             ("prefix", PREFIX, list(range(fore.P, tr.FRAMES)))):
        _, z = features(model, batch.frames, shown)
        target = truth[:, score]
        report[key] = {"matched_mse": float(np.square(z[:, score] - target).mean()),
                       "shuffled_mse": float(np.square(z[:, score] - np.roll(target, 1, 0)).mean())}
    return report


@torch.inference_mode()
def true_frame_error(model, static, tracks, stimulus, size):
    """Calibration check with the object actually present, not a forecast."""
    batch = suite.render(tracks[:, 0, 0], stimulus, size)
    target = model.target(torch.from_numpy(batch.frames)).numpy()
    return float(np.linalg.norm(decode_frames(static, target)-batch.centres, axis=-1).mean())


def score(model, endpoints, forecasts, n_fit, excluded, fit_prefix):
    model.eval()
    cue, future, static = fit_probes(model, n_fit, excluded, fit_prefix)
    rows = [{"stimulus": s, "size": size,
             "true_frame_position_error_px": (true_frame_error(model, static, endpoints, s, size)
                                              if static else None),
             "interpolation": interpolation(model, endpoints, s, size, cue, static),
             "forecast": forecast(model, forecasts, s, size, future, static)}
            for s, size in product(suite.STIMULI, suite.SIZES)]
    return {"conditions": rows, "diagnostics": diagnostics(model, static) if static else None}


def plot(runs, dest):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    series = (("jepa", "predicted_embedding", "JEPA · frame readout", "#0072B2", "o"),
              ("jepa", "context_probe", "JEPA · context probe", "#009E73", "s"),
              ("pixel", "ink_centre", "Pixel control · ink centre", "#6F4C9B", "D"),
              ("pixel", "context_probe", "Pixel control · context probe", "#D55E00", "^"))
    metrics = (("Extra flash: acceleration choice", "% correct", 100,
                lambda r, k: r["interpolation"]["plus_cue"][k]["accel_choice"]),
               ("No endpoint: fast–slow separation", "% of true separation", 100,
                lambda r, k: r["forecast"][k]["straight"][-1]["separation_gain"]),
               ("No endpoint: position error", "Distance (px)", 1,
                lambda r, k: r["forecast"][k]["straight"][-1]["mean_error_px"]))
    for ax, (title, ylabel, scale, metric) in zip(axes, metrics):
        for i, (model, key, label, colour, marker) in enumerate(series):
            vals = scale*np.array([[metric(r, key) for r in run["conditions"]]
                                   for run in runs if run["objective"] == model])
            x = np.arange(vals.shape[1]) + (i-1.5)*.16
            ax.vlines(x, vals.min(0), vals.max(0), color=colour)
            ax.scatter(x, vals.mean(0), color=colour, marker=marker, s=25, label=label)
        ax.set(title=title, ylabel=ylabel, xticks=range(9), xticklabels=list(suite.SIZES)*3,
               xlabel="Square           Cross           Hollow square\nWidth (px)")
        for x0 in (2.5, 5.5):
            ax.axvline(x0, color=".85", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
    controls = [axes[0].axhline(100/3, color=".45", ls=":", label="Chance"),
                axes[1].axhline(100, color=".45", ls=":", label="True separation")]
    velocity = runs[0]["conditions"][0]["forecast"]["last_velocity"]["straight"][-1]
    controls.append(axes[2].axhline(velocity["mean_error_px"], color=".45",
                                    ls=":", label="Last velocity"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(series)], labels[:len(series)], loc="outside lower center",
               ncols=4, frameon=False, fontsize=8)
    for ax, line in zip(axes, controls):
        ax.legend(handles=[line], loc="center left", fontsize=8, frameon=True, framealpha=.85)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cue-fit-movies", type=int, default=4096)
    parser.add_argument("--forecast-fit-groups", type=int, default=512)
    parser.add_argument("--test-groups", type=int, default=192)
    parser.add_argument("--out", type=Path, default=Path("results/motion_jepa.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/motion_jepa.png"))
    args = parser.parse_args()
    if min(args.seeds, args.cue_fit_movies, args.forecast_fit_groups, args.test_groups) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(2)
    endpoints, forecasts = suite.geometry(args.test_groups)
    excluded, last = suite.excluded_pairs(endpoints, forecasts)
    prefix_fit = fore.dataset(args.forecast_fit_groups, fore.FIT_SEED, last)
    runs = []
    for name, seed in product(MODELS, range(args.seeds)):
        model = MODELS[name]()
        ckpt = torch.load(args.checkpoints / f"{name}_seed{seed}.pt", weights_only=True,
                          map_location="cpu")
        model.load_state_dict(ckpt.pop("model"))
        runs.append({**ckpt, **score(model, endpoints, forecasts, args.cue_fit_movies,
                                     excluded, prefix_fit)})
        print(f"Scored {name} {seed}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cue_fit_movies": args.cue_fit_movies,
                                    "forecast_fit_groups": args.forecast_fit_groups,
                                    "test_groups": args.test_groups,
                                    "interp_test_seed": suite.INTERP_SEED,
                                    "forecast_test_seed": suite.FORECAST_SEED,
                                    "diagnostic_seed": DIAG_SEED, "runs": runs},
                                   indent=2, allow_nan=False) + "\n")
    plot(runs, args.figure)


if __name__ == "__main__":
    main()
