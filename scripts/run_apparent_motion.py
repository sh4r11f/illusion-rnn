"""Train imputation models on masked real motion, test on apparent motion.

Writes results/apparent_motion.json (metrics), results/apparent_motion.npz
(example reconstructions) and figures/apparent_motion.png.

    uv run python scripts/run_apparent_motion.py --quick     # smoke run
    uv run python scripts/run_apparent_motion.py             # ~1h on 8 CPU cores
    uv run python scripts/run_apparent_motion.py --plot-only
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap

N_FRAMES = 8
IMG_SIZE = 32

# name -> (readout, longest run of blanked frames seen in training)
CONDITIONS = {
    "held-out": ("final", N_FRAMES - 4),
    "trained-on": ("final", None),
    "online": ("online", N_FRAMES - 4),
}
LABELS = {
    "held-out": f"after-sequence readout, training gaps <= {N_FRAMES - 4}",
    "trained-on": "after-sequence readout, training gaps unbounded",
    "online": f"online readout, training gaps <= {N_FRAMES - 4}",
}


def run_one(job):
    condition, seed, n_steps, hidden, n_trials, save_dir, eval_only, prior_loss = job
    readout, max_gap = CONDITIONS[condition]
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = ap.ImputationNet(IMG_SIZE**2, hidden, N_FRAMES, readout=readout)
    checkpoint = Path(save_dir) / f"{condition}_seed{seed}.pt" if save_dir else None
    if eval_only:
        model.load_state_dict(torch.load(checkpoint))
        final_loss = prior_loss
    else:
        history = ap.train_imputation(model, n_steps=n_steps, max_gap=max_gap, seed=seed,
                                      device="cpu", log_every=0, img_size=IMG_SIZE)
        final_loss = float(np.mean(history["loss"][-100:]))
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
    kwargs = {"n_trials": n_trials, "device": "cpu", "img_size": IMG_SIZE,
              "seed": 10_000 + seed}
    record = {
        "condition": condition, "readout": readout, "max_gap": max_gap, "seed": seed,
        "final_loss": final_loss,
        "gap": {str(g): ap.evaluate_imputation(model, "gap", gap=g, **kwargs)
                for g in range(1, N_FRAMES - 1)},
        "flicker": ap.evaluate_imputation(model, "flicker", **kwargs),
    }
    examples = None
    if seed == 0:
        batch = ap.sample_motion(np.random.default_rng(123), 3, n_frames=N_FRAMES,
                                 img_size=IMG_SIZE, p_static=0.0)
        visible = ap.endpoint_mask(3, N_FRAMES)
        examples = (batch.frames, ap.reconstruct(model, batch.frames, visible, device="cpu"))
    print(f"done {condition} seed {seed}: apparent-motion iou_true "
          f"{record['gap'][str(N_FRAMES - 2)]['iou_true']:.3f}", flush=True)
    return record, examples


def plot(results_path: Path, figure_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = json.loads(results_path.read_text())["records"]
    examples = np.load(results_path.with_suffix(".npz"))
    gaps = list(range(1, N_FRAMES - 1))

    fig = plt.figure(figsize=(18, 4.6))
    grid = fig.add_gridspec(3, N_FRAMES + 14, wspace=0.08, hspace=0.08)
    truth = examples["truth"][0]
    rows = [
        ("shown", ap.apply_mask(truth[None], ap.endpoint_mask(1, N_FRAMES))[0]),
        ("real motion", truth),
        ("reconstruction", examples["held-out"][0]),
    ]
    for r, (name, movie) in enumerate(rows):
        for t in range(N_FRAMES):
            ax = fig.add_subplot(grid[r, t])
            ax.imshow(movie[t], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(name, fontsize=8)
            if r == 0:
                ax.set_title(f"t{t + 1}", fontsize=8)

    ax = fig.add_subplot(grid[:, N_FRAMES + 2:N_FRAMES + 7])
    for condition in CONDITIONS:
        runs = [rec for rec in records if rec["condition"] == condition]
        if not runs:
            continue
        values = np.array([[run["gap"][str(g)]["iou_true"] for g in gaps] for run in runs])
        line, = ax.plot(gaps, values.mean(axis=0), marker="o", label=LABELS[condition])
        ax.fill_between(gaps, values.min(axis=0), values.max(axis=0),
                        color=line.get_color(), alpha=0.2)
    jump = [rec["gap"][str(g)]["iou_nearest"] for rec in records
            if rec["condition"] == "held-out" for g in [gaps[-1]]]
    ax.axvline(N_FRAMES - 2, color="k", ls=":", lw=1)
    ax.text(N_FRAMES - 2.05, 0.93, "apparent\nmotion", ha="right", va="top", fontsize=8)
    ax.set_xlabel("consecutive blanked frames at test")
    ax.set_ylabel("IoU with true intermediate frame")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"mean and range over seeds (n={len(jump)})", fontsize=9)
    ax = fig.add_subplot(grid[:, N_FRAMES + 9:])
    frames = [str(t) for t in range(1, N_FRAMES - 1)]
    xs = [int(t) + 1 for t in frames]
    apparent = str(N_FRAMES - 2)

    def per_frame(condition, key):
        runs = [rec["gap"][apparent]["position"]["by_frame"] for rec in records
                if rec["condition"] == condition]
        return np.array([[np.nan if run[t][key] is None else run[t][key] for t in frames]
                         for run in runs], dtype=float)

    for condition in CONDITIONS:
        values = per_frame(condition, "patch_err")
        if not len(values):
            continue
        line, = ax.plot(xs, np.nanmean(values, axis=0), marker="o", label=LABELS[condition])
        ax.fill_between(xs, np.nanmin(values, axis=0), np.nanmax(values, axis=0),
                        color=line.get_color(), alpha=0.2)
    ax.plot(xs, np.nanmean(per_frame("held-out", "jump_err"), axis=0), "k--", lw=1,
            label="patch at the nearer endpoint")
    ax.plot(xs, np.nanmean(per_frame("held-out", "truth_err"), axis=0), "k:", lw=1,
            label="true movie (best attainable)")
    ax.set_xticks(xs, [f"t{x}" for x in xs])
    ax.set_xlabel("blanked frame (apparent-motion test)")
    ax.set_ylabel("patch centre error (px)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.set_title("position of the drawn patch, start-position ink discarded", fontsize=9)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"wrote {figure_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir", default=None, help="write model state dicts here")
    parser.add_argument("--eval-only", action="store_true",
                        help="re-evaluate the models in --save-dir without retraining")
    parser.add_argument("--out", default="results/apparent_motion.json")
    parser.add_argument("--figure", default="figures/apparent_motion.png")
    args = parser.parse_args()
    out = Path(args.out)

    if not args.plot_only:
        steps, hidden, seeds, n_trials = args.steps, args.hidden, args.seeds, 500
        if args.quick:
            steps, hidden, seeds, n_trials = 300, 64, 1, 50
        prior = {}
        if args.eval_only:
            if not args.save_dir:
                parser.error("--eval-only needs --save-dir")
            # Training metadata is not in the checkpoints: carry it over from an
            # earlier results file when there is one, otherwise leave it unknown.
            steps = None
            if out.exists():
                earlier = json.loads(out.read_text())
                steps = earlier["config"].get("n_steps")
                prior = {(r["condition"], r["seed"]): r["final_loss"]
                         for r in earlier["records"]}
        jobs = [(c, s, steps, hidden, n_trials, args.save_dir, args.eval_only,
                 prior.get((c, s)))
                for c in CONDITIONS for s in range(seeds)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            done = list(pool.map(run_one, jobs))
        records = [record for record, _ in done]
        arrays = {record["condition"]: ex[1] for record, ex in done if ex is not None}
        arrays["truth"] = next(ex[0] for _, ex in done if ex is not None)

        out.parent.mkdir(parents=True, exist_ok=True)
        config = {"n_frames": N_FRAMES, "img_size": IMG_SIZE, "n_steps": steps,
                  "hidden_size": hidden, "seeds": seeds, "n_eval_trials": n_trials,
                  "cell": "ctrnn"}
        out.write_text(json.dumps({"config": config, "records": records}, indent=2,
                                  allow_nan=False))
        np.savez_compressed(out.with_suffix(".npz"), **arrays)
        print(f"wrote {out}")

    plot(out, Path(args.figure))


if __name__ == "__main__":
    main()
