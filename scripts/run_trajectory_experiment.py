"""Fine-tune the straight-motion checkpoints, then test matched trajectory alternatives."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn import trajectories as tr
from illusion_rnn.analysis import ridge_predict


CUE = 3  # fourth frame; exclude it from scoring even when it is not shown
SCORE = [t for t in range(1, tr.FRAMES-1) if t != CUE]
LR = 3e-4
BATCH_SIZE = 32
FIT_SEED, TEST_SEED = 61001, 61002


def load(weights):
    model = ap.ImputationNet(tr.SIDE**2, weights["fc.weight"].shape[1], tr.FRAMES)
    model.load_state_dict(weights)
    return model


def adapt(job):
    args, seed = job
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    weights = torch.load(args.base / f"held-out_seed{seed}.pt", map_location="cpu",
                         weights_only=True)
    model = load(weights)
    ap.train_imputation(model, n_steps=args.steps, batch_size=BATCH_SIZE, lr=LR,
                        max_gap=4, seed=51000+seed, device="cpu", log_every=0,
                        motion_sampler=tr.sample_trajectories)
    torch.save({"model": model.state_dict(), "steps": args.steps, "seed": seed},
               args.checkpoints / f"trajectory_seed{seed}.pt")


def inputs(batch, cued):
    shown = ap.endpoint_mask(len(batch.frames), tr.FRAMES)
    shown[:, CUE] = cued
    return ap._to_model_input(ap.apply_mask(batch.frames, shown), "cpu")


def endpoint_keys(batch):
    return [row.tobytes() for row in
            np.packbits(batch.frames[:, [0, -1]].astype(bool), axis=-1)]


def fitting_batches(n, exclude):
    rng = np.random.default_rng(FIT_SEED)
    while n:
        batch = tr.sample_trajectories(rng, 64)
        take = [i for i, k in enumerate(endpoint_keys(batch)) if k not in exclude][:n]
        if not take:
            continue
        yield batch, take
        n -= len(take)


@torch.inference_mode()
def fit_examples(model, n, exclude):
    xs, ys = [], []
    for batch, take in fitting_batches(n, exclude):
        for cue in (False, True):
            activity, _ = model.rnn(inputs(batch, cue))
            xs.append(activity[-1, take].numpy())
            ys.append(batch.centres[take][:, SCORE])
    return np.concatenate(xs), np.concatenate(ys)


def visible_line(centres, cued):
    """Piecewise uniform motion between visible centroids, with no hidden-frame input."""
    visible = [0, CUE, tr.FRAMES-1] if cued else [0, tr.FRAMES-1]
    pieces = []
    for t in SCORE:
        right = next(i for i in visible if i > t)
        left = max(i for i in visible if i < t)
        pieces.append(centres[:, left] + (centres[:, right] - centres[:, left]) *
                      ((t - left) / (right - left)))
    return np.stack(pieces, axis=1)


def factor_scores(prediction, truth):
    """(accel, path, chord, time, xy); choose one factor with the other held fixed."""
    def choice(axis):
        pred, target = (a.swapaxes(axis, 0) for a in (prediction, truth))
        # [true class, candidate class, other factor, chord]
        err = np.linalg.norm(pred[:, None] - target[None], axis=-1).mean(-1)
        ties = np.isclose(err, err.min(1, keepdims=True), atol=1e-7, rtol=0)
        return float((ties / ties.sum(1, keepdims=True))[np.arange(3), np.arange(3)].mean())

    errors = np.linalg.norm(prediction - truth, axis=-1)
    return {"mean_px": float(errors.mean()), "within_2px": float((errors <= 2).mean()),
            "accel_choice": choice(0), "path_choice": choice(1),
            "mean_px_by_accel_path": errors.mean(axis=(-2, -1)).tolist()}


@torch.inference_mode()
def test_seed(model, n_fit, n_chords):
    model.eval()
    chords = tr.sample_chords(np.random.default_rng(TEST_SEED), n_chords)
    exclude = set(endpoint_keys(tr.render(chords, "steady", "straight")))
    fit_x, fit_y = fit_examples(model, n_fit, exclude)
    scores = {}
    for cued in (False, True):
        xs, ys, lines, patch_err = [], [], [], []
        for accel in tr.ACCEL:
            for path in tr.PATH:
                batch = tr.render(chords, accel, path)
                x = inputs(batch, cued)
                states, errors = [], []
                for start in range(0, n_chords, 32):
                    logits, activity = model(x[:, start:start+32])
                    states.append(activity[-1].numpy())
                    images = logits[SCORE].sigmoid().transpose(0, 1).reshape(
                        -1, len(SCORE), tr.SIDE, tr.SIDE).numpy()
                    for i, movie in enumerate(images, start):
                        keep = ~ap._dilate(batch.frames[i, 0] > .5, 1)
                        for t, frame in zip(SCORE, movie):
                            xy = ap._largest_patch(frame * keep, .25)
                            errors.append(np.nan if xy is None else
                                          np.linalg.norm(xy - batch.centres[i, t]))
                xs.append(np.concatenate(states))
                ys.append(batch.centres[:, SCORE])
                lines.append(visible_line(batch.centres, cued))
                patch_err.append(errors)
        shape = (len(tr.ACCEL), len(tr.PATH), n_chords, len(SCORE), 2)
        truth = np.reshape(ys, shape)
        pred = ridge_predict(fit_x, fit_y, np.concatenate(xs)).reshape(shape)
        err = np.array(patch_err)
        scores["plus_cue" if cued else "endpoints"] = {
            "probe": factor_scores(pred, truth),
            "visible_line": factor_scores(np.reshape(lines, shape), truth),
            "patch_found": float(np.isfinite(err).mean()),
            "patch_within_2px": float((err <= 2).mean()),
        }
    return scores


def evaluate(args):
    torch.set_num_threads(2)
    records = []
    for seed in range(args.seeds):
        ckpt = torch.load(args.checkpoints / f"trajectory_seed{seed}.pt",
                          weights_only=True, map_location="cpu")
        scores = test_seed(load(ckpt["model"]), args.fit_trials, args.test_chords)
        rec = {"seed": seed, "adaptation_steps": ckpt["steps"], "scores": scores}
        if args.base is not None:
            original = torch.load(args.base / f"held-out_seed{seed}.pt", map_location="cpu",
                                  weights_only=True)
            rec["before_adaptation"] = test_seed(load(original), args.fit_trials, args.test_chords)
        records.append(rec)
    results = {"acceleration": tr.ACCEL, "path": tr.PATH, "fit_movies": args.fit_trials,
               "fit_seed": FIT_SEED, "test_chords": args.test_chords, "test_seed": TEST_SEED,
               "cue_frame_1based": CUE+1, "scored_frames_1based": [t+1 for t in SCORE],
               "adapt_batch_size": BATCH_SIZE, "adapt_lr": LR, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    plot(records, args.figure)


def plot(records, dest):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.4), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1])
    cells = left.subplots(3, 3, sharex=True, sharey=True)
    chord = tr.Chords(np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[16.]]),
                      np.array([4]), np.array(["square"]))
    for i, path in enumerate(tr.PATH):
        for j, accel in enumerate(tr.ACCEL):
            xy = tr.trajectory(chord, accel, path)[0]
            ax = cells[i, j]
            ax.plot(*xy.T, color="#adc0cb", lw=1)
            ax.scatter(*xy.T, s=15, c="#0072B2")
            ax.scatter(*xy[CUE], s=38, marker="D", color="#D55E00")
            ax.set(xlim=(-9, 9), ylim=(6, -6), aspect="equal", xticks=[], yticks=[])
            ax.spines[:].set_visible(False)
            if i == 0:
                ax.set_title(["Steady", "Speeds up", "Slows down"][j], size=10)
            if j == 0:
                ax.set_ylabel(["Straight", "Left arc", "Right arc"][i], size=10)
    left.suptitle("Same endpoints, nine possible movies\n"
                  "Each dot is one frame; diamond = optional flash", fontsize=12)

    ax = right.subplots()
    series = [("scores", "probe", "Varied-motion RNN", "#0072B2"),
              ("scores", "visible_line", "Join the visible flashes", "#D55E00")]
    if "before_adaptation" in records[0]:
        series.insert(0, ("before_adaptation", "probe", "Straight-trained RNN", "#777777"))
    for shift, (stage, key, label, colour) in zip(np.linspace(-.2, .2, len(series)), series):
        values = np.array([[r[stage][mode][key][factor]
                            for mode in ("endpoints", "plus_cue")
                            for factor in ("accel_choice", "path_choice")] for r in records]) * 100
        x = np.array([0, 1, 2.5, 3.5]) + shift
        ax.vlines(x, values.min(0), values.max(0), colors=colour, lw=2)
        ax.scatter(x, values.mean(0), color=colour, s=45, label=label)
    ax.axhline(100/3, color="#444444", ls=":", lw=1, label="Chance (1 of 3)")
    ax.axvline(1.75, color="#cccccc", lw=1)
    ax.set(ylim=(0, 105), ylabel="Correct trajectory choice (%)",
           xticks=[0, 1, 2.5, 3.5], xticklabels=["Accel.", "Path", "Accel.", "Path"])
    ax.set_xlabel("Endpoints only                 + intermediate flash")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    right.suptitle("Which trajectory fits the blank frames?\n"
                   f"Mean and range across {len(records)} RNN seeds", fontsize=12)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    train = cmds.add_parser("adapt")
    train.add_argument("--base", type=Path, required=True)
    train.add_argument("--steps", type=int, default=2000)
    test = cmds.add_parser("evaluate")
    test.add_argument("--base", type=Path, help="also probe the original straight-motion models")
    test.add_argument("--fit-trials", type=int, default=4096)
    test.add_argument("--test-chords", type=int, default=192)
    test.add_argument("--out", type=Path, default=Path("results/trajectory_experiment.json"))
    test.add_argument("--figure", type=Path, default=Path("figures/trajectory_experiment.png"))
    train.add_argument("--workers", type=int, default=3)
    for sub in (train, test):
        sub.add_argument("--seeds", type=int, default=3)
        sub.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seeds, getattr(args, "workers", 1), getattr(args, "steps", 1),
           getattr(args, "fit_trials", 1), getattr(args, "test_chords", 1)) < 1:
        parser.error("counts must be positive")
    if args.command == "adapt":
        args.checkpoints.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(adapt, [(args, s) for s in range(args.seeds)]))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
