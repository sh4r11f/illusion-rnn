"""Compare frozen JEPA and RNN positions on the original six-blank apparent-motion display."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn.analysis import fit_ridge
from illusion_rnn.jepa import MotionJEPA
from scripts import evaluate_motion_jepa as je, run_position_probe as rp
from scripts.run_trajectory_experiment import load

FIT_SEED, TEST_SEED = 101001, 101002
TIMES = np.arange(2, rp.FRAMES)
ORANGE, BLUE, GREY = "#C75318", "#006DA8", "#5A6270"


def controls(centres):
    a, b = centres[:, :1], centres[:, -1:]
    truth = centres[:, 1:-1]
    return {
        "line": a + (b-a) * np.arange(1, rp.FRAMES-1)[None, :, None] / (rp.FRAMES-1),
        # An oracle endpoint-only jump: choose whichever is closer to the hidden truth.
        "jump": np.where(np.linalg.norm(truth-a, axis=-1, keepdims=True) <
                         np.linalg.norm(truth-b, axis=-1, keepdims=True), a, b),
    }


def metrics(xy, truth):
    error = np.linalg.norm(xy - truth, axis=-1)

    def summary(e):
        valid = np.isfinite(e)
        return {"mean_error_px": float(e[valid].mean()) if valid.any() else None,
                "within_2px_all": float((e <= 2).mean()), "coverage": float(valid.mean())}
    return {**summary(error), "by_frame": [summary(e) for e in error.T]}


@torch.inference_mode()
def jepa_outputs(model, endpoints):
    h, z = [], []
    for i in range(0, len(endpoints), 128):
        ends = endpoints[i:i+128]
        frames = np.zeros((len(ends), rp.FRAMES, rp.SIDE, rp.SIDE), dtype=bool)
        frames[:, je.END] = ends
        state, emb = je.features(model, frames, je.END)
        h.append(state)
        z.append(emb[:, 1:-1])
    return np.concatenate(h), np.concatenate(z)


@torch.inference_mode()
def rnn_ink(model, endpoints):
    centres = []
    yy, xx = np.mgrid[:rp.SIDE, :rp.SIDE]
    for i in range(0, len(endpoints), 64):
        ends = endpoints[i:i+64]
        frames = np.zeros((len(ends), rp.FRAMES, rp.SIDE, rp.SIDE), dtype=np.float32)
        frames[:, je.END] = ends
        p = ap.reconstruct(model, frames, ap.endpoint_mask(len(ends), rp.FRAMES),
                           device="cpu")[:, 1:-1]
        mass = p.sum((-2, -1)).clip(min=1e-20)
        centres.append(np.stack([(p*g).sum((-2, -1))/mass for g in (xx, yy)], axis=-1))
    return np.concatenate(centres)


@torch.inference_mode()
def predictions(rnn, jepa, fit, test):
    truth = fit.centres[:, 1:-1]
    rh, _ = rp.extract(rnn, fit.endpoints)
    rt, patches = rp.extract(rnn, test.endpoints, patches=True)
    jh, _ = jepa_outputs(jepa, fit.endpoints)
    jt, z = jepa_outputs(jepa, test.endpoints)
    # Single-image calibration: only visible fit flashes and their own coordinates.
    target_z = np.concatenate([jepa.target(torch.from_numpy(x).float()).flatten(0, 1).numpy()
                               for x in np.array_split(fit.endpoints, (len(truth)+127)//128)])
    static = fit_ridge(target_z, fit.centres[:, je.END].reshape(-1, 2))
    return {"rnn_state": fit_ridge(rh, truth)(rt), "jepa_state": fit_ridge(jh, truth)(jt),
            "rnn_output": rnn_ink(rnn, test.endpoints), "jepa_output": je.decode_frames(static, z),
            "rnn_patch": patches}


def paired_interval(differences, draws=4000):
    """Resample movies after averaging over their six frames and the fixed network seeds."""
    delta = np.mean(differences, axis=0)
    idx = np.random.default_rng(TEST_SEED+1).integers(len(delta), size=(draws, len(delta)))
    return {"jepa_minus_rnn_px": float(delta.mean()),
            "paired_95_px": np.quantile(delta[idx].mean(1), [.025, .975]).tolist()}


def examples(test, preds):
    rows = []
    for shape in rp.FAMILIAR:
        matches = np.flatnonzero(test.shapes == shape)
        if not len(matches):
            continue
        i = matches[0]  # first trial of this shape, independent of either model's error
        rows.append({"shape": shape, "trial": int(i), "truth": test.centres[i].tolist(),
                     "flashes_yx": [np.argwhere(mask).tolist() for mask in test.endpoints[i]],
                     **{k: v[i].tolist() for k, v in preds.items() if k.endswith("_output")}})
    return rows


def plot_accuracy(result, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.25), layout="constrained",
                             gridspec_kw={"width_ratios": (1, 1, 1.15)})
    for ax, key, title in zip(axes[:2], ("output", "state"),
                              ("A   Model outputs", "B   Same position-probe task")):
        for k, label, color, ls in (("jump", "Nearer endpoint", GREY, "--"),
                                    ("line", "Straight line", GREY, ":")):
            ax.plot(TIMES, [v["mean_error_px"] for v in result["controls"][k]["by_frame"]],
                    color=color, ls=ls, lw=1.5, label=label)
        for model, color, mk in (("rnn", ORANGE, "o"), ("jepa", BLUE, "s")):
            v = np.array([[x["mean_error_px"] for x in r[model+"_"+key]["by_frame"]]
                          for r in result["records"]])
            ax.fill_between(TIMES, v.min(0), v.max(0), color=color, alpha=.15)
            ax.plot(TIMES, v.mean(0), marker=mk, ms=5, color=color, label=model.upper())
        ax.set(title=title, xlabel="Hidden frame", xticks=TIMES, ylim=(0, None))
    top = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(0, top)
    axes[0].set_ylabel("Mean distance from true position (px)")
    axes[1].legend(fontsize=8, frameon=True)
    keys = ("rnn_output", "jepa_output", "rnn_state", "jepa_state", "rnn_patch")
    labels = ("RNN ink centre", "JEPA embedding", "RNN state", "JEPA state", "RNN non-start patch")
    ax = axes[-1]
    for i, k in enumerate(keys):
        vals = 100*np.array([r[k]["within_2px_all"] for r in result["records"]])
        ax.barh(i, vals.mean(), height=.53, color=ORANGE if k.startswith("rnn") else BLUE)
        ax.plot([vals.min(), vals.max()], [i, i], lw=2, color="#232733", marker="|")
        ax.text(vals.max()+2, i, f"{vals.mean():.0f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(title="C   Within 2 px of target", xlabel="% of all missing positions",
           yticks=range(len(keys)), yticklabels=labels, xlim=(0, 106), xticks=[0, 25, 50, 75, 100])
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.set_axisbelow(True)
        a.grid(axis="x" if a == ax else "y", alpha=.15)
    fig.suptitle(f"Apparent motion  |  Two flashes, six missing positions\n"
                 f"{result['test_trials']:,} shared test movies · mean and range of "
                 f"{len(result['records'])} seeds · missing patches count as misses in C", fontsize=13)
    fig.savefig(path, dpi=175)
    plt.close(fig)


def plot_examples(rows, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, len(rows), squeeze=False, figsize=(3.4*len(rows)+.5, 6.2),
                             layout="constrained", gridspec_kw={"height_ratios": (1.35, 1)})
    styles = (("truth", "#333333", "o"), ("rnn_output", ORANGE, "^"), ("jepa_output", BLUE, "s"))
    for j, ex in enumerate(rows):
        ax, lower = axes[:, j]
        xy = np.array(ex["truth"])
        a, b = xy[[0, -1]]
        direction = b-a
        progress = direction / (direction @ direction)
        for label, pixels, point in zip(("A", "B"), ex["flashes_yx"], (a, b)):
            mask = np.zeros((rp.SIDE, rp.SIDE, 4))
            yy, xx = np.transpose(pixels)
            mask[yy, xx] = [.32, .34, .38, .26]
            ax.imshow(mask, interpolation="nearest", origin="upper")
            ax.annotate(label, point, xytext=(0, 4), textcoords="offset points",
                        fontsize=9, ha="center", weight="bold")
        for k, color, marker in styles:
            coords = np.array(ex[k])
            frames = np.arange(1, rp.FRAMES+1) if k == "truth" else TIMES
            kw = {"color": color, "marker": marker, "ms": 4, "lw": 1.4,
                  "ls": ":" if k == "truth" else "-"}
            ax.plot(*coords.T, **kw)
            lower.plot(frames, 100*((coords-a) @ progress), **kw)
        ax.set(aspect="equal", xticks=[], yticks=[], title=ex["shape"].capitalize())
        # Preserve pixel scale and include estimates even when they leave the stimulus canvas.
        all_xy = np.vstack([ex[k] for k, _, _ in styles])
        ax.set_xlim(min(-.5, all_xy[:, 0].min()-1), max(rp.SIDE-.5, all_xy[:, 0].max()+1))
        ax.set_ylim(max(rp.SIDE-.5, all_xy[:, 1].max()+1), min(-.5, all_xy[:, 1].min()-1))
        lower.set(xticks=range(1, rp.FRAMES+1), xlabel="Frame", yticks=[0, 50, 100])
        lower.spines[["top", "right"]].set_visible(False)
        lower.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Position on canvas")
    axes[1, 0].set_ylabel("Progress A → B (%)")
    lo, hi = zip(*(ax.get_ylim() for ax in axes[1]))
    for ax in axes[1]:
        ax.set_ylim(min(lo), max(hi))
    fig.suptitle("Where did each model put the object?\n"
                 "Only the shaded endpoint shapes were shown · seed 0, first test movie of each shape",
                 fontsize=13)
    handles = [Line2D([0], [0], color=c, marker=m, ls=":" if k == "truth" else "-")
               for k, c, m in styles]
    fig.legend(handles, ["True position", "RNN · centre of drawn ink", "JEPA · decoded embedding"],
               loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.savefig(path, dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnn", type=Path, required=True, help="original gap-limited checkpoints")
    parser.add_argument("--jepa", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fit-trials", type=int, default=4096)
    parser.add_argument("--test-trials", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/main_task_comparison.json"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if min(args.seeds, args.fit_trials, args.test_trials) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(2)
    seen = set()
    fit = rp.sample_trials(args.fit_trials, FIT_SEED, fit=True, seen=seen, holdouts=False)
    test = rp.sample_trials(args.test_trials, TEST_SEED, fit=False, seen=seen, holdouts=False)
    truth = test.centres[:, 1:-1]
    records, deltas, illustrations = [], {"output": [], "state": []}, []
    for s in range(args.seeds):
        rnn = load(torch.load(args.rnn / f"held-out_seed{s}.pt", weights_only=True,
                              map_location="cpu")).eval()
        jepa = MotionJEPA().eval()
        jepa.load_state_dict(torch.load(args.jepa / f"jepa_seed{s}.pt", weights_only=True,
                                        map_location="cpu")["model"])
        preds = predictions(rnn, jepa, fit, test)
        records.append({"seed": s, **{k: metrics(v, truth) for k, v in preds.items()}})
        for k in deltas:
            j, r = (np.linalg.norm(preds[m+"_"+k]-truth, axis=-1) for m in ("jepa", "rnn"))
            deltas[k].append((j-r).mean(1))
        if s == 0:
            illustrations = examples(test, preds)
    result = {"fit_trials": args.fit_trials, "test_trials": args.test_trials,
              "fit_seed": FIT_SEED, "test_seed": TEST_SEED, "hidden_frames": TIMES.tolist(),
              "controls": {k: metrics(v, truth) for k, v in controls(test.centres).items()},
              "records": records, "paired_differences": {k: paired_interval(v)
                                                         for k, v in deltas.items()},
              "examples_seed0": illustrations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_accuracy(result, args.figures / "jepa_rnn_main_accuracy.png")
    plot_examples(illustrations, args.figures / "jepa_rnn_main_examples.png")


if __name__ == "__main__":
    main()
