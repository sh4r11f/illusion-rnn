#!/usr/bin/env python3
"""Figures for the correspondence experiment.

1. accuracy by architecture and family, with 95% CIs and a chance line at 0.5
   (reads the pre-computed sweep summary in results/sweep.json).
2. outline accuracy as a function of ink-mass ratio -- separating "cannot
   infer motion from outline form" from "input too far out of distribution
   to drive the network". This trains ONE RNNNet on the (balanced, filled)
   train split and evaluates that single trained model on test_style at
   several stroke widths, rather than retraining per width: the scientific
   question is whether a filled-bar-trained model TRANSFERS to outline
   strokes, which is an evaluation question, not a retraining one.

Usage:
    .venv/bin/python scripts/plot_results.py
    .venv/bin/python scripts/plot_results.py --skip-ink-curve   # figure 1 only, fast
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display needed to write PNGs

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from illusion_rnn import RNNNet, evaluate, make_dataset, make_env, train  # noqa: E402
from illusion_rnn.generate import ink_mass, render_trial, sampler_for  # noqa: E402


def accuracy_figure(summary, path=Path("figures/correspondence_accuracy.png")):
    """Bar chart of train-split accuracy by architecture, one panel per
    family, with 95% CIs and a chance line at 0.5."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, family in zip(axes, ("balanced", "classic")):
        rows = [r for r in summary
                if r["family"] == family and r["split"] == "train"]
        rows.sort(key=lambda r: r["accuracy_mean"])
        names = [r["architecture"] for r in rows]
        means = [r["accuracy_mean"] for r in rows]
        # NaN CIs (single-seed cells) would break barh's xerr; clip to 0 so a
        # missing interval renders as no error bar instead of raising.
        err = [[0.0 if np.isnan(r["accuracy_ci_low"]) else m - r["accuracy_ci_low"]
                for m, r in zip(means, rows)],
               [0.0 if np.isnan(r["accuracy_ci_high"]) else r["accuracy_ci_high"] - m
                for m, r in zip(means, rows)]]
        ax.barh(names, means, xerr=err, color="#4c72b0")
        ax.axvline(0.5, color="crimson", ls="--", lw=1, label="chance")
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{family} family")
        ax.set_xlabel("accuracy (train split)")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def train_reference_model(*, img_size=64, hidden_size=256, n_epochs=1500,
                           batch_size=64, seed=0, device=None,
                           family="balanced"):
    """Train one RNNNet on the (balanced, filled) train split.

    Mirrors illusion_rnn.sweep.run_cell's RNNNet path exactly (same seeding
    order, same seq_len-equals-one-trial convention) so this model is
    directly comparable to the sweep's RNNNet cells -- it is not a separate,
    differently-configured model.
    """
    import torch

    torch.manual_seed(seed)
    env = make_env("correspondence", split="train", img_size=img_size,
                   family=family, transform="growth")
    env.seed(seed)
    env.new_trial()
    n_steps = env.ob.shape[0]

    model = RNNNet(img_size * img_size, hidden_size, 6, dt=50)
    dataset = make_dataset(env, batch_size=batch_size, seq_len=n_steps)
    train(model, dataset, n_epochs=n_epochs, device=device, log_every=0)
    return model


def ink_curve(model, *, img_size=64, n_eval_trials=500, seed=0, device=None,
              family="balanced", widths=(1, 2, 3, 4), n_mass_samples=200,
              path=Path("figures/outline_ink_curve.png"),
              data_path=Path("results/ink_curve.json")):
    """Accuracy against ink-mass ratio for outline renders at several stroke
    widths, evaluated with the single filled-bar-trained ``model`` passed in.

    A single outline accuracy number cannot distinguish "the model cannot
    infer motion from outline form" (accuracy stays low across every width)
    from "the outline input is simply too far out of distribution to drive
    the network at all" (accuracy recovers as the outline's ink mass
    approaches the filled reference). A curve over stroke width -- which
    controls how much ink the outline carries -- separates the two.
    """
    rng = np.random.default_rng(seed)

    # Reference: mean ink mass of a *filled* frame1 on the train split. Every
    # outline ratio below is measured against this single reference so the
    # ratios are comparable to each other. sampler_for defaults img_size to
    # 64 internally and is otherwise unaware of the canvas render_trial will
    # use below, so img_size must be passed through explicitly here -- same
    # requirement TAMCorrespondenceTask documents for its own sampler.
    fill_sample = sampler_for("train", family=family, img_size=img_size)
    filled_ref = float(np.mean([
        ink_mass(render_trial(fill_sample(rng), img_size=img_size).frame1)
        for _ in range(n_mass_samples)
    ]))

    # Sweep BOTH ink-matching modes. Under `energy` the outline's intensity is
    # rescaled so its total ink equals the filled reference, which pins the
    # ratio at ~1.0 for every stroke width -- so that condition alone gives a
    # degenerate x-axis and cannot answer the question. Under `none` the ink
    # genuinely varies with stroke width. Together they separate the two
    # explanations: if accuracy tracks ink under `none` but holds up under
    # `energy`, the failure is about input drive, not about outline form.
    series = {}
    for ink_match in ("none", "energy"):
        ratios, accuracies = [], []
        for width in widths:
            sample = sampler_for("test_style", family=family,
                                 stroke_width=width, ink_match=ink_match,
                                 img_size=img_size)
            masses = [
                ink_mass(render_trial(sample(rng), img_size=img_size).frame1)
                for _ in range(n_mass_samples)
            ]
            ratio = float(np.mean(masses)) / filled_ref
            ratios.append(ratio)

            env = make_env("correspondence", split="test_style",
                           img_size=img_size, family=family,
                           stroke_width=width, ink_match=ink_match)
            env.seed(seed)
            result = evaluate(model, env, n_trials=n_eval_trials, device=device)
            accuracies.append(result.accuracy)
            print(f"ink_match={ink_match:6s} stroke_width={width}  "
                  f"ink_ratio={ratio:.3f}  accuracy={result.accuracy:.3f}")
        series[ink_match] = {"stroke_width": list(widths),
                             "ink_ratio": ratios, "accuracy": accuracies}

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(series, indent=2))
    print(f"wrote {data_path}")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = {"none": "raw outline (ink varies)",
              "energy": "energy-matched (ink fixed)"}
    for ink_match, marker in (("none", "o-"), ("energy", "s--")):
        s = series[ink_match]
        ax.plot(s["ink_ratio"], s["accuracy"], marker, label=labels[ink_match])
        for w, x, y in zip(s["stroke_width"], s["ink_ratio"], s["accuracy"]):
            ax.annotate(f"w={w}", (x, y), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, alpha=0.7)
    ax.axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
    ax.set_xlabel("outline ink mass / filled ink mass")
    ax.set_ylabel("accuracy (test_style)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    return ratios, accuracies


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=Path("results/sweep.json"))
    parser.add_argument("--accuracy-out", type=Path,
                        default=Path("figures/correspondence_accuracy.png"))
    parser.add_argument("--ink-curve-out", type=Path,
                        default=Path("figures/outline_ink_curve.png"))
    parser.add_argument("--ink-curve-data", type=Path,
                        default=Path("results/ink_curve.json"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1500)
    parser.add_argument("--n-eval-trials", type=int, default=500)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-ink-curve", action="store_true",
                        help="only regenerate the accuracy figure (fast)")
    args = parser.parse_args()

    data = json.loads(args.summary.read_text())
    accuracy_figure(data["summary"], path=args.accuracy_out)

    if not args.skip_ink_curve:
        model = train_reference_model(
            img_size=args.img_size, hidden_size=args.hidden_size,
            n_epochs=args.n_epochs, seed=args.seed, device=args.device,
        )
        ink_curve(
            model, img_size=args.img_size, n_eval_trials=args.n_eval_trials,
            seed=args.seed, device=args.device, widths=tuple(args.widths),
            path=args.ink_curve_out, data_path=args.ink_curve_data,
        )


if __name__ == "__main__":
    main()
