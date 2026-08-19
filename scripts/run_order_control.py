# scripts/run_order_control.py
"""Run the temporal-order control: RNNNet vs RNNNet-shuffled under
``transform="growth+shrink"``.

Under plain ``transform="growth"`` (the main sweep), frame 1 is always a
shape and frame 2 is always a bar, so a model that ignores temporal order
(``RNNNet-shuffled``, which randomly permutes the frame slots before they
reach the network) can still solve the task from content alone -- order
carries no information in that condition. ``transform="growth+shrink"``
mixes in "shrink" trials, which reverse the growth trial's frame1/frame2
roles and its label for the *same underlying frame pair* -- so the correct
answer flips depending on which frame arrived first. Only under this
condition does destroying frame order remove information a binding model
needs.

Prediction: RNNNet-shuffled should drop toward chance (~0.50) on
balanced/train while RNNNet (order-preserving) stays high. If it does not,
that is a real finding about the order control, not a bug to paper over --
report it plainly.

Usage:
    .venv/bin/python scripts/run_order_control.py

Runtime: 2 architectures x 5 seeds = 10 training runs at the same config as
the main sweep (img_size=64, hidden=256, n_epochs=1500). On MPS this is
several minutes per cell; expect this script to run for a while.
"""
import argparse
import json
from pathlib import Path

from illusion_rnn.sweep import aggregate, run_grid

CONFIG = dict(
    architectures=("RNNNet", "RNNNet-shuffled"),
    families=("balanced",),
    hidden_sizes=(256,),
    seeds=tuple(range(5)),
    img_size=64, n_epochs=1500, n_eval_trials=500,
    transform="growth+shrink",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-repeats", type=int, default=4,
        help="timesteps frame 2 occupies. At the default 4 the shuffle "
             "control is vacuous -- a shuffled model identifies frame 1 as "
             "the lone non-repeated frame by count. Pass 1 for the "
             "condition where shuffling genuinely destroys order.",
    )
    parser.add_argument("--out", default=None,
                        help="output path; defaults to a name derived from "
                             "--n-repeats")
    args = parser.parse_args()

    config = dict(CONFIG, n_repeats=args.n_repeats)
    records = run_grid(**config)
    summary = aggregate(records)

    default = ("results/sweep_order.json" if args.n_repeats == 4
               else f"results/sweep_order_n{args.n_repeats}.json")
    out = Path(args.out or default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {k: list(v) if isinstance(v, tuple) else v
                    for k, v in config.items()},
         "records": records, "summary": summary}, indent=2,
    ))
    print(f"wrote {out}  ({len(records)} records, {len(summary)} cells)")

    for row in sorted(summary, key=lambda r: (r["split"], r["architecture"])):
        if row["split"] != "train":
            continue
        print(f"{row['architecture']:16s} {row['accuracy_mean']:.3f} "
              f"[{row['accuracy_ci_low']:.3f}, {row['accuracy_ci_high']:.3f}]")


if __name__ == "__main__":
    main()
