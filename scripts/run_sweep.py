# scripts/run_sweep.py
"""Run the correspondence sweep and write records to results/.

Local smoke run (minutes, MPS):
    .venv/bin/python scripts/run_sweep.py --quick --out results/sweep_local.json

Full grid (HF Jobs, ~1h on l4x1):
    .venv/bin/python scripts/run_sweep.py --out results/sweep.json
"""
import argparse
import json
from pathlib import Path

from illusion_rnn.sweep import FULL_GRID, aggregate, run_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="tiny grid for a local smoke test")
    parser.add_argument("--out", default="results/sweep.json")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    if args.quick:
        config = dict(
            architectures=("FF2Only", "RNNNet"), families=("balanced",),
            hidden_sizes=(64,), seeds=tuple(range(args.seeds)),
            img_size=32, n_epochs=200, n_eval_trials=100,
        )
    else:
        # The full grid is the one shared definition in illusion_rnn.sweep so
        # this entry point and scripts/hf_sweep.py (the Hugging Face Jobs
        # runner) cannot drift apart -- see FULL_GRID's docstring comment.
        config = dict(FULL_GRID, seeds=tuple(range(args.seeds)))

    records = run_grid(**config)
    summary = aggregate(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {k: list(v) if isinstance(v, tuple) else v
                    for k, v in config.items()},
         "records": records, "summary": summary}, indent=2,
    ))
    print(f"wrote {out}  ({len(records)} records, {len(summary)} cells)")

    for row in sorted(summary, key=lambda r: (r["family"], r["split"],
                                              r["architecture"])):
        if row["split"] != "train":
            continue
        print(f"{row['family']:9s} {row['architecture']:16s} "
              f"{row['accuracy_mean']:.3f} "
              f"[{row['accuracy_ci_low']:.3f}, {row['accuracy_ci_high']:.3f}]")


if __name__ == "__main__":
    main()
