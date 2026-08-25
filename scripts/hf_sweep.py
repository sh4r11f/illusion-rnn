# scripts/hf_sweep.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "illusion-rnn @ git+https://github.com/sh4r11f/illusion-rnn@correspondence",
# ]
# ///
"""Run the full correspondence sweep on Hugging Face Jobs.

No data upload is needed: stimuli are generated from a seed inside the job.
Results are printed as JSON on the final line so they can be recovered from the
job log even if the artifact upload fails.

    hf jobs uv run --flavor l4x1 --timeout 2h scripts/hf_sweep.py
"""
import json

from illusion_rnn.sweep import FULL_GRID, aggregate, run_grid


def main():
    # FULL_GRID is the one shared grid definition (illusion_rnn/sweep.py) --
    # both this HF Jobs runner and the local scripts/run_sweep.py build their
    # run_grid() call and their emitted `config` block from it, so the two
    # cannot silently drift apart the way they used to.
    records = run_grid(**FULL_GRID)
    summary = aggregate(records)
    for row in summary:
        if row["split"] == "train":
            print(f"{row['family']:9s} {row['architecture']:16s} "
                  f"{row['accuracy_mean']:.3f}", flush=True)
    config = {k: list(v) if isinstance(v, tuple) else v
              for k, v in FULL_GRID.items()}
    print("RESULTS_JSON " + json.dumps(
        {"config": config, "records": records, "summary": summary},
    ))


if __name__ == "__main__":
    main()
