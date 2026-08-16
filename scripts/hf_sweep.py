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

from illusion_rnn.sweep import ARCHITECTURES, aggregate, run_grid


def main():
    records = run_grid(
        architectures=ARCHITECTURES,
        families=("balanced", "classic"),
        hidden_sizes=(256,),
        seeds=tuple(range(5)),
        img_size=64, n_epochs=1500, n_eval_trials=500,
    )
    summary = aggregate(records)
    for row in summary:
        if row["split"] == "train":
            print(f"{row['family']:9s} {row['architecture']:16s} "
                  f"{row['accuracy_mean']:.3f}", flush=True)
    print("RESULTS_JSON " + json.dumps({"records": records, "summary": summary}))


if __name__ == "__main__":
    main()
