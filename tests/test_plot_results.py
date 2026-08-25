"""Smoke test for scripts/plot_results.py.

Mirrors tests/test_analysis.py::test_run_analyses_script_smoke and
tests/test_plot_generalization.py: invoke the script as a subprocess with a
tiny model/eval configuration so it completes quickly, then assert it
produces both figures and the ink-curve data file. No checkpoint or LFS
dependency here -- the ink-curve half of the script trains its own (tiny,
throwaway) model from scratch, so there is nothing to skip-guard.

The accuracy-figure half is fed a small synthetic summary (not the real,
multi-hour results/sweep.json) so the test doesn't depend on that file's
presence or its expensive provenance.
"""
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_tiny_summary(path):
    """A minimal but well-formed results/sweep.json-shaped summary, with
    real (architecture, family, split) rows for accuracy_figure to plot."""
    summary = []
    for family in ("balanced", "classic"):
        for i, arch in enumerate(("RNNNet", "FF2Only")):
            summary.append({
                "architecture": arch, "family": family, "hidden_size": 8,
                "split": "train", "n_seeds": 2,
                "accuracy_mean": 0.5 + 0.1 * i,
                "accuracy_ci_low": 0.4 + 0.1 * i,
                "accuracy_ci_high": 0.6 + 0.1 * i,
            })
    path.write_text(json.dumps({"records": [], "summary": summary}))


def test_plot_results_script_smoke(tmp_path):
    summary_path = tmp_path / "sweep.json"
    _write_tiny_summary(summary_path)
    accuracy_out = tmp_path / "accuracy.png"
    ink_out = tmp_path / "ink_curve.png"
    ink_data = tmp_path / "ink_curve.json"

    proc = subprocess.run(
        [
            "uv", "run", "python", "scripts/plot_results.py",
            "--summary", str(summary_path),
            "--accuracy-out", str(accuracy_out),
            "--ink-curve-out", str(ink_out),
            "--ink-curve-data", str(ink_data),
            "--img-size", "32", "--hidden-size", "8",
            "--n-epochs", "5", "--n-eval-trials", "4",
            "--widths", "1", "2",
            "--device", "cpu",
        ],
        capture_output=True, text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    assert accuracy_out.exists() and accuracy_out.stat().st_size > 0
    assert ink_out.exists() and ink_out.stat().st_size > 0

    # The curve sweeps both ink-matching modes: `none` (ink varies with stroke
    # width) and `energy` (ink rescaled to match the filled reference). Both
    # are needed -- `energy` alone pins every ratio near 1.0 and gives a
    # degenerate x-axis that cannot separate a form effect from an
    # input-drive effect.
    data = json.loads(ink_data.read_text())
    assert set(data) == {"none", "energy"}
    for ink_match, series in data.items():
        assert series["stroke_width"] == [1, 2], ink_match
        assert len(series["ink_ratio"]) == 2, ink_match
        assert len(series["accuracy"]) == 2, ink_match
        # Ratios are measured against a filled reference, so they stay positive
        # and should not materially exceed it.
        assert all(0.0 < r <= 1.1 for r in series["ink_ratio"]), ink_match

    # Energy matching should pull the ink ratio closer to the filled reference
    # than the raw outline does at the thinnest stroke -- that is the whole
    # point of having both series.
    assert data["energy"]["ink_ratio"][0] > data["none"]["ink_ratio"][0]
