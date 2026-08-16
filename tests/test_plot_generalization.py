"""Smoke test for scripts/plot_generalization.py.

Mirrors tests/test_analysis.py::test_run_analyses_script_smoke: invoke the
script as a subprocess against a temp output path with a small trial count,
skip if the flagship checkpoint isn't materialized (git lfs checkout), and
assert it produces figures/generalization.png without error.
"""
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_plot_generalization_script_smoke(tmp_path):
    checkpoint = REPO_ROOT / "checkpoints" / "rnn-pixel_h2048_tam-horiz.pt"
    if not checkpoint.exists() or checkpoint.stat().st_size <= 1024:
        pytest.skip("flagship checkpoint not materialized (git lfs checkout)")

    out = tmp_path / "generalization.png"
    proc = subprocess.run(
        [
            "uv", "run", "python", "scripts/plot_generalization.py",
            "--n-trials", "10", "--seed", "0", "--out", str(out),
        ],
        capture_output=True, text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.exists()
    assert out.stat().st_size > 0

    # The script's printed accuracies must only ever mention the two
    # genuinely disjoint variants -- "basic" duplicates TAM_task (see
    # tests/test_stimuli_integrity.py) and must never reappear here.
    assert "standard" in proc.stdout
    assert "outline" in proc.stdout
    assert "basic" not in proc.stdout
