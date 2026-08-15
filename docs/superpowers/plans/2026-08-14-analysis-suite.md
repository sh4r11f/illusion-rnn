# Analysis Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Numpy-only dynamics/RSA analysis module + compute script + two executed visualization notebooks, reproducing (and correcting) the legacy analyses against the flagship checkpoint.

**Architecture:** `illusion_rnn/analysis.py` holds pure computation (PCA via SVD, condition averaging, RDMs, second-order RSA). `scripts/run_analyses.py` evaluates the flagship on standard/outline/basic (shape-balanced, seeded), computes everything, writes `results/dynamics.npz`, `results/rsa.npz`, `results/meta.json` (committed). Notebooks 03/04 load `results/` and only plot.

**Tech Stack:** numpy (no sklearn/scipy/seaborn/pandas), matplotlib in notebooks only, existing `illusion_rnn` API (`load_rnn`, `make_env`, `evaluate`), nbformat/nbconvert for notebook builds.

**Spec:** `docs/superpowers/specs/2026-08-14-analysis-suite-design.md`.

## Global Constraints

- Work on branch `analysis`. Never touch `main`.
- `pyproject.toml` runtime dependencies must remain exactly: `neurogym>=2.3`, `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`, `pillow>=10.0`. The analysis module imports numpy only (dataclasses/stdlib allowed).
- Frozen direction labels on horizontal TAM: left=1, middle=2, right=3. Direction names in artifacts/plots: `{1: "left", 2: "middle", 3: "right"}`.
- Trial timeline (dt=50, default TAMTask): steps 0–1 fixation, 2 frame1, **3 frame2 (first informative frame — default RSA `cutoff=3`)**, 3–6 frames2–5, 7–8 decision. T=9.
- Flagship checkpoint: `checkpoints/rnn-pixel_h2048_tam-horiz.pt` (LFS; refuse politely on pointer files ≤1024 bytes). Known accuracies (seed 0): standard 0.967, outline 0.100, basic 0.967 — script-printed accuracies should be within ~0.05 for standard/basic.
- Every commit message ends with these two trailer lines:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ
  ```
- Run everything through uv (`uv run pytest`, `uv run python`, `uv run jupyter`).
- Never write `.pt`/`.zip` scratch files in-repo; `results/*.npz` are small (<5 MB total) and ARE committed.
- Notebooks are committed EXECUTED with `kernelspec` metadata; builder scripts are deleted after use, never committed.

---

### Task 1: Analysis module — PCA, binning, condition averaging

**Files:**
- Create: `illusion_rnn/analysis.py`, `tests/test_analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: numpy; `EvalResult` shape (`.activity` list of `(T, N)`, `.trials` dicts with `ground_truth`).
- Produces (consumed by Tasks 2–5):
  - `stack_activity(result) -> (activity (n_trials, T, N) float64, labels (n_trials,) int)`
  - `PCAResult` dataclass: `components (k, N)`, `mean (N,)`, `explained_variance_ratio (k,)`, method `transform(x: (..., N)) -> (..., k)`
  - `run_pca(activity, n_components=2) -> PCAResult`
  - `BinnedPCA` dataclass: `bin_edges (n_bins+1,) int`, `loadings (n_bins, n_components, N)`, `explained_variance_ratio (n_bins, n_components)`
  - `binned_pca_loadings(activity, n_bins=3, n_components=2) -> BinnedPCA`
  - `top_units(loadings, k=10) -> (n_bins, n_components, k) int` (top by signed loading value, sorted descending — legacy semantics)
  - `unique_units(top) -> (M,) int` sorted unique
  - `condition_average(activity, labels, conditions=None) -> (cond_mean (C, T, N), cond_sem (C, T, N), conditions (C,))` — `conditions=None` uses sorted unique labels; explicit `conditions` raises `ValueError` naming any condition with zero trials.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis.py`:

```python
import numpy as np
import pytest

from illusion_rnn.analysis import (
    BinnedPCA,
    PCAResult,
    binned_pca_loadings,
    condition_average,
    run_pca,
    stack_activity,
    top_units,
    unique_units,
)


class _FakeResult:
    def __init__(self, activity, labels):
        self.activity = list(activity)
        self.trials = [{"ground_truth": int(l), "choice": 0, "correct": False} for l in labels]


def _rank2_activity(n_trials=40, n_t=9, n_units=30, seed=0):
    """Activity whose variance lives in a planted 2-D subspace."""
    rng = np.random.RandomState(seed)
    basis = rng.randn(2, n_units)
    coeffs = rng.randn(n_trials, n_t, 2)
    return coeffs @ basis + 1e-6 * rng.randn(n_trials, n_t, n_units)


def test_stack_activity():
    acts = [np.full((9, 4), float(i)) for i in range(3)]
    result = _FakeResult(acts, [1, 2, 3])
    activity, labels = stack_activity(result)
    assert activity.shape == (3, 9, 4)
    np.testing.assert_array_equal(labels, [1, 2, 3])
    assert activity[2, 0, 0] == 2.0


def test_run_pca_recovers_planted_subspace():
    activity = _rank2_activity()
    result = run_pca(activity, n_components=3)
    assert isinstance(result, PCAResult)
    assert result.components.shape == (3, 30)
    evr = result.explained_variance_ratio
    assert evr[0] + evr[1] > 0.999
    assert evr[2] < 1e-3


def test_run_pca_transform_shape_and_determinism():
    activity = _rank2_activity()
    a = run_pca(activity)
    b = run_pca(activity)
    np.testing.assert_array_equal(a.components, b.components)
    out = a.transform(activity)
    assert out.shape == (40, 9, 2)
    single = a.transform(activity[0, 0])
    assert single.shape == (2,)


def test_run_pca_sign_convention():
    activity = _rank2_activity()
    comps = run_pca(activity).components
    for comp in comps:
        assert comp[np.argmax(np.abs(comp))] > 0


def test_binned_pca_shapes_and_edges():
    activity = _rank2_activity(n_t=9)
    binned = binned_pca_loadings(activity, n_bins=3, n_components=2)
    assert isinstance(binned, BinnedPCA)
    np.testing.assert_array_equal(binned.bin_edges, [0, 3, 6, 9])
    assert binned.loadings.shape == (3, 2, 30)
    assert binned.explained_variance_ratio.shape == (3, 2)


def test_top_units_finds_planted_unit():
    rng = np.random.RandomState(1)
    activity = 0.01 * rng.randn(50, 9, 20)
    activity[:, :, 7] += 10.0 * rng.randn(50, 9)  # dominant-variance unit
    binned = binned_pca_loadings(activity, n_bins=3, n_components=2)
    top = top_units(binned.loadings, k=3)
    assert top.shape == (3, 2, 3)
    # unit 7 dominates PC1 of every bin (sign convention makes its loading positive)
    assert all(7 in top[b, 0] for b in range(3))
    assert 7 in unique_units(top)


def test_top_units_sorted_descending():
    loadings = np.array([[[0.1, 0.9, 0.5, -0.3]]])
    top = top_units(loadings, k=3)
    np.testing.assert_array_equal(top[0, 0], [1, 2, 0])


def test_condition_average_exact():
    activity = np.zeros((4, 2, 1))
    activity[0, :, 0] = [1.0, 2.0]
    activity[1, :, 0] = [3.0, 4.0]
    activity[2, :, 0] = [10.0, 20.0]
    activity[3, :, 0] = [10.0, 20.0]
    labels = np.array([1, 1, 2, 2])
    mean, sem, conditions = condition_average(activity, labels)
    np.testing.assert_array_equal(conditions, [1, 2])
    np.testing.assert_allclose(mean[0, :, 0], [2.0, 3.0])
    np.testing.assert_allclose(mean[1, :, 0], [10.0, 20.0])
    np.testing.assert_allclose(sem[0, :, 0], np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
    np.testing.assert_allclose(sem[1, :, 0], 0.0)


def test_condition_average_explicit_conditions():
    activity = np.ones((2, 3, 2))
    labels = np.array([1, 3])
    mean, _, conditions = condition_average(activity, labels, conditions=np.array([1, 3]))
    assert mean.shape == (2, 3, 2)
    with pytest.raises(ValueError, match="no trials"):
        condition_average(activity, labels, conditions=np.array([1, 2, 3]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_analysis.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'illusion_rnn.analysis'`.

- [ ] **Step 3: Implement `illusion_rnn/analysis.py` (first half)**

```python
"""Dynamics and RSA analyses for TAM-trained networks (numpy-only).

Reimplements the analyses of the 2022-2023 legacy notebooks
(``notebooks/legacy/prem_analysis*.ipynb``, ``analysis_*.ipynb``) with these
documented corrections and deviations:

- Per-condition activity is trial-averaged; the legacy code overwrote its
  per-condition buffer on every trial, so RDMs reflected only the last trial.
- Second-order RDMs correlate flattened strict-lower-triangle vectors
  (``np.tril_indices(..., k=-1)``). The legacy ``np.corrcoef(A, B)[0][1]`` on
  2-D inputs correlated two rows of A, and its ``np.tril`` kept zero entries.
- Silent/constant units are excluded via an explicit active-unit mask before
  any correlation (Pearson r is undefined on a flat timecourse; at 2048 ReLU
  units many are silent).
- PCA is mean-centered SVD with a deterministic sign convention (the
  largest-|loading| entry of each component is positive).
- ``top_units`` keeps the legacy semantics: top-k by *signed* loading value.
"""

from dataclasses import dataclass

import numpy as np


def stack_activity(result):
    """Stack an ``EvalResult`` into ``(activity (n_trials, T, N), labels)``."""
    activity = np.stack(result.activity, axis=0).astype(np.float64)
    labels = np.array([t["ground_truth"] for t in result.trials], dtype=int)
    return activity, labels


@dataclass
class PCAResult:
    """Principal components fit: ``components (k, N)``, ``mean (N,)``,
    ``explained_variance_ratio (k,)``."""

    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project ``(..., N)`` data onto the components -> ``(..., k)``."""
        return (np.asarray(x) - self.mean) @ self.components.T


def _fit_pca(data: np.ndarray, n_components: int):
    mean = data.mean(axis=0)
    centered = data - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    evr = variance / variance.sum()
    components = vt[:n_components].copy()
    for i, component in enumerate(components):
        if component[np.argmax(np.abs(component))] < 0:
            components[i] = -component
    return components, mean, evr[:n_components]


def run_pca(activity: np.ndarray, n_components: int = 2) -> PCAResult:
    """Fit PCA on all timepoints of all trials (``(n_trials, T, N)``)."""
    data = np.asarray(activity).reshape(-1, activity.shape[-1])
    components, mean, evr = _fit_pca(data, n_components)
    return PCAResult(components=components, mean=mean, explained_variance_ratio=evr)


@dataclass
class BinnedPCA:
    """Per-time-bin PCA fits: ``bin_edges (n_bins+1,)``,
    ``loadings (n_bins, n_components, N)``,
    ``explained_variance_ratio (n_bins, n_components)``."""

    bin_edges: np.ndarray
    loadings: np.ndarray
    explained_variance_ratio: np.ndarray


def binned_pca_loadings(
    activity: np.ndarray, n_bins: int = 3, n_components: int = 2,
) -> BinnedPCA:
    """Fit a separate PCA on each of ``n_bins`` equal time bins."""
    n_timepoints, n_units = activity.shape[1], activity.shape[2]
    edges = np.linspace(0, n_timepoints, n_bins + 1).astype(int)
    loadings = np.zeros((n_bins, n_components, n_units))
    evr = np.zeros((n_bins, n_components))
    for b in range(n_bins):
        data = activity[:, edges[b]:edges[b + 1], :].reshape(-1, n_units)
        loadings[b], _, evr[b] = _fit_pca(data, n_components)
    return BinnedPCA(bin_edges=edges, loadings=loadings, explained_variance_ratio=evr)


def top_units(loadings: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k units per (bin, component) by signed loading, sorted descending."""
    order = np.argsort(loadings, axis=-1)[..., ::-1]
    return order[..., :k].astype(int)


def unique_units(top: np.ndarray) -> np.ndarray:
    """Sorted unique unit ids from a ``top_units`` array."""
    return np.unique(np.asarray(top, dtype=int).ravel())


def condition_average(
    activity: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray | None = None,
):
    """Trial-average activity per condition.

    Returns ``(cond_mean (C, T, N), cond_sem (C, T, N), conditions (C,))``.
    With explicit ``conditions``, raises ``ValueError`` if any has no trials.
    """
    labels = np.asarray(labels)
    if conditions is None:
        conditions = np.unique(labels)
    conditions = np.asarray(conditions)
    n_cond = len(conditions)
    _, n_t, n_units = activity.shape
    cond_mean = np.zeros((n_cond, n_t, n_units))
    cond_sem = np.zeros((n_cond, n_t, n_units))
    for i, condition in enumerate(conditions):
        selected = activity[labels == condition]
        if selected.shape[0] == 0:
            msg = f"condition {condition} has no trials"
            raise ValueError(msg)
        cond_mean[i] = selected.mean(axis=0)
        ddof = 1 if selected.shape[0] > 1 else 0
        cond_sem[i] = selected.std(axis=0, ddof=ddof) / np.sqrt(selected.shape[0])
    return cond_mean, cond_sem, conditions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_analysis.py -v`
Expected: all PASS. If `test_top_units_finds_planted_unit` fails, check the sign convention runs BEFORE selection (it does not affect `argsort` order per component, but the planted unit's loading must be positive — the convention guarantees that because unit 7 has the largest |loading|).

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q` — expected: 84 old + 9 new = 93 passed.

```bash
git add illusion_rnn/analysis.py tests/test_analysis.py
git commit -m "Add analysis module: PCA, binned loadings, condition averaging" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 2: Analysis module — RDMs and second-order RSA

**Files:**
- Modify: `illusion_rnn/analysis.py` (append), `tests/test_analysis.py` (append)
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: Task 1's `condition_average` outputs.
- Produces (consumed by Tasks 3–5):
  - `active_unit_mask(cond_means: list[np.ndarray], cutoff=3, eps=1e-12) -> (N,) bool` — True only for units whose post-cutoff timecourse varies (std > eps) in EVERY condition of EVERY provided array.
  - `compute_rdms(cond_mean, cutoff=3, units=None) -> (C, U, U)` — per condition `1 − corrcoef` of unit timecourses `cond_mean[c, cutoff:, units].T`.
  - `second_order_rdm(rdms (K, U, U)) -> (K, K)` — `1 − Pearson` between strict-lower-triangle vectors; diagonal exactly 0; symmetric.
  - `cross_variant_rdm(rdms_by_key: dict[str, (U, U)]) -> (list[str], (K, K))` — insertion order preserved.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_analysis.py` (add `active_unit_mask, compute_rdms, cross_variant_rdm, second_order_rdm` to the existing `from illusion_rnn.analysis import ...`):

```python
def test_active_unit_mask():
    cond_mean = np.zeros((2, 9, 3))
    cond_mean[:, :, 0] = np.arange(9)          # varies in both conditions
    cond_mean[0, :, 1] = np.arange(9)          # varies in cond 0 only
    cond_mean[:, :, 2] = 5.0                   # constant everywhere
    from illusion_rnn.analysis import active_unit_mask

    mask = active_unit_mask([cond_mean], cutoff=3)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_active_unit_mask_joint_across_variants():
    a = np.zeros((1, 9, 2))
    a[0, :, 0] = np.arange(9)
    a[0, :, 1] = np.arange(9)
    b = a.copy()
    b[0, :, 1] = 1.0  # unit 1 flat in variant b
    from illusion_rnn.analysis import active_unit_mask

    np.testing.assert_array_equal(active_unit_mask([a, b], cutoff=0), [True, False])


def _sine_cond_mean():
    """(1, 9, 3): unit1 == unit0, unit2 == -unit0 (post-cutoff)."""
    t = np.linspace(0, 2 * np.pi, 9)
    cond_mean = np.zeros((1, 9, 3))
    cond_mean[0, :, 0] = np.sin(t)
    cond_mean[0, :, 1] = np.sin(t)
    cond_mean[0, :, 2] = -np.sin(t)
    return cond_mean


def test_compute_rdms_correlation_structure():
    from illusion_rnn.analysis import compute_rdms

    rdms = compute_rdms(_sine_cond_mean(), cutoff=0)
    assert rdms.shape == (1, 3, 3)
    np.testing.assert_allclose(np.diag(rdms[0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(rdms[0, 0, 1], 0.0, atol=1e-12)  # identical -> 0
    np.testing.assert_allclose(rdms[0, 0, 2], 2.0, atol=1e-12)  # anti -> 2
    assert not np.isnan(rdms).any()


def test_compute_rdms_units_subset():
    from illusion_rnn.analysis import compute_rdms

    rdms = compute_rdms(_sine_cond_mean(), cutoff=0, units=np.array([0, 2]))
    assert rdms.shape == (1, 2, 2)
    np.testing.assert_allclose(rdms[0, 0, 1], 2.0, atol=1e-12)


def test_second_order_rdm():
    from illusion_rnn.analysis import second_order_rdm

    rng = np.random.RandomState(0)
    base = rng.rand(5, 5)
    base = (base + base.T) / 2
    np.fill_diagonal(base, 0)
    other = rng.rand(5, 5)
    other = (other + other.T) / 2
    np.fill_diagonal(other, 0)
    rdms = np.stack([base, base.copy(), other])
    out = second_order_rdm(rdms)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-12)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-12)  # identical RDMs
    assert out[0, 2] > 1e-3
    np.testing.assert_allclose(out, out.T, atol=1e-12)


def test_cross_variant_rdm_key_order():
    from illusion_rnn.analysis import cross_variant_rdm

    rng = np.random.RandomState(1)
    rdms = {}
    for name in ("b-first", "a-second", "c-third"):
        m = rng.rand(4, 4)
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0)
        rdms[name] = m
    keys, matrix = cross_variant_rdm(rdms)
    assert keys == ["b-first", "a-second", "c-third"]
    assert matrix.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_analysis.py -v`
Expected: Task 1 tests PASS; new tests ERROR with `ImportError`.

- [ ] **Step 3: Append to `illusion_rnn/analysis.py`**

```python
def active_unit_mask(cond_means, cutoff: int = 3, eps: float = 1e-12) -> np.ndarray:
    """Units whose post-cutoff timecourse varies in EVERY condition of EVERY
    provided ``(C, T, N)`` array (Pearson r is undefined on flat timecourses)."""
    masks = []
    for cond_mean in cond_means:
        stds = cond_mean[:, cutoff:, :].std(axis=1)  # (C, N)
        masks.append((stds > eps).all(axis=0))
    return np.logical_and.reduce(masks)


def compute_rdms(
    cond_mean: np.ndarray, cutoff: int = 3, units: np.ndarray | None = None,
) -> np.ndarray:
    """Per-condition unit-by-unit RDMs: ``1 - corrcoef`` of post-cutoff
    condition-mean timecourses. Returns ``(C, U, U)``."""
    if units is None:
        units = np.arange(cond_mean.shape[-1])
    units = np.asarray(units, dtype=int)
    n_cond = cond_mean.shape[0]
    rdms = np.zeros((n_cond, len(units), len(units)))
    for c in range(n_cond):
        timecourses = cond_mean[c, cutoff:, units].T  # (U, T')
        rdms[c] = 1.0 - np.corrcoef(timecourses)
        np.fill_diagonal(rdms[c], 0.0)
    return rdms


def second_order_rdm(rdms: np.ndarray) -> np.ndarray:
    """``1 - Pearson`` between the strict lower triangles of ``(K, U, U)``
    RDMs. Diagonal is exactly 0."""
    rows, cols = np.tril_indices(rdms.shape[1], k=-1)
    vectors = rdms[:, rows, cols]  # (K, U*(U-1)/2)
    out = 1.0 - np.corrcoef(vectors)
    np.fill_diagonal(out, 0.0)
    return out


def cross_variant_rdm(rdms_by_key: dict) -> tuple[list, np.ndarray]:
    """Second-order RDM across labeled RDMs; insertion order preserved."""
    keys = list(rdms_by_key)
    stacked = np.stack([rdms_by_key[key] for key in keys], axis=0)
    return keys, second_order_rdm(stacked)
```

- [ ] **Step 4: Run tests, full suite, commit**

Run: `uv run pytest tests/test_analysis.py -v` then `uv run pytest -q`
Expected: all pass (84 + 15 analysis tests = 99).

```bash
git add illusion_rnn/analysis.py tests/test_analysis.py
git commit -m "Add RSA: active-unit masking, RDMs, second-order and cross-variant" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 3: Compute script + real run producing `results/`

**Files:**
- Create: `scripts/run_analyses.py`, `results/` artifacts (dynamics.npz, rsa.npz, meta.json)
- Modify: `tests/test_analysis.py` (append smoke test)
- Test: smoke test + the real run itself

**Interfaces:**
- Consumes: full `illusion_rnn` API + Tasks 1–2 functions.
- Produces: the artifact schema Tasks 4–5 load (keys exactly as below).

- [ ] **Step 1: Append the smoke test**

Append to `tests/test_analysis.py`:

```python
def test_run_analyses_script_smoke(tmp_path):
    import json
    import subprocess
    from pathlib import Path

    checkpoint = Path(__file__).resolve().parent.parent / "checkpoints" / "rnn-pixel_h2048_tam-horiz.pt"
    if not checkpoint.exists() or checkpoint.stat().st_size <= 1024:
        pytest.skip("flagship checkpoint not materialized (git lfs checkout)")
    proc = subprocess.run(
        [
            "uv", "run", "python", "scripts/run_analyses.py",
            "--n-trials", "60", "--top-k", "3", "--rdm-top-k", "5",
            "--seed", "0", "--out", str(tmp_path), "--device", "cpu",
        ],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    dynamics = np.load(tmp_path / "dynamics.npz")
    rsa = np.load(tmp_path / "rsa.npz")
    for variant in ("standard", "outline", "basic"):
        assert f"{variant}_trajectories" in dynamics
        assert f"{variant}_mean_trajectories" in dynamics
        assert f"{variant}_unit_mean" in dynamics
        assert f"{variant}_unit_rdms" in rsa
        assert f"{variant}_condition_rdm" in rsa
        assert rsa[f"{variant}_unit_rdms"].shape == (3, 5, 5)
        assert not np.isnan(rsa[f"{variant}_condition_rdm"]).any()
    assert dynamics["components"].shape[0] == 2
    assert rsa["cross_rdm"].shape == (9, 9)
    assert len(rsa["cross_keys"]) == 9
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["n_trials"] == 60
    assert set(meta["accuracy"]) == {"standard", "outline", "basic"}
```

Note: with `--n-trials 60` (20 per shape, 60 per variant) and seed 0, every direction appears with overwhelming probability; the run is deterministic, so verify once — if a direction happens to be missing, the script exits with the clear error from `condition_average` and the fix is bumping the smoke test to `--n-trials 90`.

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `uv run pytest tests/test_analysis.py::test_run_analyses_script_smoke -v`
Expected: FAIL (`returncode != 0`, stderr shows the script doesn't exist yet).

- [ ] **Step 3: Implement `scripts/run_analyses.py`**

```python
#!/usr/bin/env python3
"""Run the dynamics/RSA analysis suite on a TAM checkpoint.

Evaluates the model shape-balanced (square/circle/triangle) per stimulus
variant, then computes PCA trajectories, binned loadings, top-unit
timecourses, unit RDMs, second-order condition RDMs, and the cross-variant
second-order RDM. Saves compact artifacts to --out for the visualization
notebooks (03_dynamics, 04_rsa).

Usage:
    uv run python scripts/run_analyses.py [--n-trials 300] [--seed 0]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from illusion_rnn import SHAPES, __version__, evaluate, load_rnn, make_env  # noqa: E402
from illusion_rnn import analysis as an  # noqa: E402

DIRECTION_NAMES = {1: "left", 2: "middle", 3: "right"}
CONDITIONS = np.array([1, 2, 3])
DT_MS = 50.0  # default TAMTask dt; envs below are constructed with defaults


def collect_variant(model, variant, n_trials, seed, device):
    """Shape-balanced evaluation; returns (activity, labels, mean accuracy)."""
    activities, labels, accuracies = [], [], []
    per_shape = max(1, n_trials // len(SHAPES))
    for shape in SHAPES:
        env = make_env(
            "tam", box_shape=shape, variant=variant,
            stim_ori="horizontal", img_size=64,
        )
        env.seed(seed)
        result = evaluate(model, env, n_trials=per_shape, device=device)
        activity, trial_labels = an.stack_activity(result)
        activities.append(activity)
        labels.append(trial_labels)
        accuracies.append(result.accuracy)
    return (
        np.concatenate(activities, axis=0),
        np.concatenate(labels, axis=0),
        float(np.mean(accuracies)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/rnn-pixel_h2048_tam-horiz.pt"))
    parser.add_argument("--variants", nargs="+",
                        default=["standard", "outline", "basic"])
    parser.add_argument("--n-trials", type=int, default=300,
                        help="per variant (split across the 3 shapes)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff", type=int, default=3,
                        help="first analyzed timestep (frame2 onset)")
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10,
                        help="top loading units per PC per bin")
    parser.add_argument("--rdm-top-k", type=int, default=30,
                        help="units in the stored RDM heatmaps")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists() or args.checkpoint.stat().st_size <= 1024:
        msg = (
            f"{args.checkpoint} is missing or an LFS pointer; run "
            "'git lfs install --local && git lfs checkout' first"
        )
        raise SystemExit(msg)

    model = load_rnn(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- evaluate ----------------------------------------------------------
    runs = {}
    accuracy = {}
    for variant in args.variants:
        activity, labels, acc = collect_variant(
            model, variant, args.n_trials, args.seed, args.device,
        )
        cond_mean, cond_sem, _ = an.condition_average(
            activity, labels, conditions=CONDITIONS,
        )
        runs[variant] = (activity, labels, cond_mean, cond_sem)
        accuracy[variant] = round(acc, 3)
        print(f"{variant}: n={len(labels)} accuracy={acc:.3f}")

    # ---- dynamics ----------------------------------------------------------
    basis_variant = "standard" if "standard" in runs else args.variants[0]
    basis = an.run_pca(runs[basis_variant][0], n_components=2)
    n_timepoints = next(iter(runs.values()))[0].shape[1]
    dynamics = {
        "components": basis.components.astype(np.float32),
        "pca_mean": basis.mean.astype(np.float32),
        "t_ms": np.arange(n_timepoints) * DT_MS,
        "conditions": CONDITIONS,
        "basis_variant": np.array(basis_variant),
    }
    for variant, (activity, labels, cond_mean, cond_sem) in runs.items():
        evr = an.run_pca(activity, n_components=min(20, activity.shape[-1]))
        binned = an.binned_pca_loadings(
            activity, n_bins=args.n_bins, n_components=2,
        )
        top = an.top_units(binned.loadings, k=args.top_k)
        unit_ids = an.unique_units(top)
        dynamics.update({
            f"{variant}_trajectories": basis.transform(activity).astype(np.float32),
            f"{variant}_labels": labels,
            f"{variant}_mean_trajectories": basis.transform(cond_mean),
            f"{variant}_evr": evr.explained_variance_ratio,
            f"{variant}_loadings": binned.loadings.astype(np.float32),
            f"{variant}_top_units": top,
            f"{variant}_unit_ids": unit_ids,
            f"{variant}_unit_mean": cond_mean[:, :, unit_ids],
            f"{variant}_unit_sem": cond_sem[:, :, unit_ids],
        })
    dynamics["bin_edges"] = binned.bin_edges
    np.savez_compressed(args.out / "dynamics.npz", **dynamics)

    # ---- rsa ---------------------------------------------------------------
    cond_means = [runs[v][2] for v in args.variants]
    mask = an.active_unit_mask(cond_means, cutoff=args.cutoff)
    active_ids = np.where(mask)[0]
    if len(active_ids) < args.rdm_top_k:
        msg = f"only {len(active_ids)} active units (< --rdm-top-k {args.rdm_top_k})"
        raise SystemExit(msg)
    pooled_rate = np.mean(
        [cm[:, args.cutoff:, :].mean(axis=(0, 1)) for cm in cond_means], axis=0,
    )
    order = np.argsort(pooled_rate[active_ids])[::-1]
    rdm_unit_ids = active_ids[order[: args.rdm_top_k]]

    rsa = {
        "active_units": mask,
        "rdm_unit_ids": rdm_unit_ids,
        "cutoff": np.array(args.cutoff),
    }
    cross = {}
    for variant in args.variants:
        cond_mean = runs[variant][2]
        rsa[f"{variant}_unit_rdms"] = an.compute_rdms(
            cond_mean, cutoff=args.cutoff, units=rdm_unit_ids,
        )
        full_rdms = an.compute_rdms(cond_mean, cutoff=args.cutoff, units=active_ids)
        rsa[f"{variant}_condition_rdm"] = an.second_order_rdm(full_rdms)
        for i, condition in enumerate(CONDITIONS):
            cross[f"{variant}-{DIRECTION_NAMES[int(condition)]}"] = full_rdms[i]
    keys, cross_rdm = an.cross_variant_rdm(cross)
    rsa["cross_keys"] = np.array(keys)
    rsa["cross_rdm"] = cross_rdm
    np.savez_compressed(args.out / "rsa.npz", **rsa)

    # ---- meta --------------------------------------------------------------
    meta = {
        "checkpoint": str(args.checkpoint),
        "variants": list(args.variants),
        "n_trials": args.n_trials,
        "seed": args.seed,
        "cutoff": args.cutoff,
        "n_bins": args.n_bins,
        "top_k": args.top_k,
        "rdm_top_k": args.rdm_top_k,
        "device": args.device,
        "accuracy": accuracy,
        "n_active_units": int(mask.sum()),
        "package_version": __version__,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out}/dynamics.npz, rsa.npz, meta.json "
          f"({mask.sum()} active units)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_analysis.py::test_run_analyses_script_smoke -v`
Expected: PASS (~30–60 s: 180 evals + analysis). If it fails with "condition N has no trials", bump the smoke test's `--n-trials` to 90 (deterministic seed — verify once).

- [ ] **Step 5: The real run**

```bash
uv run python scripts/run_analyses.py
ls -la results/
uv run python -c "
import json, numpy as np
meta = json.load(open('results/meta.json'))
print(meta['accuracy'])
d = np.load('results/dynamics.npz'); r = np.load('results/rsa.npz')
print('dynamics keys:', len(d.files), '| rsa keys:', len(r.files))
print('active units:', meta['n_active_units'])
print('cross_rdm:\n', np.round(r['cross_rdm'], 2))
"
```

Expected: accuracies ≈ {standard ~0.95–1.0, outline ~0.05–0.15, basic ~0.95–1.0} (within 0.05 of manifest for standard/basic); artifacts total < 5 MB (`du -sh results/`). Record the printed cross-RDM in your report — Task 5's notebook narrates it.

- [ ] **Step 6: Full suite + commit (script, test, artifacts)**

Run: `uv run pytest -q` — expected 100 passed (99 + smoke).

```bash
git add scripts/run_analyses.py tests/test_analysis.py results/
git commit -m "Add analysis compute script and committed results artifacts" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 4: Dynamics notebook (03)

**Files:**
- Create: `notebooks/03_dynamics.ipynb` (committed executed), `figures/pca_trajectories.png`
- Temp (create, run, delete): `notebooks/_build_dynamics.py`
- Test: nbconvert execution + zero-error check

**Interfaces:**
- Consumes: `results/dynamics.npz`, `results/meta.json` (Task 3 schema).
- Produces: `figures/pca_trajectories.png` (README embed, Task 6).

- [ ] **Step 1: Create `notebooks/_build_dynamics.py`**

```python
"""One-shot builder for 03_dynamics.ipynb (run once, then delete)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        "# Hidden dynamics: PCA of the TAM RNN\n\n"
        "Visualizes precomputed artifacts from `scripts/run_analyses.py` "
        "(`results/dynamics.npz`) for the flagship pixel RNN. The analyses "
        "reimplement the 2022–2023 legacy notebooks with documented "
        "corrections (see `illusion_rnn/analysis.py`). The PC basis is fit "
        "on the *standard* variant — the trained regime — and all variants "
        "are projected into that shared space."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n"
        "D = np.load(ROOT / \"results\" / \"dynamics.npz\", allow_pickle=False)\n"
        "META = json.loads((ROOT / \"results\" / \"meta.json\").read_text())\n"
        "VARIANTS = META[\"variants\"]\n"
        "CONDITIONS = D[\"conditions\"]\n"
        "DIR_NAMES = {1: \"left\", 2: \"middle\", 3: \"right\"}\n"
        "DIR_COLORS = {1: \"#1f77b4\", 2: \"#7f7f7f\", 3: \"#d62728\"}\n"
        "T_MS = D[\"t_ms\"]\n"
        "print(\"accuracies:\", META[\"accuracy\"])"
    ),
    md(
        "## 1. Trajectories in the trained state space\n\n"
        "Thin lines: single trials (up to 50 per variant). Bold lines: "
        "condition means. Marker: trial start."
    ),
    code(
        "fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5 * len(VARIANTS), 4.6),\n"
        "                         sharex=True, sharey=True)\n"
        "for ax, variant in zip(axes, VARIANTS):\n"
        "    traj = D[f\"{variant}_trajectories\"]\n"
        "    labels = D[f\"{variant}_labels\"]\n"
        "    mean_traj = D[f\"{variant}_mean_trajectories\"]\n"
        "    for t in range(min(50, traj.shape[0])):\n"
        "        color = DIR_COLORS[int(labels[t])]\n"
        "        ax.plot(traj[t, :, 0], traj[t, :, 1], color=color, alpha=0.15, lw=0.8)\n"
        "    for i, cond in enumerate(CONDITIONS):\n"
        "        ax.plot(mean_traj[i, :, 0], mean_traj[i, :, 1],\n"
        "                color=DIR_COLORS[int(cond)], lw=2.5, label=DIR_NAMES[int(cond)])\n"
        "        ax.scatter(mean_traj[i, 0, 0], mean_traj[i, 0, 1],\n"
        "                   color=DIR_COLORS[int(cond)], s=25, zorder=5)\n"
        "    ax.set_title(f\"{variant} (acc {META['accuracy'][variant]:.2f})\")\n"
        "    ax.set_xlabel(\"PC1\")\n"
        "axes[0].set_ylabel(\"PC2\")\n"
        "axes[0].legend(frameon=False, fontsize=9)\n"
        "fig.suptitle(\"Hidden-state trajectories in the standard-fit PC space\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(ROOT / \"figures\" / \"pca_trajectories.png\", dpi=150,\n"
        "            bbox_inches=\"tight\")"
    ),
    md("## 2. How much variance do the leading PCs carry?"),
    code(
        "fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 3.2),\n"
        "                         sharey=True)\n"
        "for ax, variant in zip(axes, VARIANTS):\n"
        "    evr = D[f\"{variant}_evr\"][:10]\n"
        "    ax.bar(np.arange(1, len(evr) + 1), evr, color=\"#4c72b0\")\n"
        "    ax.set_title(variant)\n"
        "    ax.set_xlabel(\"PC\")\n"
        "axes[0].set_ylabel(\"explained variance ratio\")\n"
        "fig.tight_layout()"
    ),
    md(
        "## 3. Time-binned PCA: which units carry the components?\n\n"
        "Loadings of the standard variant's top units, per time bin "
        "(bins split the 9-step trial into thirds)."
    ),
    code(
        "unit_ids = D[\"standard_unit_ids\"]\n"
        "loadings = D[\"standard_loadings\"]  # (n_bins, 2, N)\n"
        "n_bins = loadings.shape[0]\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 0.28 * len(unit_ids) + 2))\n"
        "for pc in range(2):\n"
        "    mat = loadings[:, pc, :][:, unit_ids]  # (n_bins, M)\n"
        "    im = axes[pc].imshow(mat.T, aspect=\"auto\", cmap=\"RdBu_r\",\n"
        "                         vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())\n"
        "    axes[pc].set_title(f\"PC{pc + 1} loadings\")\n"
        "    axes[pc].set_xlabel(\"time bin\")\n"
        "    axes[pc].set_xticks(range(n_bins))\n"
        "    axes[pc].set_yticks(range(len(unit_ids)))\n"
        "    axes[pc].set_yticklabels(unit_ids, fontsize=7)\n"
        "    fig.colorbar(im, ax=axes[pc], shrink=0.8)\n"
        "axes[0].set_ylabel(\"unit id\")\n"
        "fig.suptitle(\"Standard variant: top-unit loadings per time bin\")\n"
        "fig.tight_layout()"
    ),
    md(
        "## 4. Top-unit timecourses by condition\n\n"
        "Condition means ± s.e.m. for the standard variant's top units, on "
        "standard (left column) vs outline (right column) stimuli. Dashed "
        "lines: frame1 onset, frame2 onset (first informative frame), "
        "decision onset."
    ),
    code(
        "show_variants = [v for v in (\"standard\", \"outline\") if v in VARIANTS]\n"
        "unit_ids = D[\"standard_unit_ids\"]\n"
        "n_show = min(6, len(unit_ids))\n"
        "fig, axes = plt.subplots(n_show, len(show_variants),\n"
        "                         figsize=(5.5 * len(show_variants), 2.0 * n_show),\n"
        "                         sharex=True, squeeze=False)\n"
        "for col, variant in enumerate(show_variants):\n"
        "    ids = D[f\"{variant}_unit_ids\"]\n"
        "    mean = D[f\"{variant}_unit_mean\"]\n"
        "    sem = D[f\"{variant}_unit_sem\"]\n"
        "    for row in range(n_show):\n"
        "        ax = axes[row][col]\n"
        "        unit = unit_ids[row]\n"
        "        where = np.where(ids == unit)[0]\n"
        "        if len(where) == 0:\n"
        "            ax.set_visible(False)\n"
        "            continue\n"
        "        j = int(where[0])\n"
        "        for i, cond in enumerate(CONDITIONS):\n"
        "            color = DIR_COLORS[int(cond)]\n"
        "            ax.plot(T_MS, mean[i, :, j], color=color,\n"
        "                    label=DIR_NAMES[int(cond)] if row == 0 else None)\n"
        "            ax.fill_between(T_MS, mean[i, :, j] - sem[i, :, j],\n"
        "                            mean[i, :, j] + sem[i, :, j],\n"
        "                            color=color, alpha=0.2, lw=0)\n"
        "        for onset in (100, 150, 350):\n"
        "            ax.axvline(onset, color=\"k\", ls=\"--\", lw=0.6, alpha=0.5)\n"
        "        ax.set_ylabel(f\"unit {unit}\", fontsize=8)\n"
        "        if row == 0:\n"
        "            ax.set_title(variant)\n"
        "            if col == 0:\n"
        "                ax.legend(frameon=False, fontsize=8)\n"
        "for col in range(len(show_variants)):\n"
        "    axes[-1][col].set_xlabel(\"time (ms)\")\n"
        "fig.tight_layout()"
    ),
    md(
        "## Reading the figures\n\n"
        "On the standard and basic variants the condition means separate into "
        "direction-specific branches after frame2 (the first informative "
        "frame) — the geometry the readout exploits at 0.97 accuracy. Compare "
        "the outline panel: if its trajectories fail to branch (or branch "
        "into a different geometry), the behavioral collapse to ~0.10 has a "
        "representational counterpart. Notebook 04 quantifies exactly that "
        "with RSA."
    ),
]

nbf.write(nb, "notebooks/03_dynamics.ipynb")
print("wrote notebooks/03_dynamics.ipynb")
```

- [ ] **Step 2: Build, execute, sanity-check, clean up**

```bash
uv run python notebooks/_build_dynamics.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/03_dynamics.ipynb
rm notebooks/_build_dynamics.py
uv run python - <<'EOF'
import json
nb = json.load(open("notebooks/03_dynamics.ipynb"))
errors = [o for c in nb["cells"] if c["cell_type"] == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert not errors, errors
assert "kernelspec" in nb["metadata"]
print("ok:", sum(c["cell_type"] == "code" for c in nb["cells"]), "code cells, no errors")
EOF
ls -la figures/pca_trajectories.png
```

Expected: execution in seconds (loads npz only); PNG exists (>30 KB).

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_dynamics.ipynb figures/pca_trajectories.png
git commit -m "Add executed dynamics notebook (PCA trajectories, loadings, unit timecourses)" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 5: RSA notebook (04)

**Files:**
- Create: `notebooks/04_rsa.ipynb` (committed executed), `figures/rsa_cross_variant.png`
- Temp (create, run, delete): `notebooks/_build_rsa.py`
- Test: nbconvert execution + zero-error check

**Interfaces:**
- Consumes: `results/rsa.npz`, `results/meta.json`.
- Produces: `figures/rsa_cross_variant.png` (README embed, Task 6).

- [ ] **Step 1: Create `notebooks/_build_rsa.py`**

```python
"""One-shot builder for 04_rsa.ipynb (run once, then delete)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        "# Representational similarity analysis\n\n"
        "Visualizes `results/rsa.npz` from `scripts/run_analyses.py`. RDMs "
        "are 1 − Pearson r between unit timecourses (condition means, from "
        "frame2 onward). Corrections vs the legacy notebooks: trial "
        "averaging, strict-lower-triangle second-order correlation, and "
        "active-unit masking — see `illusion_rnn/analysis.py`."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n"
        "R = np.load(ROOT / \"results\" / \"rsa.npz\", allow_pickle=False)\n"
        "META = json.loads((ROOT / \"results\" / \"meta.json\").read_text())\n"
        "VARIANTS = META[\"variants\"]\n"
        "DIRS = [\"left\", \"middle\", \"right\"]\n"
        "print(\"active units:\", META[\"n_active_units\"], \"of 2048\")\n"
        "print(\"accuracies:\", META[\"accuracy\"])"
    ),
    md(
        "## 1. Unit-by-unit RDMs per direction\n\n"
        "The same most-active units (shared across variants) — standard on "
        "top, outline below. Similar structure across the three directions "
        "within a variant means direction is encoded by trajectory geometry "
        "more than by which units co-fire."
    ),
    code(
        "show_variants = [v for v in (\"standard\", \"outline\") if v in VARIANTS]\n"
        "fig, axes = plt.subplots(len(show_variants), 3,\n"
        "                         figsize=(11, 3.6 * len(show_variants)),\n"
        "                         squeeze=False)\n"
        "for row, variant in enumerate(show_variants):\n"
        "    rdms = R[f\"{variant}_unit_rdms\"]\n"
        "    for col in range(3):\n"
        "        ax = axes[row][col]\n"
        "        im = ax.imshow(rdms[col], vmin=0, vmax=2, cmap=\"viridis\")\n"
        "        ax.set_title(f\"{variant} — {DIRS[col]}\", fontsize=10)\n"
        "        ax.set_xticks([])\n"
        "        ax.set_yticks([])\n"
        "fig.colorbar(im, ax=axes[:, -1], shrink=0.75, label=\"1 − r\")\n"
        "fig.suptitle(f\"Unit RDMs ({rdms.shape[1]} most-active shared units)\")"
    ),
    md(
        "## 2. Second-order RDMs: do directions share similarity structure?\n\n"
        "Each cell compares two directions' full-network RDMs "
        "(active units only)."
    ),
    code(
        "fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 3.6))\n"
        "for ax, variant in zip(axes, VARIANTS):\n"
        "    matrix = R[f\"{variant}_condition_rdm\"]\n"
        "    im = ax.imshow(matrix, vmin=0, cmap=\"magma\")\n"
        "    ax.set_xticks(range(3))\n"
        "    ax.set_xticklabels(DIRS, fontsize=8)\n"
        "    ax.set_yticks(range(3))\n"
        "    ax.set_yticklabels(DIRS, fontsize=8)\n"
        "    for i in range(3):\n"
        "        for j in range(3):\n"
        "            ax.text(j, i, f\"{matrix[i, j]:.2f}\", ha=\"center\",\n"
        "                    va=\"center\", color=\"w\", fontsize=8)\n"
        "    ax.set_title(f\"{variant}\")\n"
        "    fig.colorbar(im, ax=ax, shrink=0.8)\n"
        "fig.suptitle(\"Condition (direction) second-order RDMs\")\n"
        "fig.tight_layout()"
    ),
    md(
        "## 3. Cross-variant RSA: is outline TAM represented like standard TAM?\n\n"
        "Second-order dissimilarity between all 9 (variant × direction) "
        "full-network RDMs. Block structure by variant (gridlines every 3) "
        "answers the headline question directly."
    ),
    code(
        "keys = [str(k) for k in R[\"cross_keys\"]]\n"
        "matrix = R[\"cross_rdm\"]\n"
        "fig, ax = plt.subplots(figsize=(7.5, 6.5))\n"
        "im = ax.imshow(matrix, vmin=0, cmap=\"magma\")\n"
        "ax.set_xticks(range(len(keys)))\n"
        "ax.set_xticklabels(keys, rotation=45, ha=\"right\", fontsize=8)\n"
        "ax.set_yticks(range(len(keys)))\n"
        "ax.set_yticklabels(keys, fontsize=8)\n"
        "for cut in (2.5, 5.5):\n"
        "    ax.axhline(cut, color=\"w\", lw=1.5)\n"
        "    ax.axvline(cut, color=\"w\", lw=1.5)\n"
        "fig.colorbar(im, label=\"1 − r (second-order)\")\n"
        "ax.set_title(\"Cross-variant representational similarity\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(ROOT / \"figures\" / \"rsa_cross_variant.png\", dpi=150,\n"
        "            bbox_inches=\"tight\")\n"
        "block = lambda a, b: float(np.mean(matrix[a * 3:(a + 1) * 3, b * 3:(b + 1) * 3]))\n"
        "print(\"mean within-standard block:\", round(block(0, 0), 3))\n"
        "print(\"mean standard-outline block:\", round(block(0, 1), 3))\n"
        "print(\"mean standard-basic block:\", round(block(0, 2), 3))"
    ),
    md(
        "## Reading the matrix\n\n"
        "If the standard–basic off-diagonal block is nearly as similar as the "
        "within-variant blocks while the standard–outline block stands apart, "
        "the network carries a shared TAM representation that outline stimuli "
        "fail to engage — the representational counterpart of the behavioral "
        "generalization pattern (standard/basic ≈ 0.97, outline ≈ 0.10). If "
        "instead outline sits inside the same representational family, the "
        "failure is downstream of the representation (a readout problem). The "
        "printed block means above give the quantitative answer for this run."
    ),
]

nbf.write(nb, "notebooks/04_rsa.ipynb")
print("wrote notebooks/04_rsa.ipynb")
```

- [ ] **Step 2: Build, execute, sanity-check, clean up**

```bash
uv run python notebooks/_build_rsa.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/04_rsa.ipynb
rm notebooks/_build_rsa.py
uv run python - <<'EOF'
import json
nb = json.load(open("notebooks/04_rsa.ipynb"))
errors = [o for c in nb["cells"] if c["cell_type"] == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert not errors, errors
assert "kernelspec" in nb["metadata"]
print("ok, no errors")
EOF
ls -la figures/rsa_cross_variant.png
```

Expected: seconds; PNG exists (>30 KB). Record the three printed block means in your report.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_rsa.ipynb figures/rsa_cross_variant.png
git commit -m "Add executed RSA notebook (unit RDMs, second-order, cross-variant)" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 6: README section + final verification

**Files:**
- Modify: `README.md`
- Test: full suite + acceptance checklist (spec §12)

**Interfaces:**
- Consumes: everything.
- Produces: the finished branch.

- [ ] **Step 1: Add the Analyses section to README.md**

Insert the following between the "## The generalization result" section (after its manifest link line) and "## Plug in your own model":

```markdown
## Analyses

`scripts/run_analyses.py` reproduces the project's dynamics and RSA analyses
(corrected reimplementations of the 2022–23 exploratory notebooks — see the
deviations documented in `illusion_rnn/analysis.py`): PCA of hidden-state
trajectories in the trained state space, time-binned PCA with top-loading
units, condition-averaged unit timecourses, unit-by-unit RDMs, and
second-order RSA within and across stimulus variants.

```bash
uv run python scripts/run_analyses.py   # writes results/*.npz (committed)
```

`notebooks/03_dynamics.ipynb` and `notebooks/04_rsa.ipynb` visualize the
committed artifacts. The cross-variant RSA asks the headline question —
is outline TAM represented like standard TAM? —

![Cross-variant RSA](figures/rsa_cross_variant.png)
```

- [ ] **Step 2: Verify README paths and full suite**

```bash
uv run python - <<'EOF'
from pathlib import Path
for p in ("figures/rsa_cross_variant.png", "figures/pca_trajectories.png",
          "notebooks/03_dynamics.ipynb", "notebooks/04_rsa.ipynb",
          "scripts/run_analyses.py", "results/dynamics.npz",
          "results/rsa.npz", "results/meta.json", "illusion_rnn/analysis.py"):
    assert Path(p).exists(), p
print("all referenced paths exist")
EOF
uv run pytest -q
git status --short
```

Expected: paths ok; 100 passed; clean tree (before README staging).

- [ ] **Step 3: Walk the acceptance checklist (spec §12)**

1. Full suite green with checkpoint tests active — from Step 2.
2. `results/` artifacts exist, committed (Task 3), accuracies in meta.json within 0.05 of manifest for standard/basic.
3. Both notebooks executed via nbconvert, committed executed, read only `results/` (grep both notebooks for `evaluate(`/`load_rnn(` — expect zero hits).
4. README section + both figures committed.
5. `git diff main -- pyproject.toml` shows no dependency changes.
6. Branch is `analysis`; `main` untouched.

Run the notebook-purity check:

```bash
uv run python - <<'EOF'
import json
for f in ("notebooks/03_dynamics.ipynb", "notebooks/04_rsa.ipynb"):
    nb = json.load(open(f))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "evaluate(" not in src and "load_rnn(" not in src and "make_env(" not in src, f
print("notebooks load results/ only")
EOF
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document the analysis suite in README" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
git log --oneline main..analysis
```

Expected: a tidy commit series. Do not push or merge — that is the user's call.

