import numpy as np
import pytest

from illusion_rnn.analysis import (
    BinnedPCA,
    PCAResult,
    active_unit_mask,
    binned_pca_loadings,
    compute_rdms,
    condition_average,
    cross_variant_rdm,
    run_pca,
    second_order_rdm,
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


def test_compute_rdms_boolean_mask_equivalent():
    from illusion_rnn.analysis import compute_rdms

    cond_mean = _sine_cond_mean()
    mask = np.array([True, False, True])
    via_mask = compute_rdms(cond_mean, cutoff=0, units=mask)
    via_ids = compute_rdms(cond_mean, cutoff=0, units=np.array([0, 2]))
    np.testing.assert_allclose(via_mask, via_ids)
    assert via_mask.shape == (1, 2, 2)


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
    # "basic" is excluded from the script's default --variants (all of its
    # images duplicate TAM_task, see tests/test_stimuli_integrity.py), so the
    # smoke test only checks the two genuinely disjoint variants.
    for variant in ("standard", "outline"):
        assert f"{variant}_trajectories" in dynamics
        assert f"{variant}_mean_trajectories" in dynamics
        assert f"{variant}_unit_mean" in dynamics
        assert f"{variant}_unit_rdms" in rsa
        assert f"{variant}_condition_rdm" in rsa
        assert rsa[f"{variant}_unit_rdms"].shape == (3, 5, 5)
        assert not np.isnan(rsa[f"{variant}_condition_rdm"]).any()
    assert dynamics["components"].shape[0] == 2
    # 2 variants (standard, outline) x 3 directions = 6 cross-variant keys.
    assert rsa["cross_rdm"].shape == (6, 6)
    assert len(rsa["cross_keys"]) == 6
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["n_trials"] == 60
    assert set(meta["accuracy"]) == {"standard", "outline"}
