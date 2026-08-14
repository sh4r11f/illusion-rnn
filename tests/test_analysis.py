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
