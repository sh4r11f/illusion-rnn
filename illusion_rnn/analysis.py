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
    condition-mean timecourses. Returns ``(C, U, U)``.

    Boolean masks are accepted for ``units`` and converted to indices.
    """
    if units is None:
        units = np.arange(cond_mean.shape[-1])
    units_arr = np.asarray(units)
    units = np.flatnonzero(units_arr) if units_arr.dtype == bool else units_arr.astype(int)
    n_cond = cond_mean.shape[0]
    rdms = np.zeros((n_cond, len(units), len(units)))
    for c in range(n_cond):
        timecourses = cond_mean[c, cutoff:, :][:, units].T  # (U, T')
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
