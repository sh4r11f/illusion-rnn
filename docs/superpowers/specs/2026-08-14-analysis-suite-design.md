# Dynamics & RSA Analysis Suite — Design Spec

**Date:** 2026-08-14
**Branch:** `analysis` (off merged `main` @ fd16d95)
**Status:** Approved design, pending implementation plan

## 1. Goal

Reimplement the analyses from the 2022–2023 legacy notebooks (`notebooks/legacy/
prem_analysis*.ipynb`, `analysis_normal.ipynb`, `analysis_outline.ipynb`) as a
clean, tested, numpy-only module plus a compute script and two visualization
notebooks, run against the current flagship checkpoint. Scripts compute and save
artifacts; notebooks load artifacts and visualize. The suite shows *how* the
network represents TAM — and how that representation changes on the variants
where behavior collapses (standard 0.967 vs outline ~0.1).

## 2. Legacy inventory (what "the analyses I had" are)

All four legacy analysis notebooks share one pipeline per stimulus variant:

1. **Global PCA** (2 components) on hidden activity concatenated across trials;
   per-trial trajectories in PC1–2 colored by motion direction.
2. **Time-binned PCA**: split the trial into 3 time bins, fit PCA per bin,
   extract the top-k loading units per component per bin.
3. **Single-unit timecourses**: condition-averaged activity of those top units.
4. **RSA**: per-direction unit×unit RDMs (1 − Pearson between unit timecourses
   after a stimulus-onset cutoff), network-wide and top-units-only, plus a
   second-order condition×condition RDM.
5. **Input analysis**: a TODO stub, never implemented (stays out of scope).

**Legacy bugs corrected in this implementation** (documented deviations, §7):
- Per-condition activity arrays were overwritten each trial (`slc_activity[gt-1]
  = ac`), so RDMs reflected the *last* trial only → we trial-average.
- Second-order RDM used `np.corrcoef(A, B)[0][1]` on 2-D inputs, which
  correlates two rows of A, not A with B → we correlate flattened
  lower-triangle vectors.
- `np.tril` kept upper-triangle zeros in those vectors → we use
  `np.tril_indices(k=-1)`.
- One variant mixed unit ids and indices (`neu1` vs `n2`).

Legacy targets were 128-unit prototypes (`rnn_v9`/`v13`, 20-step trials,
cutoff = step 5 ≈ 250 ms). New target: the 2048-unit flagship on 9-step trials.

## 3. Established facts the design builds on

- `illusion_rnn.evaluate(model, env, n_trials, ...)` returns `EvalResult` with
  `trials` (dicts incl. `ground_truth`) and `activity` (list of `(T, N)` float
  arrays, T = 9 for default TAMTask timing).
- Trial timeline at dt = 50: steps 0–1 fixation, step 2 frame1 (init frame),
  **step 3 frame2 = first informative frame**, steps 3–6 frames 2–5,
  steps 7–8 decision. Default RSA cutoff is therefore **3**.
- Flagship: `checkpoints/rnn-pixel_h2048_tam-horiz.pt`, 4096→2048→6, CUDA
  tensors (`load_rnn` handles `map_location="cpu"`). Measured (seed 0):
  standard 0.967, outline 0.100, basic 0.967.
- Directions on horizontal: left = 1, middle = 2, right = 3 (frozen map).
- ReLU CTRNN at 2048 units ⇒ many silent/constant units whose Pearson r is
  undefined. **Active-unit masking is mandatory** (std > 1e-12 over the
  post-cutoff condition-mean timecourses, computed jointly across ALL variants
  and directions so every RDM shares one unit set).
- Runtime-dependency constraint carries over: **no sklearn/scipy/seaborn/pandas**.
  PCA is implemented with `np.linalg.svd` (mean-centered; identical results).

## 4. Deliverables & layout

```
illusion_rnn/analysis.py     # numpy-only computation module (no plotting)
scripts/run_analyses.py      # CLI: evaluate flagship on 3 variants -> results/
results/                     # committed artifacts (few MB): dynamics.npz, rsa.npz, meta.json
notebooks/03_dynamics.ipynb  # visualization: PCA + binned loadings + unit timecourses
notebooks/04_rsa.ipynb       # visualization: RDMs + second-order + cross-variant
tests/test_analysis.py       # synthetic-data correctness + script smoke test
figures/pca_trajectories.png # saved by notebook 03 (README embed)
figures/rsa_cross_variant.png# saved by notebook 04 (README embed)
README.md                    # + "Analyses" section
```

## 5. `illusion_rnn/analysis.py` API

All functions are pure, deterministic, and numpy-only. `activity` is
`(n_trials, T, N)` float64; `labels` is `(n_trials,)` int.

- `stack_activity(result: EvalResult) -> tuple[np.ndarray, np.ndarray]` —
  stacks `result.activity` into `(n_trials, T, N)` and labels from
  `result.trials` (`ground_truth`).
- `@dataclass PCAResult`: `components (k, N)`, `mean (N,)`,
  `explained_variance_ratio (k,)`; method `transform(x) -> x @ components.T`
  after mean-centering (accepts `(..., N)`).
- `run_pca(activity, n_components=2) -> PCAResult` — fit on
  `activity.reshape(-1, N)`, mean-centered SVD; deterministic sign convention
  (flip each component so its largest-|loading| entry is positive).
- `binned_pca_loadings(activity, n_bins=3, n_components=2) ->` dataclass
  `BinnedPCA` with `bin_edges (n_bins+1,)` (equal splits of T),
  `loadings (n_bins, n_components, N)`, `explained_variance_ratio
  (n_bins, n_components)`.
- `top_units(loadings, k=10) -> np.ndarray (n_bins, n_components, k)` — top-k
  by loading value per component per bin (legacy `argpartition` semantics,
  sorted descending); `unique_units(top) -> (M,)` sorted unique ids.
- `condition_average(activity, labels) -> (cond_mean (C, T, N), cond_sem
  (C, T, N), conditions (C,))` — conditions sorted ascending.
- `active_unit_mask(cond_means: list[np.ndarray], cutoff=3, eps=1e-12) ->
  (N,) bool` — std over pooled post-cutoff condition-mean timecourses across
  every provided array exceeds eps.
- `compute_rdms(cond_mean, cutoff=3, units=None) -> (C, U, U)` — per
  condition, `1 − corrcoef` of `cond_mean[c, cutoff:, units].T` (units are
  rows). Callers pass `units` = active mask (full-network) or top-unit ids.
- `second_order_rdm(rdms (K, U, U)) -> (K, K)` — Pearson on
  `np.tril_indices(U, k=-1)` vectors, `1 − r`; diagonal exactly 0.
- `cross_variant_rdm(rdms_by_key: dict[str, (U, U)]) -> (keys: list[str],
  matrix (K, K))` — second-order RDM across arbitrary labeled RDMs (used for
  the 9 = 3 variants × 3 directions grid). Insertion order preserved.

## 6. `scripts/run_analyses.py`

CLI (argparse):

```
uv run python scripts/run_analyses.py \
  [--checkpoint checkpoints/rnn-pixel_h2048_tam-horiz.pt] \
  [--variants standard outline basic] [--n-trials 300] [--seed 0] \
  [--cutoff 3] [--n-bins 3] [--top-k 10] [--rdm-top-k 30] \
  [--out results] [--device cpu]
```

Behavior: refuse with a clear message if the checkpoint is an LFS pointer
(size ≤ 1024). For each variant, evaluate **shape-balanced**: one env per
shape (`square`, `circle`, `triangle`) × variant, `n_trials // 3` trials each,
activity concatenated — so conditions reflect motion direction, not shape
identity. Every env seeded with `--seed`; model on `--device`.

Computes and saves:

- `results/dynamics.npz`:
  - `components (2, N)`, `pca_mean (N,)` — **shared basis fit on the standard
    variant** (trained regime); all variants project into it.
  - per variant `v`: `{v}_trajectories (n_trials, T, 2)`, `{v}_labels
    (n_trials,)`, `{v}_mean_trajectories (C, T, 2)`, `{v}_evr (20,)`
    (per-variant fit, first 20 components), `{v}_loadings (n_bins, 2, N)`,
    `{v}_top_units (n_bins, 2, top_k)`, `{v}_unit_ids (M,)` (unique top
    units), `{v}_unit_mean (C, T, M)`, `{v}_unit_sem (C, T, M)`.
  - `bin_edges`, `conditions (C,)`, `t_ms (T,)` (step times in ms).
- `results/rsa.npz`:
  - `active_units (N,) bool` (joint mask), per variant `{v}_unit_rdms
    (C, K, K)` over the top `rdm_top_k` **most active** units (by mean
    post-cutoff activity across variants — one shared unit list
    `rdm_unit_ids (K,)` so heatmaps are comparable), `{v}_condition_rdm
    (C, C)` from full active-unit RDMs, `cross_keys (9,) str`,
    `cross_rdm (9, 9)`, `cutoff`.
- `results/meta.json`: checkpoint path, per-variant accuracy from the runs,
  n_trials, seed, all CLI params, package version, ISO date.

Prints per-variant accuracy as it goes (sanity: ≈ manifest numbers).

## 7. Documented deviations from the legacy analyses

Recorded in the module docstring and the notebooks' intro markdown:
trial-averaging (bug fix), correct second-order correlation (bug fix),
`tril_indices(k=-1)` (bug fix), active-unit masking (necessity at 2048 ReLU
units), shared PC basis fit on standard (legacy fit per notebook), cutoff at
frame2 onset (step 3 of 9) rather than legacy 250 ms (step 5 of 20), shape-
balanced evaluation, top-K RDM storage (2048² heatmaps are unreadable; the
full-network structure enters via the second-order stats).

## 8. Notebooks (load `results/` only; no heavy recompute; executed committed)

- `03_dynamics.ipynb`: (a) PC1–2 trajectories, 1×3 panel per variant — thin
  per-trial spaghetti (≤50/variant) + bold direction means, shared axes,
  saved to `figures/pca_trajectories.png`; (b) explained-variance bars
  (first 10) per variant; (c) binned loading heatmaps for the standard
  variant's top units; (d) top-unit condition-mean timecourses (mean ± sem)
  for standard vs outline side by side, period boundaries marked. Interpretive
  markdown ties trajectory separation (or collapse) to the behavioral numbers.
- `04_rsa.ipynb`: (a) unit RDM heatmaps (the shared most-active `rdm_top_k`
  units) per direction for standard and outline; (b) second-order condition RDMs per variant (1×3); (c) the 9×9
  cross-variant RDM with variant-block gridlines, saved to
  `figures/rsa_cross_variant.png`; interpretive markdown: does outline sit in
  the standard representational family or outside it, and how that mirrors
  0.967 → 0.100.
- Both: `kernelspec` metadata included; matplotlib only; runtime seconds.

## 9. Tests (`tests/test_analysis.py`)

Synthetic-data correctness, fast (<10 s):
- `run_pca` recovers planted rank-2 structure (EVR[0:2] ≈ all variance,
  transform shape, deterministic across calls, sign convention stable).
- `binned_pca_loadings`/`top_units` shapes; planted high-variance unit appears
  in top units.
- `condition_average` exact values on a tiny hand-built array; sem correctness.
- `active_unit_mask` excludes constant units, includes varying ones.
- `compute_rdms`: identical timecourses → off-diagonal 0; anti-correlated
  pair → 2; diagonal 0; no NaNs when given active units.
- `second_order_rdm`: identical RDMs → 0; diagonal 0; symmetric.
- `cross_variant_rdm` key ordering preserved.
- Script smoke test: `run_analyses.py --n-trials 6 --top-k 3 --rdm-top-k 5
  --out <tmp>` (subprocess) writes both `.npz` + `meta.json` with the expected
  keys; **skips** when the checkpoint is an LFS pointer (CI-safe).

## 10. README

Add an "## Analyses" section after the generalization result: what the suite
computes (one paragraph), the script invocation, pointers to notebooks 03/04,
and the `figures/rsa_cross_variant.png` embed. Mention the corrected-legacy
provenance in one sentence.

## 11. Out of scope

The legacy "input analysis" stub, temporal (T×T) RDMs, fixed-point/linearized
dynamics analyses, noise ceilings/permutation statistics, CNN-feature
checkpoints, vertical orientation, motion-control (`cnt`/`track`) tasks.

## 12. Acceptance criteria

1. `uv run pytest -q` green (existing 84 + new analysis tests), checkpoint
   tests active locally.
2. `uv run python scripts/run_analyses.py` (defaults) completes on this
   machine; `results/` artifacts written; printed accuracies within 0.05 of
   the manifest numbers for standard/basic (sampling noise allowance).
3. Both notebooks execute end-to-end via nbconvert, committed executed, and
   read only `results/` + save the two README figures.
4. README section added; both figures committed.
5. `pyproject.toml` runtime dependencies unchanged (numpy-only analysis).
6. No change to `main`; everything lands on `analysis`.
