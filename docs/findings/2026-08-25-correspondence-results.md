# The correspondence testbed: results and notes

**Date:** 2026-08-25
**Branch:** `correspondence`, merged to `main` as `d1c1e72`
**Artifacts:** `results/sweep.json`, `results/sweep_order.json`,
`results/sweep_order_n1.json`, `results/ink_curve.json`
**Spec / plan:** `docs/superpowers/{specs,plans}/2026-08-15-correspondence-testbed*`

---

## 1. What problem this solves

The 2023 experiment trained a CTRNN to report motion direction on
Transformational Apparent Motion (TAM) stimuli and reached 0.967 accuracy. That
number was read as evidence the network infers motion from shape correspondence
across two frames.

It cannot support that reading. In the original stimulus set frame 1 is a single
fixed image per shape (`init-1`, always index `[0]`), and frames 2–5 repeat one
image. The direction label is therefore **fully determined by frame 2 alone**.
Two very different hypotheses —

- *H1: the network binds frame 1 to frame 2 and infers motion from the
  correspondence*, and
- *H2: the network classifies a single static image*

— predict exactly the same 0.967. Nothing in the experiment separates them.

Supporting measurements taken during the review:

- The entire standard set is **22 direction-bearing frames**. Train and eval
  draw from the same pool, so 0.967 is a training-set number.
- Leave-one-out 1-NN on raw pixels of a **single** frame reaches 0.727 against
  0.333 chance.
- A separate data-integrity problem: `TAM_basic`, documented and reported as a
  held-out variant with 0.967 "transfer," consists of **24 images that are all
  byte-identical to training images** (21 share filenames, 3 are internal
  duplicates). That transfer number was measured on training data.

## 2. The corrected task

`TAMCorrespondenceTask` (`illusion_rnn/envs.py`) with stimuli generated
programmatically by `illusion_rnn/generate.py`.

The key design property: **the bar's geometry is sampled before and
independently of the direction label.** Frame 2 is built from
`(bar_left, bar_length, shape_size)` only; `direction` is drawn last. So frame 2
is bit-identical whichever direction the trial is, and a frame-2-only model is
at chance *by construction*, not by hope. This is pinned by a test asserting the
two directions produce bit-identical `frame2` arrays — if that test ever fails,
the experiment is void.

Two families ship, and the contrast between them is the argument:

| family | what it is | frame 2 readable? |
|---|---|---|
| `balanced` | the corrected task | no — provably 50% |
| `classic` | the 2023 geometry reproduced at scale | yes, by design |

`classic` exists to quantify the defect, not to train on.

### A design constraint worth recording

On a bounded canvas, **both static baselines cannot be driven to chance
simultaneously.** Sampling the bar first makes frame 2 exactly independent of
the label but leaves a frame-1 residual near the canvas edges; sampling the
shape first moves the leak to frame 2. There is no non-empty balanced support on
a line — the constraint propagates outward and breaks at the edge. A periodic
canvas would satisfy both exactly, at the cost of a stimulus that wraps.

The design takes the provable frame-2 floor (the one the 2023 defect is about)
and *measures* the frame-1 residual rather than assuming it away. See spec §3.

## 3. Headline result

5 seeds × 6 architectures × 2 families, 64px, `hidden=256`, `n_epochs=1500`.
Mean accuracy on the training distribution, 95% CI (`results/sweep.json`):

| architecture | `balanced` (corrected) | `classic` (2023 geometry) |
|---|---|---|
| `RNNNet` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| `GRUNet` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| `FFStack` | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| `FF1Only` | 0.818 [0.805, 0.832] | 0.516 [0.489, 0.543] |
| **`FF2Only`** | **0.503 [0.472, 0.534]** | **1.000 [1.000, 1.000]** |

**The bottom row is the whole argument.** On the corrected stimuli a
frame-2-only model sits exactly at chance while binding models are perfect; on a
faithful reproduction of the 2023 geometry, frame 2 alone is *sufficient*. The
original 0.967 is fully consistent with a network doing no temporal integration
whatsoever.

This was verified independently three times at different seeds during
development (0.500, 0.522, 0.510) before the full sweep, and the frame-2
bit-identity was re-checked at 0/500 trials across all four splits.

### On the 0.818 frame-1 floor

`FF1Only` at 0.818 is **an artifact of our own position split**, not a property
of the task. The `train` split restricts bars to the lower 60% of the track, and
narrowing that band makes shape position *more* diagnostic of direction:

| `bar_centre_range` | frame-1-only Bayes-optimal |
|---|---|
| `(0.0, 1.0)` full track | 0.697 |
| `(0.0, 0.6)` — the `train` split | **0.818** |
| `(0.6, 1.0)` — `test_position` | 0.942 |

`FF1Only` is simply sitting at the optimum for the band it trains on. So 0.82 —
not 0.50 — is the bar a binding model has to clear here. Documented at the
`SPLITS` definition in `generate.py` so it doesn't get mis-explained again.

*(An earlier note in this project attributed the 0.818 to "a smooth neural
classifier beating a per-pixel majority-vote heuristic." That was wrong and is
superseded by the table above.)*

## 4. Full results, all splits

Mean accuracy, `balanced` family:

| architecture | train | test_shape | test_position | test_style |
|---|---|---|---|---|
| `RNNNet` | 1.000 | 1.000 | 0.523 | 0.764 |
| `GRUNet` | 1.000 | 1.000 | 0.562 | 0.994 |
| `FFStack` | 1.000 | 1.000 | 0.407 | 0.996 |
| `FF1Only` | 0.818 | 0.821 | 0.495 | 0.816 |
| `FF2Only` | 0.503 | 0.503 | 0.512 | 0.500 |
| `RNNNet-shuffled` | 1.000 | 0.982 | 0.519 | 0.675 |

Splits are defined on generator *parameters*, so disjointness is asserted rather
than hoped for: novel bar positions (upper 40% of the track), novel shapes
(train on square+circle, test on triangle+cross+hexagon), and energy-matched
outline rendering.

## 5. Three findings that were not in the plan

All three came out of *running* the experiments rather than reasoning about
them. Each contradicted a prediction.

### 5.1 The frame-order control was vacuous as designed

`growth+shrink` is supposed to make temporal order load-bearing: the same frame
pair appears in both orders with opposite labels, so destroying order should
destroy the answer. It doesn't.

| | `RNNNet` | `RNNNet-shuffled` |
|---|---|---|
| `n_repeats=4` (the default) | 1.000 | **0.999** — control vacuous |
| `n_repeats=1` | 1.000 | **0.524** — control works |

**Cause is repetition, not order.** Frame 1 occupies one timestep and frame 2 is
repeated four times, so the shuffled multiset is `{f1, f2, f2, f2, f2}` — a
model identifies `f1` as the *singleton* by **count**, and recovers frame role
without ever using position. Order is only genuinely destroyed when each frame
appears exactly once.

Neither the spec, the plan, nor any of the thirteen task reviews caught this. It
surfaced only because the control was actually run and returned a number that
contradicted the prediction.

**Generalizable lesson:** any shuffle-based order control on a stimulus with
repeated frames should be checked for this. The repetition structure leaks the
thing the shuffle is meant to destroy.

### 5.2 Nothing generalizes across position

On `test_position`, *every* architecture collapses to chance — `RNNNet` 0.523,
`GRUNet` 0.562, `FFStack` 0.407 — with **zero abstention**, so these are
committed wrong answers, not refusals.

These are fully-connected models on flattened pixels with no translation
invariance, so this is expected in hindsight. But it means the models scoring
1.000 are solving the task in a **position-locked** way rather than learning an
abstract correspondence rule. Held-out *shapes* transfer perfectly (1.000);
held-out *positions* do not transfer at all.

This is arguably the most important result in the artifact: a model can look
like a perfect "binder" on the training distribution and still have learned
nothing transferable about correspondence.

### 5.3 The outline deficit is a form effect, not weak input drive

The 2023 checkpoint scored 0.100 on outline stimuli, but one number cannot say
whether the network *can't infer motion from outline form* or whether the
outline input is simply too faint to drive it. Evaluating one filled-trained
`RNNNet` across stroke widths under both raw and energy-matched rendering
separates them (`results/ink_curve.json`):

| stroke width | raw: ink ratio → acc | energy-matched: ink ratio → acc |
|---|---|---|
| 1 | 0.490 → **0.818** | 0.970 → **0.778** |
| 2 | 0.827 → 1.000 | 0.974 → 1.000 |
| 3 | 0.937 → 1.000 | 0.987 → 1.000 |
| 4 | 1.003 → 1.000 | 1.000 → 1.000 |

At width 1, restoring the missing ink (0.490 → 0.970) does **not** restore
accuracy (0.818 → 0.778). The thin-stroke deficit is about outline *form*, not
input drive. At widths ≥ 2 the filled-trained model transfers to outline at
ceiling — so on the corrected task outline transfer largely *succeeds*, in
contrast to the 2023 checkpoint's near-total outline failure.

Note that sweeping only the energy-matched condition pins every ratio near 1.0
and yields a degenerate x-axis that cannot answer the question. Both modes are
needed.

## 6. The `TAM_basic` correction

Independent of the new experiment, the repo was publicly reporting a false
result. All 24 `TAM_basic` images are byte-identical to `TAM_task` images, so
the documented 0.967 "transfer to basic" was measured on training data, and the
RSA claim that basic sits inside standard's representational family was true for
the same reason and equally uninformative.

Removed from the transfer comparison in `README.md`, `checkpoints/MANIFEST.md`,
`results/meta.json`, `figures/rsa_cross_variant.png`, and all affected
notebooks. `tests/test_stimuli_integrity.py` now pins which variant pairs are
genuinely disjoint, so regenerating `TAM_basic` as novel images would fail
loudly and force the docs to be revisited.

The outline result was also restated correctly. Accuracy alone hid the fact that
the model **abstains on 77.7% of outline trials** — it mostly refuses to answer
rather than answering wrong. `evaluate()` now reports `abstention_rate`,
`committed_accuracy`, and the full confusion matrix, and those fields are
persisted to `results/meta.json` rather than computed and discarded.

## 7. Notes on what this does and does not show

**Does show:** the corrected task cannot be solved from either frame alone; the
2023 geometry can be solved from frame 2 alone; recurrent, gated, and
no-recurrence architectures all reach ceiling on the corrected task when trained
and tested in-distribution.

**Does not show:** anything about human perception. There is no psychophysics
anchor here. Nothing demonstrates these networks resemble human TAM perception —
the claims are about what the networks *compute*, not about whether they *see*
the illusion. That remains the largest gap before this is publishable science.

**Also unresolved:**

- Position generalization fails completely. Convolutional or otherwise
  translation-equivariant architectures would be the obvious next comparison.
- The order control only works at `n_repeats=1`, which is a shorter memory
  demand than the main sweep uses. The two conditions are therefore not directly
  comparable, and the main sweep's `RNNNet-shuffled` row should not be read as
  an order control at all.
- `FFStack` (no recurrence) matches `RNNNet` everywhere it matters, so nothing
  here establishes that *recurrence* is necessary — only that access to both
  frames is.

## 8. Reproducing

```bash
uv run python scripts/verify_generator.py                 # checkpoint 1: frame-2 label independence
uv run python scripts/verify_baselines.py                 # checkpoint 2: baselines land at their ceilings
uv run python scripts/run_sweep.py --quick                # fast local grid
uv run python scripts/run_order_control.py                # order control (n_repeats=4)
uv run python scripts/run_order_control.py --n-repeats 1  # ... and the condition where it works
uv run python scripts/plot_results.py                     # figures + ink curve
uv run pytest -m slow                                     # the two training-based gates
```

The full grid ran on Hugging Face Jobs (`scripts/hf_sweep.py`, ~1h on one L4,
~$1.20). Suite is 186 tests: 183 fast plus 3 training-based gates marked `slow`
and excluded from default CI.

## 9. Process notes

Built as thirteen tasks, each implemented by a fresh agent and reviewed by a
separate one, with a final whole-branch review. Findings worth recording about
that process:

- **Every substantive bug was found by running something, not by reading it.**
  The order-control failure, the split-boundary overlap, and the `growth+shrink`
  pairing defect all passed code review and failed empirically.
- **Task-scoped review has a blind spot.** One reviewer flagged deliberate
  forward-compatible scaffolding as a Critical bug because it could not see the
  task that consumed it. Conversely, a file explicitly named in the spec's Step 0
  was silently dropped and neither the implementer nor its reviewer noticed —
  it took the whole-branch review to catch.
- **Reviewers that re-execute claims caught things reviewers that read code did
  not.** The strongest verifications in this project were a gradient-flow check
  proving architectural isolation, an independent retrain at a different seed,
  and a known-answer calibration of the mutual-information estimator.
- **Two predictions in the plan were simply wrong** and had to be corrected
  mid-flight: the claim that frame-shuffling caps accuracy at 50% (it doesn't,
  for a growth-only design), and the estimate of the frame-1 floor (twice).
