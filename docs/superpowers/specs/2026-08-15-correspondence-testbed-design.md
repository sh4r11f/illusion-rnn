# Correspondence testbed — design

**Date:** 2026-08-15
**Status:** approved, pre-implementation
**Branch:** `correspondence`

## 1. Why

The 2023 result in this repo — a CTRNN reaching 0.967 on standard TAM, 0.100 on
outline — cannot support the claim it is presented as supporting. Three findings
from the 2026-08-15 review, all verified against the code and data:

1. **The `basic` variant is the training set.** All 24 files in
   `illusion_rnn/stimuli_data/TAM_basic/` are byte-identical to files in
   `TAM_task/` (21 share filenames; 3 are internal duplicates). The reported
   "transfer to basic = 0.967" is measured on training images. This is stated as
   a generalization result in `README.md`, `checkpoints/MANIFEST.md`,
   `results/meta.json`, and `notebooks/04_rsa.ipynb`.

2. **No condition distinguishes motion-from-form from static template
   matching.** In `envs.py`, frame 1 is a single fixed image per shape
   (`init-1`, always index `[0]`) and frames 2–5 repeat one image. The direction
   label is fully determined by that one image. This is *faithful* to
   Transformational Apparent Motion — in TAM frame 1 is constant and direction
   does come from frame 2's geometry — but it means both hypotheses predict
   0.967 equally well. The signature of correspondence-based inference is that
   the answer **flips when frame 1 changes while frame 2 is held fixed**, and no
   trial tests that.

3. **The outline "failure" is an abstention, not an error.** The shipped RNN
   answers the fixation class on 233/300 outline trials (78%). Of the 67 trials
   where it commits to a direction, 30 are correct (0.45 against 0.33 chance),
   biased toward "right". Reporting "accuracy 0.100" collapses abstention and
   error into one number that reads as below-chance confusion, which is not what
   is happening.

Supporting measurements: the standard set contains **22 direction-bearing
frames** total; train and eval draw from the same pool, so 0.967 is training
accuracy. Leave-one-out 1-NN on raw pixels of a *single* frame reaches 0.727
against 0.333 chance. Outline frames sit at cosine 0.306 from their nearest
standard frame, so "cannot infer motion from outline form" and "input too far
out of distribution to drive the network" are not currently separable.

## 2. Goals

Build the experiment that can distinguish the two hypotheses:

- **G1** A task where the direction label depends on the *relation* between
  frame 1 and frame 2, not on either frame alone.
- **G2** Static and order-scrambled baselines whose ceilings are known, so any
  gap above them is attributable to temporal binding.
- **G3** A programmatic stimulus generator with real train/test splits, removing
  the 22-image ceiling.
- **G4** Multi-seed architecture comparison with confidence intervals.
- **G5** An outline variant matched on ink mass, so form-transfer failure is
  separable from out-of-distribution input.

### Non-goals

- Human psychophysics or comparison to published TAM effects. Deferred; it is
  the step after this one.
- Retraining or reinterpreting the 2023 checkpoints. `TAMTask` and the shipped
  checkpoints stay frozen so the archived result keeps reproducing.
- Changing the RSA/dynamics analyses. They become interpretable only once a
  model is doing something other than static classification; revisit after G4.

## 3. The balance problem

The generator's central constraint, worked out during design because it
determines what the experiment can claim.

**Requirement.** For the task to isolate temporal binding, neither frame alone
may carry the label. Formally, with label `d ∈ {left, right}`, we want
`I(frame1; d) = 0` and `I(frame2; d) = 0`.

**Result: on a bounded canvas, both cannot be zero.**

Consider discrete anchor positions `0..N-1` on a horizontal track. A trial is a
pair `(i, d)`: frame 1 places the shape at anchor `i`, frame 2 draws a
homogeneous bar spanning anchors `i` and `i+d`. Zero mutual information requires
both:

- **(A)** every anchor `i` in the support admits both `d = +1` and `d = -1`
- **(B)** every bar span `{j, j+1}` in the support is reachable from both
  origins, i.e. both `(j, +1)` and `(j+1, -1)` are in the support

Start from any interior `i`. (A) forces `(i, +1)`, whose span `{i, i+1}` then
forces `(i+1, -1)` by (B), which forces `(i+1, +1)` by (A), whose span forces
`(i+2, -1)`, and so on. The constraint propagates outward and breaks at the
canvas edge. **No non-empty balanced support exists on a line.** A periodic
canvas (ring) satisfies both exactly, at the cost of a stimulus that wraps
around the image edge.

**Chosen resolution.** Sample the bar span *first and independently of
direction*, then choose the direction, then place frame 1's shape at the
implied end:

```
a  ~ Uniform(bar-span positions)      # independent of d
L  ~ Uniform(bar lengths)             # independent of d
d  ~ Uniform{left, right}
frame2 = bar spanning [a, a+L]        # a function of (a, L) only  ->  frame2 ⊥ d
frame1 = shape at a       if d == right
         shape at a+L-s   if d == left
```

Consequences, each of which the test suite asserts:

- **frame-2-only floor = 50%, exactly and provably.** Frame 2 is a
  deterministic function of `(a, L)`, which are drawn before and independently
  of `d`. No edge effects. This is the floor that matters, because frame-2
  readability is precisely the defect in the 2023 result.
- **frame-1-only floor ≈ 55%, measured not assumed.** The shape's position
  distribution differs slightly between labels near the canvas edges.
  Randomizing `L` per trial smears this; the residual is measured by training an
  actual frame-1-only probe.
- **The headline claim is a measured gap**, not a theoretical argument a
  reviewer must accept: "CTRNN reaches X% against a best-static-baseline of Y%."

## 4. System design

```
                        generate.py
                   (programmatic renderer)
                             │
             ┌───────────────┼───────────────┐
             │               │               │
        family=balanced  family=classic   render=outline
        (primary)        (diagnostic)     (ink-matched)
             │               │               │
             └───────────────┼───────────────┘
                             │
                    TAMCorrespondenceTask          TAMTask  (FROZEN)
                        (envs.py)                  2023 replication
                             │
                    ┌────────┴────────┐
                    │  train / eval   │  <- training.py, +metrics
                    └────────┬────────┘
                             │
        ┌──────────┬─────────┼─────────┬──────────┬──────────┐
      CTRNN      GRUNet   FFStack  CTRNN-shuf   FF2Only   FF1Only
      binding    binding  no-recur  order ctrl   static    static
        │          │         │         │           │          │
        └──────────┴─────────┴────┬────┴───────────┴──────────┘
                                  │
                             sweep.py
                     (seeds x arch x family)
                                  │
                    ┌─────────────┴─────────────┐
              local MPS (dev)          hf jobs uv run (final)
                                       scripts/hf_sweep.py
                                  │
                          results/*.npz + figures
```

Data flow for one trial, `dt = 50 ms`:

```
period     fixation   frame1    frame2 x4                decision
timestep   0    1     2         3    4    5    6         7    8
obs        fix  fix   shape     bar  bar  bar  bar       0    0
label      fix  fix   fix       d    d    d    d         d    d
                      └────────────── the binding ──────────────┘
                      shape position must be held across 4 steps
                      (CTRNN alpha = dt/tau = 0.5) to disambiguate
                      an otherwise label-neutral frame 2
```

## 5. Modules

### 5.1 `illusion_rnn/generate.py` (new)

Pure-numpy renderer. No file I/O, no PIL dependency for generation — stimuli are
drawn analytically so every parameter is recoverable.

Canvas `img_size × img_size` (default 64), float in `[0, 1]`, ink = 1 on
background = 0, matching the existing loader convention in `stimuli.py`.

| parameter | values | notes |
|---|---|---|
| `shape` | square, circle, triangle, cross, hexagon | 2 train + 3 held-out |
| `bar_span` | continuous | sampled first, independent of `d` |
| `bar_length` | continuous range | randomized per trial |
| `render` | `filled` \| `outline` | |
| `stroke_width` | 1–4 px | outline only |
| `ink_match` | `none` \| `energy` | `energy` scales outline intensity so total ink mass equals the filled render |
| `family` | `balanced` \| `classic` | |
| `direction` | left, right | 2AFC; chance 50% |

`family="classic"` reproduces the hand-made L-shape geometry
(two small squares → bar with one raised end) programmatically at scale. Frame 2
*is* label-readable there by construction. It is the diagnostic that quantifies
how much of the 2023 result was static classification.

Public surface:

```python
render_trial(rng, **params) -> TrialStimulus(frame1, frame2, label, params)
sample_params(rng, split, **constraints) -> dict
ink_mass(frame) -> float
```

### 5.2 `illusion_rnn/envs.py` (extend)

Add `TAMCorrespondenceTask(TrialEnv)`. `TAMTask` is untouched.

Timing: `fixation` 100 ms → `frame1` 50 ms → `frame2` 4 × 50 ms →
`decision` 100 ms, `dt = 50`. Configurable `isi` (default 0) between frames for
a later ISI manipulation.

Label map reuses the frozen `TAM_CHOICES` indices (`fixation=0, left=1,
right=3`) and keeps the action space at `Discrete(6)` with the frozen names,
sampling only `{left, right}` as ground truth. Sharing the action space means
the abstention metric, plotting, and the analysis suite mean the same thing
across both tasks. Ground truth is `fixation` through frame 1 and the direction
from frame 2 onward — identical in structure to `TAMTask`.

The env holds a `StimulusSampler` bound to a split, not a pre-loaded dict.

**Known neurogym 2.x trap** (from the 2026-08-13 port): `reset()` internally
consumes the trial's first timestep. The env's tests must assert the observed
period structure after a reset, not the requested one.

### 5.3 `illusion_rnn/models.py` (extend)

All models keep the existing contract:
`model(x: (T, B, F)) -> (out: (T, B, n_actions), activity: (T, B, H))`.

| class | input | ceiling | role |
|---|---|---|---|
| `RNNNet` (exists) | all frames, ordered | 100% | the binding model |
| `GRUNet` | all frames, ordered | 100% | recurrence-type control |
| `FFStack` | all frames concatenated | 100% | has both frames, no recurrence — isolates whether *recurrence* matters or merely *access* |
| `FF2Only` | frame 2 only | **50%, provable** | the static baseline |
| `FF1Only` | frame 1 only | ~55%, measured | the other static baseline |

`CTRNN-shuffled` is not a class but a training flag: frames are permuted in time
per trial. Because reversing frame order flips the correct label, its ceiling is
**50%** — the cleanest control in the set.

Feedforward models return a constant-across-time `activity` tensor to satisfy
the shared contract, so `evaluate()` and the analysis suite work unchanged.

### 5.4 `illusion_rnn/sweep.py` (new)

Grid runner. One cell = `(architecture, family, hidden_size, seed)`. Produces a
tidy record per cell, aggregated to mean ± 95% CI across seeds.

Seeds control weight init, trial sampling, and shuffling independently, so a
seed is reproducible end-to-end from an integer.

### 5.5 `scripts/hf_sweep.py` (new)

PEP-723 UV script for `hf jobs uv run --flavor l4x1`. Installs the package from
the public GitHub repo (no data upload — stimuli are generated from a seed
inside the job), runs the grid, writes results back. Local MPS is used for
development; HF only for final sweeps.

## 6. Metrics

Reporting accuracy alone is what hid the outline abstention. Every evaluation
returns, and every results file records:

- `accuracy` — argmax at the final timestep against ground truth
- `abstention_rate` — fraction answering the fixation class
- `committed_accuracy` — accuracy over non-abstaining trials only
- `confusion` — full matrix over the action space

These are retro-fitted to the existing `evaluate()` in `training.py`, so the
2023 outline number is reported correctly wherever it appears.

## 7. Held-out splits

Three orthogonal axes, each a separate test set with its own train restriction.
Splits are defined on generator *parameters*, so disjointness is checkable
rather than hoped for.

| split | train | test | question |
|---|---|---|---|
| `position` | bar centers in middle 60% of track | outer 40% | spatial generalization |
| `shape` | square, circle | triangle, cross, hexagon | form generalization |
| `style` | filled | outline, energy-matched | the original 2023 question, decontaminated |

## 8. Verification checkpoints

Each implementation step ends with a check that fails loudly. Failure at any
checkpoint voids everything downstream, so they run in order.

| # | after | check | pass condition |
|---|---|---|---|
| 1 | generator + env | per-pixel mutual information between frame 2 and the label over 10k trials, plus a sample-trial figure | max per-pixel MI below the null distribution's 95th percentile; no visible label cue |
| 2 | baselines | train `FF2Only` on `balanced` | **50 ± 2%** — anything higher means the generator leaks and the experiment is void |
| 3 | splits | parameter-set intersection between train and each test set | empty |
| 4 | sweep | `RNNNet` vs `FF2Only`, 5 seeds | non-overlapping 95% CIs |
| 5 | outline | accuracy as a function of ink-mass ratio | a curve, not a single number |

Checkpoint 2 is the load-bearing one. It is the empirical version of the
provability argument in §3, and it is what the 2023 experiment lacked.

## 9. Testing

Per project convention (`CLAUDE.md`): tests accompany every module, nothing
fails silently.

- **generator** — label correctness against rendered geometry; frame-2 ⊥ label
  as a statistical test; ink-mass matching within tolerance; split disjointness;
  every parameter combination renders without error
- **envs** — period structure, timing, frozen label map, `frame1 != frame2`,
  reproducibility from a seed
- **models** — I/O contract for each class; `FFStack` cannot see frame order;
  `FF2Only` genuinely receives only frame 2
- **sweep** — a 2-seed 2-cell grid runs end to end and produces the record schema
- **metrics** — abstention and committed-accuracy on hand-built trial lists with
  known answers
- **regression** — the existing 101 tests keep passing; `TAMTask` behavior is
  byte-identical

## 10. Repo integration

- `TAMTask`, the packaged JPG stimuli, and the four shipped checkpoints are
  frozen. The 2023 replication keeps working.
- **Step 0, approved separately:** correct the `TAM_basic` misstatement in
  `README.md`, `checkpoints/MANIFEST.md`, `results/meta.json`, and
  `notebooks/04_rsa.ipynb` before building the replacement, and report the
  outline result with the abstention split. No new science; removes a false
  claim from a public repo.
- New code is additive. No existing public function changes signature except
  `evaluate()`, which gains fields on its return dataclass.

## 11. Open questions

None blocking. Deferred by decision:

- Periodic-canvas variant for an exactly-balanced frame-1 floor — build only if
  the measured ~55% floor draws reviewer objection.
- 4-way direction task (left/right/up/down). The 2AFC statement is cleaner
  because every static baseline sits at chance; the 4-way version has a 50%
  frame-2 floor (it reveals the axis but not the sign) that needs explaining.
- Human psychophysics anchor — the step after this one.
