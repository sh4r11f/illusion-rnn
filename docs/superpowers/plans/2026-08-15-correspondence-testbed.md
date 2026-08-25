# Correspondence Testbed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TAM task whose direction label depends on the relation between frame 1 and frame 2 rather than on either frame alone, plus the baselines and multi-seed sweep needed to prove a recurrent model is binding the two frames rather than classifying one.

**Architecture:** A numpy-only programmatic stimulus generator (`generate.py`) feeds a new neurogym environment (`TAMCorrespondenceTask`). Frame 2's geometry is sampled *before and independently of* the direction label, which makes the frame-2-only baseline provably 50%. Four new model classes provide static and no-recurrence controls against the existing `RNNNet`. A grid runner (`sweep.py`) runs seeds × architectures locally on MPS for development and on HF Jobs for the final sweep. The existing `TAMTask`, its packaged JPGs, and the four shipped checkpoints are frozen so the archived 2023 result keeps reproducing.

**Tech Stack:** Python ≥3.10, numpy, PyTorch ≥2.0, neurogym 2.3.1, pytest, `hf jobs uv run` on `l4x1`.

**Spec:** `docs/superpowers/specs/2026-08-15-correspondence-testbed-design.md`

## Global Constraints

- **Nothing fails silently.** Every invalid argument raises with a message naming the bad value and the allowed set, following the existing `_validate` pattern in `illusion_rnn/envs.py:42-46`.
- **Every module gets tests.** Project convention; the existing suite is 101 tests and must stay green.
- **numpy-only in `generate.py` and `analysis.py`.** scipy is not a project dependency and must not become one.
- **Frozen label map.** `TAM_CHOICES = {"fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5}`. New code reuses these indices and keeps the action space at `Discrete(6)`.
- **Frozen artifacts.** `TAMTask`, `MotionTask`, `illusion_rnn/stimuli_data/**`, and `checkpoints/*.pt` are not modified. Only `evaluate()` changes signature, and only by gaining fields on its return dataclass.
- **Ink convention.** All frames are float arrays in `[0, 1]`, ink = 1 on background = 0, matching `illusion_rnn/stimuli.py:39-44`.
- **Checkpoint loading.** Shipped checkpoints hold CUDA tensors; any `torch.load` needs `map_location="cpu"`.
- **neurogym 2.x trap.** `reset()` internally consumes the trial's first timestep. Assert observed period structure, not requested.
- **Commit style.** Conventional-commit subject, body explaining why, and the two trailers used in this repo.
- **Run tests with** `/opt/anaconda3/bin/conda`-free tooling: this repo uses `.venv`, so `.venv/bin/python -m pytest` or `uv run pytest`.

---

### Task 1: Evaluation metrics — abstention, committed accuracy, confusion

Reporting accuracy alone is what hid the 2023 outline result: the shipped RNN answers the fixation class on 78% of outline trials, which reads as below-chance confusion when collapsed into "accuracy 0.100".

**Files:**
- Modify: `illusion_rnn/training.py:103-141` (`EvalResult`, `evaluate`)
- Modify: `illusion_rnn/__init__.py` (no new export needed; `EvalResult` already exported)
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `EvalResult` gains `abstention_rate: float`, `committed_accuracy: float`, `confusion: np.ndarray` of shape `(n_actions, n_actions)` indexed `[ground_truth, choice]`. `evaluate()` gains no parameters. Tasks 10, 11, 12, 13 read these fields.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_training.py
import numpy as np
from illusion_rnn.training import EvalResult, summarize_trials


def test_summarize_trials_counts_abstention_separately():
    """A model answering the fixation class is abstaining, not answering wrong.

    Ground truth is never `fixation` at the decision step, so any fixation
    choice is a refusal to commit. Folding those into `accuracy` is what made
    the 2023 outline number (0.100) read as below-chance confusion.
    """
    trials = [
        {"ground_truth": 1, "choice": 1, "correct": True},   # correct
        {"ground_truth": 1, "choice": 3, "correct": False},  # wrong direction
        {"ground_truth": 3, "choice": 0, "correct": False},  # abstained
        {"ground_truth": 3, "choice": 0, "correct": False},  # abstained
    ]
    acc, abstention, committed, confusion = summarize_trials(trials, n_actions=6)
    assert acc == 0.25                      # 1 of 4
    assert abstention == 0.5                # 2 of 4 answered fixation
    assert committed == 0.5                 # 1 of the 2 that committed
    assert confusion[1, 1] == 1
    assert confusion[1, 3] == 1
    assert confusion[3, 0] == 2
    assert confusion.sum() == 4


def test_summarize_trials_all_abstained_gives_nan_committed_accuracy():
    """Committed accuracy over zero committed trials must be NaN, not 0.0.

    Returning 0.0 would silently claim the model got everything wrong when in
    fact it answered nothing.
    """
    trials = [{"ground_truth": 1, "choice": 0, "correct": False}]
    acc, abstention, committed, _ = summarize_trials(trials, n_actions=6)
    assert acc == 0.0
    assert abstention == 1.0
    assert np.isnan(committed)


def test_eval_result_exposes_new_fields():
    result = EvalResult(
        accuracy=0.5, abstention_rate=0.25,
        committed_accuracy=0.667, confusion=np.zeros((6, 6)),
    )
    assert result.abstention_rate == 0.25
    assert result.committed_accuracy == 0.667
    assert result.confusion.shape == (6, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_training.py -k "summarize or new_fields" -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_trials'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/training.py -- add above EvalResult

def summarize_trials(trials: list, n_actions: int, abstain_action: int = 0):
    """Reduce per-trial records to (accuracy, abstention_rate, committed_accuracy,
    confusion).

    `abstain_action` is the fixation class. Ground truth is never fixation at the
    decision step, so choosing it is a refusal to commit rather than a wrong
    answer -- the two are reported separately because collapsing them hides the
    difference between "guessed wrong" and "never left the null state".

    `committed_accuracy` is NaN (not 0.0) when nothing was committed, so an
    all-abstain run cannot be misread as an all-wrong run.
    """
    n = len(trials)
    correct = sum(bool(t["correct"]) for t in trials)
    abstained = sum(t["choice"] == abstain_action for t in trials)
    committed_n = n - abstained

    confusion = np.zeros((n_actions, n_actions), dtype=int)
    for t in trials:
        confusion[t["ground_truth"], t["choice"]] += 1

    accuracy = correct / n if n else float("nan")
    abstention_rate = abstained / n if n else float("nan")
    committed_accuracy = correct / committed_n if committed_n else float("nan")
    return accuracy, abstention_rate, committed_accuracy, confusion
```

```python
# illusion_rnn/training.py -- replace the EvalResult dataclass

@dataclass
class EvalResult:
    """Result of ``evaluate``.

    ``accuracy`` counts abstentions as errors (the historical definition, kept
    so existing numbers stay comparable). ``abstention_rate`` and
    ``committed_accuracy`` separate the two failure modes; ``confusion`` is
    indexed ``[ground_truth, choice]``.
    """

    accuracy: float
    abstention_rate: float = float("nan")
    committed_accuracy: float = float("nan")
    confusion: np.ndarray | None = None
    trials: list = field(default_factory=list)
    activity: list = field(default_factory=list)
```

```python
# illusion_rnn/training.py -- replace the tail of evaluate()
    accuracy, abstention_rate, committed_accuracy, confusion = summarize_trials(
        trials, n_actions=env.action_space.n,
    )
    return EvalResult(
        accuracy=accuracy,
        abstention_rate=abstention_rate,
        committed_accuracy=committed_accuracy,
        confusion=confusion,
        trials=trials,
        activity=activity,
    )
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 104 tests (101 existing + 3 new). The existing tests construct `EvalResult(accuracy=..., trials=...)` positionally or by keyword; the new fields have defaults so they keep working.

- [ ] **Step 5: Verify against the shipped checkpoint**

Run:
```bash
.venv/bin/python -c "
from illusion_rnn import make_env, load_rnn, evaluate
m = load_rnn('checkpoints/rnn-pixel_h2048_tam-horiz.pt')
e = make_env('tam', box_shape='square', variant='outline', img_size=64); e.seed(0)
r = evaluate(m, e, n_trials=100, device='cpu')
print(f'acc={r.accuracy:.3f} abstain={r.abstention_rate:.3f} committed={r.committed_accuracy:.3f}')
"
```
Expected: abstention rate well above 0.5, confirming the metric reproduces the review finding on real data.

- [ ] **Step 6: Commit**

```bash
git add illusion_rnn/training.py tests/test_training.py
git commit -m "$(cat <<'EOF'
feat: report abstention and committed accuracy from evaluate()

Accuracy alone conflates "guessed the wrong direction" with "never left
the fixation state". On outline TAM the shipped RNN answers the fixation
class on 78% of trials, so its 0.100 accuracy reads as below-chance
confusion when it is actually an abstention.

summarize_trials() splits the two and returns the full confusion matrix.
committed_accuracy is NaN rather than 0.0 when nothing was committed, so
an all-abstain run cannot be misread as all-wrong.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 2: Step 0 — correct the `TAM_basic` misstatement

All 24 files in `TAM_basic/` are byte-identical to files in `TAM_task/`. The repo is public and currently reports testing-on-training-data as a generalization result.

**Files:**
- Modify: `README.md:70-77` (the generalization section), `README.md:79-96` (Analyses)
- Modify: `checkpoints/MANIFEST.md` (the eval column of the pixel checkpoint row)
- Modify: `results/meta.json` (regenerated, not hand-edited)
- Modify: `scripts/run_analyses.py` (drop `basic` from the default variant list, or mark it)
- Create: `tests/test_stimuli_integrity.py`
- Test: `tests/test_stimuli_integrity.py`

**Interfaces:**
- Consumes: `EvalResult.abstention_rate` and `.committed_accuracy` from Task 1.
- Produces: nothing later tasks depend on. This task is self-contained.

- [ ] **Step 1: Write the failing test that pins the finding**

```python
# tests/test_stimuli_integrity.py
"""Guards against the variant-contamination defect found on 2026-08-15.

TAM_basic was shipped and documented as a held-out generalization variant, but
every one of its 24 images is a byte-identical copy of a TAM_task image, so
"transfer to basic = 0.967" was measured on training data. These tests record
which variant pairs are genuinely disjoint so the mistake cannot recur silently.
"""
import hashlib
from pathlib import Path

import pytest

from illusion_rnn.stimuli import _default_source


def _hashes(dirname: str) -> set[str]:
    src = _default_source(dirname)
    return {
        hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(Path(src).glob("*.jpg"))
    }


def test_outline_is_genuinely_disjoint_from_standard():
    """outline IS a real held-out variant -- no shared images."""
    assert not (_hashes("TAM_outline") & _hashes("TAM_task"))


def test_basic_is_documented_as_a_subset_not_a_variant():
    """basic is NOT held out: every image also appears in the standard set.

    This assertion is deliberately phrased as the true state of the data. If
    someone regenerates TAM_basic as genuinely novel images, this test fails and
    forces the documentation to be revisited.
    """
    basic, standard = _hashes("TAM_basic"), _hashes("TAM_task")
    assert basic <= standard, (
        "TAM_basic is expected to be a strict subset of TAM_task; if it is no "
        "longer, README/MANIFEST claims about it must be re-checked"
    )
```

- [ ] **Step 2: Run to verify the tests pass against current data**

Run: `.venv/bin/python -m pytest tests/test_stimuli_integrity.py -v`
Expected: PASS both. These pin the finding rather than drive new code.

- [ ] **Step 3: Regenerate `results/meta.json` with corrected reporting**

Run:
```bash
.venv/bin/python scripts/run_analyses.py
```
Then confirm `results/meta.json` no longer presents `basic` accuracy as transfer. If `run_analyses.py` hardcodes `["standard", "outline", "basic"]`, change that list to `["standard", "outline"]` and add a comment naming why.

- [ ] **Step 4: Rewrite the three documentation claims**

In `README.md`, replace the "The generalization result" section body with text stating: the reference RNN transfers to outline at 0.100 accuracy with a 78% abstention rate; `basic` is not a held-out variant and is excluded. In `checkpoints/MANIFEST.md`, change the pixel checkpoint's eval cell from `standard mean 0.967 ..., outline 0.100, basic 0.967` to report standard and outline only, with the abstention figure, and add a line under Provenance recording that `TAM_basic` duplicates `TAM_task`. Regenerate `figures/generalization.png` from the corrected numbers.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 106 tests.

- [ ] **Step 6: Commit**

```bash
git add README.md checkpoints/MANIFEST.md results/meta.json scripts/run_analyses.py figures/generalization.png tests/test_stimuli_integrity.py
git commit -m "$(cat <<'EOF'
fix: stop reporting TAM_basic as a generalization result

All 24 images in TAM_basic are byte-identical to images in TAM_task (21
share filenames, 3 are internal duplicates), so the documented "transfer
to basic = 0.967" was measured on training data. The RSA claim that basic
sits inside standard's representational family is true for the same
reason and equally uninformative.

Removes basic from the transfer reporting, restates the outline result
with its 78% abstention rate, and adds tests pinning which variant pairs
are genuinely disjoint so this cannot recur silently.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 3: Shape rendering primitives

**Files:**
- Create: `illusion_rnn/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SHAPE_NAMES: tuple[str, ...]`, `shape_mask(shape: str, size: int) -> np.ndarray[bool]`, `erode(mask, radius) -> np.ndarray[bool]`, `outline_mask(mask, stroke_width) -> np.ndarray[bool]`, `ink_mass(frame) -> float`. Task 4 calls `shape_mask` and `outline_mask`; Task 13 calls `ink_mass`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py
import numpy as np
import pytest

from illusion_rnn.generate import (
    SHAPE_NAMES, erode, ink_mass, outline_mask, shape_mask,
)


@pytest.mark.parametrize("shape", SHAPE_NAMES)
def test_shape_mask_is_nonempty_and_fits_its_box(shape):
    m = shape_mask(shape, 16)
    assert m.shape == (16, 16)
    assert m.dtype == bool
    assert m.any(), f"{shape} rendered empty"
    assert not m.all() or shape == "square", f"{shape} filled its whole box"


@pytest.mark.parametrize("shape", SHAPE_NAMES)
def test_shape_mask_is_left_right_symmetric(shape):
    """Every shape is symmetric about the vertical axis, so shape identity
    cannot leak the left/right direction label."""
    m = shape_mask(shape, 16)
    np.testing.assert_array_equal(m, m[:, ::-1])


def test_shapes_are_distinguishable_from_each_other():
    masks = {s: shape_mask(s, 24) for s in SHAPE_NAMES}
    for a in SHAPE_NAMES:
        for b in SHAPE_NAMES:
            if a < b:
                assert not np.array_equal(masks[a], masks[b]), f"{a} == {b}"


def test_shape_mask_rejects_unknown_shape():
    with pytest.raises(ValueError, match="Unknown shape 'blob'"):
        shape_mask("blob", 16)


def test_erode_shrinks_a_solid_block_by_the_radius():
    m = np.zeros((11, 11), dtype=bool)
    m[2:9, 2:9] = True          # 7x7 block
    e = erode(m, 1)
    assert e.sum() == 25        # shrinks to 5x5
    assert e[3:8, 3:8].all()


def test_erode_removes_a_block_thinner_than_the_radius():
    """Erosion must not silently keep pixels it cannot support."""
    m = np.zeros((11, 11), dtype=bool)
    m[5, 2:9] = True            # 1px-tall line
    assert not erode(m, 1).any()


def test_outline_mask_is_hollow_and_inside_the_original():
    m = shape_mask("square", 20)
    o = outline_mask(m, stroke_width=2)
    assert o.sum() < m.sum(), "outline must be lighter than the filled shape"
    assert (m | o == m).all(), "outline must lie inside the filled shape"
    assert not o[8:12, 8:12].any(), "centre must be hollow"


def test_outline_stroke_width_monotonically_adds_ink():
    m = shape_mask("circle", 32)
    masses = [outline_mask(m, w).sum() for w in (1, 2, 3, 4)]
    assert masses == sorted(masses), "thicker strokes must not lose ink"


def test_ink_mass_sums_pixel_values():
    frame = np.zeros((4, 4))
    frame[0, :2] = 0.5
    assert ink_mass(frame) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'illusion_rnn.generate'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/generate.py
"""Programmatic stimulus generator for the correspondence TAM task.

Stimuli are drawn analytically rather than loaded from disk, so every geometric
parameter of a trial is recoverable and the train/test splits can be defined on
parameters instead of on filenames. numpy only -- scipy is not a project
dependency.

Ink convention matches ``illusion_rnn.stimuli``: float arrays in [0, 1] with
ink = 1 on background = 0.
"""

import numpy as np

SHAPE_NAMES = ("square", "circle", "triangle", "cross", "hexagon")


def shape_mask(shape: str, size: int) -> np.ndarray:
    """Boolean mask of ``shape`` inscribed in a ``size`` x ``size`` box.

    Every shape is symmetric about the vertical axis so that shape identity
    cannot correlate with the left/right direction label.
    """
    if shape not in SHAPE_NAMES:
        msg = f"Unknown shape {shape!r}; expected one of {SHAPE_NAMES}"
        raise ValueError(msg)

    yy, xx = np.mgrid[0:size, 0:size]
    half = (size - 1) / 2.0
    # normalized box coordinates in [-1, 1]; v increases downward
    u = (xx - half) / half
    v = (yy - half) / half

    if shape == "square":
        return np.ones((size, size), dtype=bool)
    if shape == "circle":
        return u**2 + v**2 <= 1.0
    if shape == "triangle":
        # apex at top centre, base along the bottom edge
        return np.abs(u) <= (v + 1.0) / 2.0
    if shape == "cross":
        return (np.abs(u) <= 1 / 3) | (np.abs(v) <= 1 / 3)
    # hexagon: intersection of three slabs
    root3 = np.sqrt(3.0)
    return (
        (np.abs(v) <= root3 / 2)
        & (np.abs(root3 * u + v) <= root3)
        & (np.abs(root3 * u - v) <= root3)
    )


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion with a square structuring element of ``radius``.

    A pixel survives only if every pixel within +/- radius on both axes is also
    set. Implemented as an AND over shifted copies so the module stays
    numpy-only. Shifts are slice-assigned rather than rolled, so the image
    border erodes away instead of wrapping around.
    """
    if radius < 1:
        msg = f"radius must be >= 1, got {radius}"
        raise ValueError(msg)

    height, width = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys_dst = slice(max(0, dy), height - max(0, -dy))
            ys_src = slice(max(0, -dy), height - max(0, dy))
            xs_dst = slice(max(0, dx), width - max(0, -dx))
            xs_src = slice(max(0, -dx), width - max(0, dx))
            shifted[ys_dst, xs_dst] = mask[ys_src, xs_src]
            out &= shifted
    return out


def outline_mask(mask: np.ndarray, stroke_width: int = 1) -> np.ndarray:
    """Hollow outline of ``mask``: the filled shape minus its erosion."""
    return mask & ~erode(mask, stroke_width)


def ink_mass(frame: np.ndarray) -> float:
    """Total ink in a frame -- the sum of its pixel values."""
    return float(np.asarray(frame).sum())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/generate.py tests/test_generate.py
git commit -m "$(cat <<'EOF'
feat: add numpy-only shape rendering primitives

Five vertically-symmetric shapes rendered analytically into a bounding
box, plus binary erosion and hollow-outline derivation. Symmetry is
asserted in tests because an asymmetric shape would let shape identity
correlate with the left/right direction label.

Erosion is an AND over slice-shifted copies rather than np.roll, so the
image border erodes away instead of wrapping, and scipy stays out of the
dependency list.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 4: Trial sampling and the balanced family

This task implements the load-bearing property of the whole project: frame 2 is a function of `(bar_left, bar_length, shape_size)` only, and those are drawn before and independently of `direction`.

**Files:**
- Modify: `illusion_rnn/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `shape_mask`, `outline_mask`, `ink_mass` from Task 3.
- Produces: `DIRECTIONS = ("left", "right")`; `TrialStimulus` frozen dataclass with fields `frame1: np.ndarray`, `frame2: np.ndarray`, `label: str`, `params: dict`; `sample_params(rng, img_size=64, **overrides) -> dict`; `render_trial(params, img_size=64) -> TrialStimulus`. Tasks 5, 6, 7, 8 all call these.

The parameter dict has exactly these keys: `shape`, `shape_size`, `bar_left`, `bar_length`, `direction`, `family`, `transform`, `render`, `stroke_width`, `ink_match`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py -- append
from illusion_rnn.generate import DIRECTIONS, TrialStimulus, render_trial, sample_params


def _params(**overrides):
    base = dict(
        shape="square", shape_size=8, bar_left=10, bar_length=24,
        direction="right", family="balanced", transform="growth",
        render="filled", stroke_width=1, ink_match="none",
    )
    base.update(overrides)
    return base


def test_frame2_is_bit_identical_across_directions():
    """THE load-bearing invariant of the design.

    Frame 2 is a function of (bar_left, bar_length, shape_size) only. If this
    ever fails, the frame-2-only baseline is above chance and the whole
    experiment is void.
    """
    right = render_trial(_params(direction="right"))
    left = render_trial(_params(direction="left"))
    np.testing.assert_array_equal(right.frame2, left.frame2)


def test_frame1_differs_across_directions():
    right = render_trial(_params(direction="right"))
    left = render_trial(_params(direction="left"))
    assert not np.array_equal(right.frame1, left.frame1)


def test_frame1_sits_at_the_end_the_bar_grew_from():
    """Rightward growth starts at the bar's left edge, and vice versa."""
    right = render_trial(_params(direction="right"))
    left = render_trial(_params(direction="left"))
    right_cols = np.flatnonzero(right.frame1.any(axis=0))
    left_cols = np.flatnonzero(left.frame1.any(axis=0))
    assert right_cols.min() == 10                  # bar_left
    assert left_cols.max() == 10 + 24 - 1          # bar_left + bar_length - 1


def test_frame1_lies_inside_the_frame2_bar():
    """The shape is absorbed into the bar -- that is the transformation."""
    t = render_trial(_params())
    assert ((t.frame1 > 0) & (t.frame2 == 0)).sum() == 0


def test_frames_respect_the_ink_convention():
    t = render_trial(_params())
    for frame in (t.frame1, t.frame2):
        assert frame.dtype == np.float64
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0
        assert frame.shape == (64, 64)


def test_label_matches_the_requested_direction():
    assert render_trial(_params(direction="left")).label == "left"
    assert render_trial(_params(direction="right")).label == "right"


def test_render_trial_rejects_a_bar_that_runs_off_canvas():
    with pytest.raises(ValueError, match="does not fit"):
        render_trial(_params(bar_left=50, bar_length=30), img_size=64)


def test_render_trial_rejects_a_shape_longer_than_its_bar():
    with pytest.raises(ValueError, match="shape_size"):
        render_trial(_params(shape_size=30, bar_length=24))


def test_sample_params_draws_frame2_geometry_independently_of_direction():
    """Statistical form of the invariant: across many samples, the bar geometry
    distribution must not differ by label."""
    rng = np.random.default_rng(0)
    rows = [sample_params(rng) for _ in range(4000)]
    left = np.array([(r["bar_left"], r["bar_length"]) for r in rows
                     if r["direction"] == "left"], dtype=float)
    right = np.array([(r["bar_left"], r["bar_length"]) for r in rows
                      if r["direction"] == "right"], dtype=float)
    # means must agree to well within sampling noise
    assert np.allclose(left.mean(axis=0), right.mean(axis=0), rtol=0.05)


def test_sample_params_is_reproducible_from_a_seed():
    a = [sample_params(np.random.default_rng(7)) for _ in range(3)]
    b = [sample_params(np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_outline_render_has_less_ink_than_filled():
    filled = render_trial(_params(render="filled"))
    outline = render_trial(_params(render="outline", stroke_width=1))
    assert ink_mass(outline.frame1) < ink_mass(filled.frame1)


def test_energy_matching_equalizes_ink_mass():
    filled = render_trial(_params(render="filled"))
    matched = render_trial(_params(render="outline", ink_match="energy"))
    assert ink_mass(matched.frame1) == pytest.approx(ink_mass(filled.frame1), rel=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_trial'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/generate.py -- append

from dataclasses import dataclass

DIRECTIONS = ("left", "right")
RENDERS = ("filled", "outline")
INK_MATCHES = ("none", "energy")
FAMILIES = ("balanced", "classic")
TRANSFORMS = ("growth", "shrink")


@dataclass(frozen=True)
class TrialStimulus:
    """One rendered trial: two frames, the direction label, and the exact
    parameters that produced them (so splits can be defined on parameters)."""

    frame1: np.ndarray
    frame2: np.ndarray
    label: str
    params: dict


def _validate_choice(name, value, allowed):
    if value not in allowed:
        msg = f"Unknown {name} {value!r}; expected one of {tuple(allowed)}"
        raise ValueError(msg)


def _render_element(mask: np.ndarray, params: dict, reference_mass: float | None):
    """Apply the render style to a boolean mask, returning a float frame patch.

    ``ink_match="energy"`` rescales an outline's intensity so its total ink
    equals ``reference_mass`` (the filled render's). Values may exceed 1.0; that
    is deliberate and documented -- clipping would break the match, and the
    alternative (thickening the stroke) is available separately via
    ``stroke_width``.
    """
    if params["render"] == "filled":
        return mask.astype(float)

    stroke = outline_mask(mask, params["stroke_width"]).astype(float)
    if params["ink_match"] == "none":
        return stroke
    stroke_mass = stroke.sum()
    if stroke_mass == 0:
        msg = (
            f"outline of shape {params['shape']!r} at size "
            f"{params['shape_size']} with stroke_width "
            f"{params['stroke_width']} is empty; cannot energy-match"
        )
        raise ValueError(msg)
    return stroke * (reference_mass / stroke_mass)


def render_trial(params: dict, img_size: int = 64) -> TrialStimulus:
    """Render one trial from a complete parameter dict.

    Frame 2 is built from ``bar_left``, ``bar_length`` and ``shape_size`` alone
    -- never from ``direction``. That is what makes the frame-2-only baseline
    provably 50%, and ``test_frame2_is_bit_identical_across_directions`` pins it.
    """
    _validate_choice("shape", params["shape"], SHAPE_NAMES)
    _validate_choice("direction", params["direction"], DIRECTIONS)
    _validate_choice("render", params["render"], RENDERS)
    _validate_choice("ink_match", params["ink_match"], INK_MATCHES)

    size = params["shape_size"]
    left = params["bar_left"]
    length = params["bar_length"]

    if size > length:
        msg = (
            f"shape_size {size} exceeds bar_length {length}; the shape must fit "
            f"inside the bar it grows into"
        )
        raise ValueError(msg)
    if left < 0 or left + length > img_size:
        msg = (
            f"bar spanning [{left}, {left + length}) does not fit a "
            f"{img_size}px canvas"
        )
        raise ValueError(msg)

    centre = img_size // 2
    top = centre - size // 2
    rows = slice(top, top + size)

    # --- frame 2: the bar. Depends only on (left, length, size).
    bar = np.ones((size, length), dtype=bool)
    reference_mass = float(bar.sum())
    frame2 = np.zeros((img_size, img_size))
    frame2[rows, left:left + length] = _render_element(bar, params, reference_mass)

    # --- frame 1: the shape, at the end the bar grew FROM.
    mask = shape_mask(params["shape"], size)
    shape_left = left if params["direction"] == "right" else left + length - size
    frame1 = np.zeros((img_size, img_size))
    frame1[rows, shape_left:shape_left + size] = _render_element(
        mask, params, float(mask.sum()),
    )

    if params["transform"] == "shrink":
        frame1, frame2 = frame2, frame1

    return TrialStimulus(
        frame1=frame1, frame2=frame2, label=params["direction"],
        params=dict(params),
    )


def sample_params(
    rng: np.random.Generator,
    img_size: int = 64,
    shapes: tuple = ("square", "circle"),
    bar_length_range: tuple = (16, 28),
    shape_size: int = 8,
    bar_centre_range: tuple[float, float] = (0.0, 1.0),
    family: str = "balanced",
    transform: str = "growth",
    render: str = "filled",
    stroke_width: int = 1,
    ink_match: str = "none",
) -> dict:
    """Draw one trial's parameters.

    ORDER MATTERS. ``bar_length`` and ``bar_left`` are drawn first, from
    distributions that do not reference ``direction``; ``direction`` is drawn
    last. That ordering is what makes frame 2 independent of the label.

    ``bar_centre_range`` restricts the bar's centre to a fraction of the usable
    track, which is how the ``position`` train/test split is expressed.
    """
    _validate_choice("family", family, FAMILIES)
    _validate_choice("transform", transform, TRANSFORMS)

    # 1. bar geometry -- independent of the label
    length = int(rng.integers(bar_length_range[0], bar_length_range[1] + 1))
    span = img_size - length
    if span < 1:
        msg = f"bar_length {length} leaves no room on a {img_size}px canvas"
        raise ValueError(msg)
    low = int(np.floor(bar_centre_range[0] * span))
    high = int(np.ceil(bar_centre_range[1] * span))
    left = int(rng.integers(low, max(low + 1, high)))

    # 2. nuisance variables -- also independent of the label
    shape = str(rng.choice(shapes))

    # 3. the label, drawn last
    direction = str(rng.choice(DIRECTIONS))

    return dict(
        shape=shape, shape_size=shape_size, bar_left=left, bar_length=length,
        direction=direction, family=family, transform=transform,
        render=render, stroke_width=stroke_width, ink_match=ink_match,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: PASS, 23 tests.

- [ ] **Step 5: Measure the frame-1 leak analytically**

Run:
```bash
.venv/bin/python -c "
import numpy as np
from collections import Counter
from illusion_rnn.generate import sample_params
rng = np.random.default_rng(0)
rows = [sample_params(rng) for _ in range(200000)]
# a frame-1-only observer sees the shape's left edge; best strategy is the
# majority label at that position
by_pos = {}
for r in rows:
    p = r['bar_left'] if r['direction'] == 'right' else r['bar_left'] + r['bar_length'] - r['shape_size']
    by_pos.setdefault(p, Counter())[r['direction']] += 1
best = sum(c.most_common(1)[0][1] for c in by_pos.values())
print(f'frame-1-only Bayes-optimal accuracy: {best/len(rows):.4f}')
"
```
Expected: a value near 0.55. Record it in the commit message — this is the measured floor the spec promises, and Task 10 checks a trained probe against it.

- [ ] **Step 6: Commit**

```bash
git add illusion_rnn/generate.py tests/test_generate.py
git commit -m "$(cat <<'EOF'
feat: add trial sampling and the balanced stimulus family

Frame 2 is built from (bar_left, bar_length, shape_size) alone and those
are drawn before and independently of the direction label, which makes
the frame-2-only baseline provably 50%. The invariant is pinned by a test
asserting the two directions produce bit-identical frame 2 arrays -- if
that test ever fails the experiment is void.

The frame-1-only floor cannot also be driven to chance: on a bounded
canvas no non-empty balanced support exists (see spec section 3), so the
shape's position distribution differs slightly by label near the edges.
Randomizing bar length per trial smears it; the residual is measured
rather than assumed.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 5: The `classic` diagnostic family and the `growth+shrink` condition

`classic` reproduces the hand-made 2023 geometry at scale, where frame 2 *is* label-readable. It is the diagnostic that quantifies how much of the 0.967 was static classification. `growth+shrink` is the condition where temporal order is load-bearing.

**Files:**
- Modify: `illusion_rnn/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `render_trial`, `sample_params` from Task 4.
- Produces: `sample_params` accepts `family="classic"` and `transform="growth+shrink"`; `render_trial` handles both. No new function names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py -- append

def test_classic_family_frame2_IS_label_readable():
    """The diagnostic family deliberately leaks -- that is its purpose.

    It reproduces the 2023 geometry where a raised end marks the direction, so
    a frame-2-only model can solve it. Comparing model behaviour across the two
    families is what quantifies the original defect.
    """
    right = render_trial(_params(family="classic", direction="right"))
    left = render_trial(_params(family="classic", direction="left"))
    assert not np.array_equal(right.frame2, left.frame2)


def test_classic_family_frame1_is_constant_across_directions():
    """Matching the 2023 stimulus: frame 1 is two small squares either way."""
    right = render_trial(_params(family="classic", direction="right"))
    left = render_trial(_params(family="classic", direction="left"))
    np.testing.assert_array_equal(right.frame1, left.frame1)


def test_classic_raised_end_is_on_the_labelled_side():
    right = render_trial(_params(family="classic", direction="right"))
    heights = (right.frame2 > 0).sum(axis=0)
    cols = np.flatnonzero(heights)
    assert heights[cols[-1]] > heights[cols[len(cols) // 2]], \
        "rightward trials must raise the right end"


def test_shrink_transform_swaps_the_two_frames():
    grow = render_trial(_params(transform="growth"))
    shrink = render_trial(_params(transform="shrink"))
    np.testing.assert_array_equal(grow.frame1, shrink.frame2)
    np.testing.assert_array_equal(grow.frame2, shrink.frame1)


def test_growth_plus_shrink_makes_order_load_bearing():
    """In the growth+shrink condition the SAME frame pair appears in both
    orders with opposite labels, so a model that ignores order is at chance.

    Verified by construction: a growth trial labelled `right` and a shrink
    trial labelled `left` share a frame multiset.
    """
    grow = render_trial(_params(transform="growth", direction="right"))
    shrink = render_trial(_params(transform="shrink", direction="left"))
    # shrink-left retraces growth-left backwards, so build the matching pair
    grow_left = render_trial(_params(transform="growth", direction="left"))
    np.testing.assert_array_equal(shrink.frame1, grow_left.frame2)
    np.testing.assert_array_equal(shrink.frame2, grow_left.frame1)
    assert grow.label != shrink.label


def test_sample_params_growth_plus_shrink_draws_both_transforms():
    rng = np.random.default_rng(0)
    rows = [sample_params(rng, transform="growth+shrink") for _ in range(500)]
    seen = {r["transform"] for r in rows}
    assert seen == {"growth", "shrink"}


def test_sample_params_rejects_unknown_transform():
    with pytest.raises(ValueError, match="Unknown transform"):
        sample_params(np.random.default_rng(0), transform="teleport")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -k "classic or shrink or growth_plus" -v`
Expected: FAIL — `classic` currently renders the balanced geometry, and `sample_params` rejects `"growth+shrink"`.

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/generate.py -- update the constant and add the classic renderer

TRANSFORMS = ("growth", "shrink")
TRANSFORM_MODES = ("growth", "shrink", "growth+shrink")
CLASSIC_RAISE = 2.0   # raised end is this multiple of the bar's height


def _render_classic(params: dict, img_size: int) -> tuple:
    """The 2023 hand-made geometry, rendered analytically.

    frame 1: two small squares at the ends of the track (constant across
    directions). frame 2: a bar joining them with one end raised, on the side
    named by the label. Frame 2 is label-readable BY DESIGN -- this family is
    the diagnostic that measures how much of the 2023 result was static
    classification, not a task anyone should train on and call generalization.
    """
    size = params["shape_size"]
    left = params["bar_left"]
    length = params["bar_length"]
    centre = img_size // 2
    rows = slice(centre - size // 2, centre - size // 2 + size)

    mask = shape_mask(params["shape"], size)
    patch = _render_element(mask, params, float(mask.sum()))
    frame1 = np.zeros((img_size, img_size))
    frame1[rows, left:left + size] = patch
    frame1[rows, left + length - size:left + length] = patch

    raised = int(size * CLASSIC_RAISE)
    raised_top = centre - raised // 2
    frame2 = np.zeros((img_size, img_size))
    bar = np.ones((size, length), dtype=bool)
    frame2[rows, left:left + length] = _render_element(
        bar, params, float(bar.sum()),
    )
    end_left = left + length - size if params["direction"] == "right" else left
    block = np.ones((raised, size), dtype=bool)
    frame2[raised_top:raised_top + raised, end_left:end_left + size] = (
        _render_element(block, params, float(block.sum()))
    )
    return frame1, frame2
```

```python
# illusion_rnn/generate.py -- in render_trial, replace the frame-building body
# (everything from "centre = img_size // 2" down to the transform swap) with:

    if params["family"] == "classic":
        frame1, frame2 = _render_classic(params, img_size)
    else:
        centre = img_size // 2
        top = centre - size // 2
        rows = slice(top, top + size)

        bar = np.ones((size, length), dtype=bool)
        frame2 = np.zeros((img_size, img_size))
        frame2[rows, left:left + length] = _render_element(
            bar, params, float(bar.sum()),
        )

        mask = shape_mask(params["shape"], size)
        shape_left = left if params["direction"] == "right" else left + length - size
        frame1 = np.zeros((img_size, img_size))
        frame1[rows, shape_left:shape_left + size] = _render_element(
            mask, params, float(mask.sum()),
        )

    if params["transform"] == "shrink":
        frame1, frame2 = frame2, frame1
```

```python
# illusion_rnn/generate.py -- in sample_params, replace the transform validation
# and add the draw

    _validate_choice("transform", transform, TRANSFORM_MODES)
    ...
    # after `direction` is drawn:
    if transform == "growth+shrink":
        trial_transform = str(rng.choice(TRANSFORMS))
    else:
        trial_transform = transform
```

and pass `transform=trial_transform` into the returned dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: PASS, 30 tests.

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/generate.py tests/test_generate.py
git commit -m "$(cat <<'EOF'
feat: add the classic diagnostic family and the shrink transform

classic reproduces the 2023 hand-made geometry analytically: constant
frame 1, and a frame 2 whose raised end names the direction. It leaks by
design, and comparing a model's behaviour across the two families is what
quantifies how much of the original 0.967 was static classification.

growth+shrink makes temporal order load-bearing. The same frame pair
appears in both orders with opposite labels, so a frame-shuffled model is
at exactly 50%. The growth-only condition cannot deliver that, because
frame 1 (a shape) and frame 2 (a bar) are distinguishable by content --
once a model binds them, order adds nothing. The two conditions trade off
and the generator ships both rather than picking one.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 6: Train/test splits

**Files:**
- Modify: `illusion_rnn/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `sample_params` from Task 4.
- Produces: `Split` frozen dataclass with fields `name: str`, `shapes: tuple`, `bar_centre_range: tuple`, `render: str`, `ink_match: str`, `stroke_width: int`; `SPLITS: dict[str, Split]` with keys `"train"`, `"test_position"`, `"test_shape"`, `"test_style"`; `sampler_for(split: Split | str) -> Callable[[np.random.Generator], dict]`. Task 7 (the env) takes a `Split`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py -- append
from illusion_rnn.generate import SPLITS, Split, sampler_for


def test_the_four_splits_exist():
    assert set(SPLITS) == {"train", "test_position", "test_shape", "test_style"}


def test_train_and_test_shape_use_disjoint_shapes():
    assert not (set(SPLITS["train"].shapes) & set(SPLITS["test_shape"].shapes))


def test_train_and_test_position_use_disjoint_bar_centres():
    """Sampled bar positions must not overlap between the two splits."""
    train = sampler_for("train")
    test = sampler_for("test_position")
    rng = np.random.default_rng(0)
    train_pos = {train(rng)["bar_left"] for _ in range(3000)}
    test_pos = {test(rng)["bar_left"] for _ in range(3000)}
    assert not (train_pos & test_pos), sorted(train_pos & test_pos)[:10]


def test_train_and_test_style_use_disjoint_render_styles():
    assert SPLITS["train"].render != SPLITS["test_style"].render
    assert SPLITS["test_style"].ink_match == "energy"


def test_every_split_samples_both_labels():
    rng = np.random.default_rng(0)
    for name in SPLITS:
        labels = {sampler_for(name)(rng)["direction"] for _ in range(200)}
        assert labels == {"left", "right"}, name


def test_every_split_renders_without_error():
    rng = np.random.default_rng(0)
    for name in SPLITS:
        for _ in range(50):
            render_trial(sampler_for(name)(rng))


def test_sampler_for_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="Unknown split 'holdout'"):
        sampler_for("holdout")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -k split -v`
Expected: FAIL with `ImportError: cannot import name 'SPLITS'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/generate.py -- append

TRAIN_SHAPES = ("square", "circle")
HELDOUT_SHAPES = ("triangle", "cross", "hexagon")


@dataclass(frozen=True)
class Split:
    """A named restriction on the generator's parameter space.

    Splits are defined on parameters rather than on rendered images, so
    disjointness between train and each test set is checkable rather than
    hoped for.
    """

    name: str
    shapes: tuple = TRAIN_SHAPES
    bar_centre_range: tuple = (0.0, 0.6)
    render: str = "filled"
    ink_match: str = "none"
    stroke_width: int = 1


SPLITS = {
    # bar centres are disjoint by construction: train uses the lower 60% of the
    # usable track, test_position the upper 40%.
    "train": Split("train", bar_centre_range=(0.0, 0.6)),
    "test_position": Split("test_position", bar_centre_range=(0.6, 1.0)),
    "test_shape": Split("test_shape", shapes=HELDOUT_SHAPES),
    "test_style": Split(
        "test_style", render="outline", ink_match="energy", stroke_width=1,
    ),
}


def sampler_for(split, **overrides):
    """Return a ``rng -> params`` callable bound to ``split``.

    ``overrides`` pass through to ``sample_params`` so a caller can vary
    ``family``, ``transform`` or ``stroke_width`` without redefining a split.
    """
    if isinstance(split, str):
        if split not in SPLITS:
            msg = f"Unknown split {split!r}; expected one of {tuple(SPLITS)}"
            raise ValueError(msg)
        split = SPLITS[split]

    def sample(rng: np.random.Generator) -> dict:
        kwargs = dict(
            shapes=split.shapes,
            bar_centre_range=split.bar_centre_range,
            render=split.render,
            ink_match=split.ink_match,
            stroke_width=split.stroke_width,
        )
        kwargs.update(overrides)
        return sample_params(rng, **kwargs)

    return sample
```

Note: `test_train_and_test_position_use_disjoint_bar_centres` will fail on the first run because `sample_params` maps `bar_centre_range` onto `span = img_size - length`, and a long bar in the train range can reach the same `bar_left` as a short bar in the test range. Fix by making the range apply to an absolute pixel band rather than a length-dependent one:

```python
# illusion_rnn/generate.py -- in sample_params, replace the `left` draw

    # Map the centre range onto an ABSOLUTE pixel band so the bands do not
    # move with bar_length -- otherwise a long train bar and a short test bar
    # can land on the same bar_left and the splits silently overlap.
    usable = img_size - bar_length_range[1]
    low = int(np.floor(bar_centre_range[0] * usable))
    high = int(np.ceil(bar_centre_range[1] * usable))
    left = int(rng.integers(low, max(low + 1, high)))
    if left + length > img_size:
        left = img_size - length
```

The clamp in the last two lines would reintroduce overlap, so instead cap `bar_length` to what fits:

```python
    length = int(rng.integers(bar_length_range[0],
                              min(bar_length_range[1], img_size - high) + 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_generate.py -v`
Expected: PASS, 37 tests. If the disjointness test still fails, print the overlapping positions it reports and narrow the bands until empty — do not weaken the assertion.

- [ ] **Step 5: Re-verify the load-bearing invariant still holds under splits**

Run: `.venv/bin/python -m pytest tests/test_generate.py -k "frame2_is_bit_identical or independently_of_direction" -v`
Expected: PASS. The split changes touched the sampler, so this must be rechecked.

- [ ] **Step 6: Commit**

```bash
git add illusion_rnn/generate.py tests/test_generate.py
git commit -m "$(cat <<'EOF'
feat: add parameter-defined train/test splits

Three orthogonal held-out axes -- novel bar positions, novel shapes, and
energy-matched outline rendering -- each defined as a restriction on
generator parameters rather than on rendered images, so disjointness from
the training split is asserted rather than assumed.

Bar-position bands are absolute pixel ranges, not fractions of a
length-dependent span: with the fractional version a long training bar
and a short test bar could land on the same bar_left and the two splits
would silently overlap.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 7: `TAMCorrespondenceTask` environment

**Files:**
- Modify: `illusion_rnn/envs.py`
- Modify: `illusion_rnn/training.py:34-42` (`make_env` gains the `"correspondence"` task name)
- Modify: `illusion_rnn/__init__.py`
- Test: `tests/test_envs.py`

**Interfaces:**
- Consumes: `sampler_for`, `SPLITS`, `render_trial` from Tasks 4–6.
- Produces: `TAMCorrespondenceTask(dt=50, split="train", img_size=64, sigma=0.0, n_repeats=4, rewards=None, timing=None, **sampler_overrides)`. Exposes `frame1_index: int` and `frame2_indices: tuple[int, ...]` giving the observation timesteps of each frame *after* neurogym's reset consumes the first step — Task 9's `FF1Only`/`FF2Only` need these. `make_env("correspondence", ...)` constructs it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_envs.py -- append
from illusion_rnn.envs import TAMCorrespondenceTask


def _make_corr(**kwargs):
    defaults = dict(split="train", img_size=32)
    defaults.update(kwargs)
    env = TAMCorrespondenceTask(**defaults)
    env.seed(0)
    return env


def test_correspondence_trial_structure():
    """fixation 100 + frame1 50 + frame2 4x50 + decision 100 = 450ms at dt=50."""
    env = _make_corr()
    env.reset()
    assert env.ob.shape == (9, 32, 32)
    assert env.gt.shape == (9,)
    # fixation x2 + frame1 carry the fixation label; direction from frame2 on
    assert set(env.gt[:3]) == {TAM_CHOICES["fixation"]}
    assert set(env.gt[3:]) <= {TAM_CHOICES["left"], TAM_CHOICES["right"]}
    assert len(set(env.gt[3:])) == 1


def test_correspondence_action_space_is_the_frozen_six():
    env = _make_corr()
    assert env.action_space.n == 6
    assert env.choice_names == TAM_CHOICES


def test_frame_indices_locate_the_two_frames_in_the_observation():
    env = _make_corr()
    env.reset()
    f1 = env.ob[env.frame1_index]
    f2s = [env.ob[i] for i in env.frame2_indices]
    assert f1.any(), "frame1 index points at an empty observation"
    for f2 in f2s:
        np.testing.assert_array_equal(f2, f2s[0])   # frame2 is repeated
    assert not np.array_equal(f1, f2s[0])


def test_frame1_and_frame2_always_differ():
    env = _make_corr()
    for _ in range(50):
        env.new_trial()
        assert not np.array_equal(
            env.ob[env.frame1_index], env.ob[env.frame2_indices[0]],
        )


def test_correspondence_is_reproducible_from_a_seed():
    a, b = _make_corr(), _make_corr()
    for _ in range(10):
        a.new_trial(); b.new_trial()
        np.testing.assert_array_equal(a.ob, b.ob)
        np.testing.assert_array_equal(a.gt, b.gt)


def test_correspondence_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="Unknown split"):
        TAMCorrespondenceTask(split="nope")


@pytest.mark.parametrize("split", ["train", "test_position", "test_shape", "test_style"])
def test_every_split_builds_an_env_that_runs(split):
    env = _make_corr(split=split)
    for _ in range(20):
        env.new_trial()
    assert env.ob.shape == (9, 32, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_envs.py -k correspondence -v`
Expected: FAIL with `ImportError: cannot import name 'TAMCorrespondenceTask'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/envs.py -- append

from illusion_rnn.generate import SPLITS, render_trial, sampler_for


class TAMCorrespondenceTask(TrialEnv):
    """2AFC TAM task whose label depends on the frame1-frame2 relation.

    Unlike ``TAMTask``, frame 1 varies from trial to trial and frame 2 is drawn
    independently of the direction label. A model must bind the two frames to
    answer: frame 2 alone is provably uninformative, and frame 1 alone is
    near-uninformative (see the design spec, section 3).

    Trial structure: fixation (100 ms) -> frame1 (50 ms) -> frame2
    (``n_repeats`` x 50 ms) -> decision (100 ms), dt = 50. Ground truth is
    ``fixation`` through frame 1 and the direction from frame 2 onward, matching
    ``TAMTask`` so the abstention metric means the same thing across both.

    The action space stays ``Discrete(6)`` with the frozen ``TAM_CHOICES``
    names even though only ``left`` and ``right`` are sampled, so plotting,
    metrics and the analysis suite are shared.
    """

    def __init__(
        self,
        dt: int = 50,
        split: str = "train",
        img_size: int = 64,
        sigma: float = 0.0,
        n_repeats: int = 4,
        rewards: dict | None = None,
        timing: dict | None = None,
        **sampler_overrides,
    ):
        super().__init__(dt=dt)
        if isinstance(split, str) and split not in SPLITS:
            msg = f"Unknown split {split!r}; expected one of {tuple(SPLITS)}"
            raise ValueError(msg)

        self.split = split
        self.img_size = img_size
        self.sigma = sigma
        self.n_repeats = n_repeats
        self._sample = sampler_for(split, **sampler_overrides)

        self.abort = False
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self._frame_periods = ["frame1"] + [f"frame2_{i}" for i in range(n_repeats)]
        self.timing = {"fixation": 100, "decision": 100}
        self.timing.update({period: 50 for period in self._frame_periods})
        if timing:
            self.timing.update(timing)

        self.ob_shape = (img_size, img_size)
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=self.ob_shape, dtype=np.float32,
        )
        self.choice_names = dict(TAM_CHOICES)
        self.action_space = ngym.spaces.Discrete(6, name=self.choice_names)

        # Observation-array indices of each frame. Derived from the timing dict
        # rather than hardcoded so a `timing` override cannot silently
        # desynchronise the frame-only baselines from the actual observations.
        n_fix = self.timing["fixation"] // dt
        self.frame1_index = n_fix
        self.frame2_indices = tuple(range(n_fix + 1, n_fix + 1 + n_repeats))

    def _new_trial(self, **kwargs):
        params = self._sample(self.rng)
        params.update({k: v for k, v in kwargs.items() if k in params})
        stimulus = render_trial(params, img_size=self.img_size)
        direction = self.choice_names[stimulus.label]

        trial = {"ground_truth": direction, "noise": self.sigma, **params}

        self.add_period(["fixation", *self._frame_periods, "decision"])
        self.add_ob(self._fixation_ob(), period=["fixation"])
        self.add_ob(stimulus.frame1, period=["frame1"])
        for period in self._frame_periods[1:]:
            self.add_ob(stimulus.frame2, period=[period])
        self.add_ob(np.zeros(self.ob_shape), period=["decision"])
        self.add_randn(0, self.sigma, period=self._frame_periods)

        self.set_groundtruth(
            self.choice_names["fixation"], period=["fixation", "frame1"],
        )
        self.set_groundtruth(
            direction, period=[*self._frame_periods[1:], "decision"],
        )
        return trial

    _fixation_ob = TAMTask._fixation_ob
    _step = TAMTask._step
```

```python
# illusion_rnn/training.py -- extend make_env
def make_env(task: str, **kwargs):
    """Construct a task env: ``task`` is ``"tam"`` (TAMTask), ``"motion"``
    (MotionTask), or ``"correspondence"`` (TAMCorrespondenceTask); ``kwargs``
    pass through to the constructor."""
    if task == "tam":
        return TAMTask(**kwargs)
    if task == "motion":
        return MotionTask(**kwargs)
    if task == "correspondence":
        return TAMCorrespondenceTask(**kwargs)
    msg = f"Unknown task {task!r}; expected 'tam', 'motion' or 'correspondence'"
    raise ValueError(msg)
```

Add `TAMCorrespondenceTask` to the imports and `__all__` in `illusion_rnn/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_envs.py -v`
Expected: PASS. If `frame1_index` is off by one, print `env.ob` row sums and compare against the period boundaries — neurogym's `reset()` consumes the first timestep, but `new_trial()` followed by reading `env.ob` does not, and the tests use `new_trial()`.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, ~50 new tests on top of the existing 106.

- [ ] **Step 6: Commit**

```bash
git add illusion_rnn/envs.py illusion_rnn/training.py illusion_rnn/__init__.py tests/test_envs.py
git commit -m "$(cat <<'EOF'
feat: add TAMCorrespondenceTask

A 2AFC TAM environment whose direction label depends on the relation
between frame 1 and frame 2 rather than on either frame alone. Frame 1
varies per trial and frame 2 is drawn independently of the label, so a
model must bind the two to answer.

Keeps TAMTask's frozen label map and Discrete(6) action space so
abstention reporting, plotting and the analysis suite are shared. Frame
indices are derived from the timing dict rather than hardcoded, so a
timing override cannot silently desynchronise the frame-only baselines
from the observations they are supposed to read.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 8: Verification checkpoint 1 — frame 2 carries no label information

**Files:**
- Create: `scripts/verify_generator.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `sampler_for`, `render_trial` from Tasks 4–6.
- Produces: `pixel_label_mi(frames: np.ndarray, labels: np.ndarray) -> np.ndarray` in `illusion_rnn/generate.py`, returning per-pixel mutual information in bits. Task 13 reuses it for the outline analysis.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate.py -- append
from illusion_rnn.generate import pixel_label_mi


def test_pixel_label_mi_detects_a_planted_cue():
    """Sanity check the detector before trusting it to clear frame 2."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=2000)
    frames = rng.random((2000, 4, 4)) * 0.1
    frames[labels == 1, 0, 0] = 1.0          # a blatant label cue
    mi = pixel_label_mi(frames, labels)
    assert mi[0, 0] > 0.5, "failed to detect a planted cue"
    assert mi[2, 2] < 0.05, "hallucinated a cue in a null pixel"


def test_frame2_carries_no_label_information_in_the_balanced_family():
    """Statistical form of the load-bearing invariant, over sampled trials.

    The bit-identical test proves it for a fixed geometry; this proves the
    SAMPLER does not reintroduce a correlation.
    """
    sample = sampler_for("train")
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for _ in range(3000):
        t = render_trial(sample(rng))
        frames.append(t.frame2)
        labels.append(t.label == "right")
    mi = pixel_label_mi(np.array(frames), np.array(labels))
    assert mi.max() < 0.02, f"frame 2 leaks {mi.max():.4f} bits at {np.unravel_index(mi.argmax(), mi.shape)}"


def test_frame2_DOES_carry_label_information_in_the_classic_family():
    """The diagnostic family must leak -- otherwise it is not diagnosing
    anything and the comparison across families is meaningless."""
    sample = sampler_for("train", family="classic")
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for _ in range(3000):
        t = render_trial(sample(rng))
        frames.append(t.frame2)
        labels.append(t.label == "right")
    mi = pixel_label_mi(np.array(frames), np.array(labels))
    assert mi.max() > 0.2, "classic family unexpectedly balanced"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -k mi -v`
Expected: FAIL with `ImportError: cannot import name 'pixel_label_mi'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/generate.py -- append

def pixel_label_mi(frames: np.ndarray, labels: np.ndarray, n_bins: int = 8):
    """Per-pixel mutual information in bits between pixel value and label.

    Pixel values are binned into ``n_bins`` equal-width bins; the label is
    binary. Returns an ``(H, W)`` array. Used to verify that frame 2 carries no
    direction information in the balanced family -- and that it DOES in the
    classic diagnostic family.
    """
    frames = np.asarray(frames, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_trials, height, width = frames.shape
    flat = frames.reshape(n_trials, -1)

    # per-pixel equal-width binning over that pixel's own observed range
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    binned = np.clip(((flat - lo) / span * n_bins).astype(int), 0, n_bins - 1)

    mi = np.zeros(flat.shape[1])
    p_label = np.array([(labels == v).mean() for v in (0, 1)])
    for pixel in range(flat.shape[1]):
        col = binned[:, pixel]
        joint = np.zeros((n_bins, 2))
        for value in (0, 1):
            counts = np.bincount(col[labels == value], minlength=n_bins)
            joint[:, value] = counts
        joint /= n_trials
        p_bin = joint.sum(axis=1, keepdims=True)
        expected = p_bin * p_label[np.newaxis, :]
        nz = joint > 0
        mi[pixel] = float((joint[nz] * np.log2(joint[nz] / expected[nz])).sum())
    return mi.reshape(height, width)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_generate.py -k mi -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Write the verification script and run checkpoint 1**

```python
# scripts/verify_generator.py
"""Verification checkpoint 1: the generator carries no frame-2 label cue.

Prints per-family maximum per-pixel mutual information between frame 2 and the
direction label, and writes a sample-trial figure. The balanced family must be
at the noise floor; the classic family must be well above it, or it is not
diagnosing anything.

Run: .venv/bin/python scripts/verify_generator.py
"""
import matplotlib.pyplot as plt
import numpy as np

from illusion_rnn.generate import pixel_label_mi, render_trial, sampler_for

N_TRIALS = 10_000


def collect(family, split="train"):
    sample = sampler_for(split, family=family)
    rng = np.random.default_rng(0)
    trials = [render_trial(sample(rng)) for _ in range(N_TRIALS)]
    labels = np.array([t.label == "right" for t in trials])
    return trials, labels


def main():
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, family in enumerate(("balanced", "classic")):
        trials, labels = collect(family)
        for col, frame_name in enumerate(("frame1", "frame2")):
            frames = np.array([getattr(t, frame_name) for t in trials])
            mi = pixel_label_mi(frames, labels)
            print(f"{family:9s} {frame_name}: max per-pixel MI = {mi.max():.4f} bits")
        # sample trials: one leftward, one rightward
        for col, want in enumerate(("left", "right")):
            t = next(t for t in trials if t.label == want)
            axes[row, col * 2].imshow(t.frame1, cmap="gray_r", vmin=0, vmax=1)
            axes[row, col * 2].set_title(f"{family} {want}\nframe 1", fontsize=8)
            axes[row, col * 2 + 1].imshow(t.frame2, cmap="gray_r", vmin=0, vmax=1)
            axes[row, col * 2 + 1].set_title("frame 2", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("figures/correspondence_trials.png", dpi=150)
    print("wrote figures/correspondence_trials.png")


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/verify_generator.py`

**Pass condition:** `balanced frame2` max MI below 0.02 bits. **If it is not, stop — the generator leaks and every downstream result is void.** Inspect the figure: the two `balanced` frame-2 panels must be visually identical.

- [ ] **Step 6: Commit**

```bash
git add illusion_rnn/generate.py scripts/verify_generator.py tests/test_generate.py figures/correspondence_trials.png
git commit -m "$(cat <<'EOF'
feat: add verification checkpoint 1 -- frame-2 label independence

Per-pixel mutual information between frame 2 and the direction label,
with the detector itself sanity-checked against a planted cue before it
is trusted to clear anything.

The balanced family must sit at the noise floor and the classic family
must sit well above it. Both are asserted, because a classic family that
did not leak would mean the cross-family comparison is measuring nothing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 9: Baseline model classes

**Files:**
- Modify: `illusion_rnn/models.py`
- Modify: `illusion_rnn/__init__.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `env.frame1_index`, `env.frame2_indices` from Task 7.
- Produces four classes, all honouring the existing contract `forward(x: (T,B,F)) -> (out: (T,B,output_size), activity: (T,B,H))`:
  - `GRUNet(input_size, hidden_size, output_size)`
  - `FFStack(input_size, hidden_size, output_size, n_steps)`
  - `FrameOnlyNet(input_size, hidden_size, output_size, n_steps, frame_index)`
  - `shuffle_frames(x: torch.Tensor, frame_indices: Sequence[int], generator) -> torch.Tensor`

  `FF1Only` and `FF2Only` are `FrameOnlyNet` with different `frame_index`, not separate classes. Task 11 constructs all of these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py -- append
import torch

from illusion_rnn.models import FFStack, FrameOnlyNet, GRUNet, shuffle_frames

T, B, F, H, OUT = 9, 4, 64, 16, 6


@pytest.mark.parametrize("build", [
    lambda: GRUNet(F, H, OUT),
    lambda: FFStack(F, H, OUT, n_steps=T),
    lambda: FrameOnlyNet(F, H, OUT, n_steps=T, frame_index=3),
])
def test_models_honour_the_shared_contract(build):
    model = build()
    out, activity = model(torch.randn(T, B, F))
    assert out.shape == (T, B, OUT)
    assert activity.shape[:2] == (T, B)


def test_frame_only_net_reads_exactly_one_timestep():
    """If it can see any other timestep the baseline is not a baseline."""
    model = FrameOnlyNet(F, H, OUT, n_steps=T, frame_index=3)
    x = torch.zeros(T, B, F)
    base, _ = model(x)
    for t in range(T):
        probe = torch.zeros(T, B, F)
        probe[t] = 1.0
        out, _ = model(probe)
        changed = not torch.allclose(out, base)
        assert changed == (t == 3), f"timestep {t}: changed={changed}"


def test_ffstack_sees_every_timestep():
    model = FFStack(F, H, OUT, n_steps=T)
    x = torch.zeros(T, B, F)
    base, _ = model(x)
    for t in range(T):
        probe = torch.zeros(T, B, F)
        probe[t] = 1.0
        out, _ = model(probe)
        assert not torch.allclose(out, base), f"blind to timestep {t}"


def test_ffstack_output_varies_across_time():
    """It must be able to answer `fixation` early and a direction late, or the
    per-timestep loss punishes it for a handicap the recurrent models escape."""
    model = FFStack(F, H, OUT, n_steps=T)
    out, _ = model(torch.randn(T, B, F))
    assert not torch.allclose(out[0], out[-1])


def test_shuffle_frames_permutes_only_the_named_timesteps():
    x = torch.arange(T * 1 * 2, dtype=torch.float32).reshape(T, 1, 2)
    g = torch.Generator().manual_seed(0)
    y = shuffle_frames(x, frame_indices=(2, 3, 4, 5, 6), generator=g)
    for t in (0, 1, 7, 8):
        torch.testing.assert_close(y[t], x[t])
    assert sorted(y[i, 0, 0].item() for i in (2, 3, 4, 5, 6)) == \
           sorted(x[i, 0, 0].item() for i in (2, 3, 4, 5, 6))


def test_shuffle_frames_actually_changes_the_order_sometimes():
    x = torch.arange(T * 1 * 2, dtype=torch.float32).reshape(T, 1, 2)
    g = torch.Generator().manual_seed(0)
    permuted = [shuffle_frames(x, (2, 3, 4, 5, 6), g) for _ in range(20)]
    assert any(not torch.allclose(p, x) for p in permuted)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'GRUNet'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/models.py -- append

class GRUNet(nn.Module):
    """GRU with a linear readout -- a recurrence-type control for ``RNNNet``.

    Same contract as ``RNNNet``: ``(T, B, input_size)`` in,
    ``(out (T, B, output_size), activity (T, B, hidden_size))`` out.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        activity, _ = self.rnn(x)
        return self.fc(activity), activity


class FFStack(nn.Module):
    """Feedforward MLP over every timestep concatenated -- no recurrence.

    Isolates whether *recurrence* matters or merely *access to both frames*:
    this model sees the whole trial at once but has no state.

    It emits a distinct prediction per timestep (the MLP maps to
    ``n_steps * output_size``) so it can answer ``fixation`` early and a
    direction late. Broadcasting one prediction across all timesteps would
    handicap it against the recurrent models under the per-timestep loss and
    make the comparison unfair.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size * n_steps, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, n_steps * output_size)

    def forward(self, x: torch.Tensor):
        n_steps, batch, _ = x.shape
        flat = x.permute(1, 0, 2).reshape(batch, -1)
        hidden = self.net(flat)
        out = self.fc(hidden).reshape(batch, self.n_steps, self.output_size)
        activity = hidden.unsqueeze(0).expand(n_steps, batch, hidden.shape[-1])
        return out.permute(1, 0, 2), activity


class FrameOnlyNet(nn.Module):
    """Feedforward MLP that sees exactly ONE timestep of the trial.

    ``frame_index=env.frame2_indices[0]`` gives the frame-2-only baseline, whose
    ceiling on the balanced family is provably 50%.
    ``frame_index=env.frame1_index`` gives the frame-1-only baseline.

    The single-timestep restriction is enforced by indexing, not by masking, so
    there is no path by which other timesteps can reach the output --
    ``test_frame_only_net_reads_exactly_one_timestep`` verifies it empirically.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 n_steps: int, frame_index: int):
        super().__init__()
        if not 0 <= frame_index < n_steps:
            msg = f"frame_index {frame_index} out of range for {n_steps} steps"
            raise ValueError(msg)
        self.n_steps = n_steps
        self.frame_index = frame_index
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, n_steps * output_size)

    def forward(self, x: torch.Tensor):
        n_steps, batch, _ = x.shape
        hidden = self.net(x[self.frame_index])
        out = self.fc(hidden).reshape(batch, self.n_steps, self.output_size)
        activity = hidden.unsqueeze(0).expand(n_steps, batch, hidden.shape[-1])
        return out.permute(1, 0, 2), activity


def shuffle_frames(x: torch.Tensor, frame_indices, generator=None) -> torch.Tensor:
    """Randomly permute the named timesteps of ``x`` independently per trial.

    Used for the order control. On the ``growth+shrink`` condition this caps
    accuracy at 50%, because the same frame pair appears in both orders with
    opposite labels. On the growth-only condition it does NOT -- frame 1 is a
    shape and frame 2 is a bar, so a model can tell them apart by content and
    order carries nothing extra.
    """
    idx = list(frame_indices)
    out = x.clone()
    batch = x.shape[1]
    for b in range(batch):
        perm = torch.randperm(len(idx), generator=generator)
        out[idx, b] = x[[idx[p] for p in perm], b]
    return out
```

Add `FFStack`, `FrameOnlyNet`, `GRUNet`, `shuffle_frames` to `illusion_rnn/__init__.py` imports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/models.py illusion_rnn/__init__.py tests/test_models.py
git commit -m "$(cat <<'EOF'
feat: add GRU, no-recurrence and frame-only baseline models

FrameOnlyNet sees exactly one timestep and is the baseline the whole
experiment turns on; a test probes every timestep individually to confirm
no other one can reach its output.

FFStack emits a distinct prediction per timestep rather than broadcasting
one answer across the trial. Broadcasting would force it to predict a
direction during the fixation period, handicapping it against the
recurrent models under the per-timestep loss and making the "is
recurrence needed" comparison unfair.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 10: Verification checkpoint 2 — the frame-2 baseline must land at 50%

**This is the load-bearing gate.** It is the empirical version of the provability argument, and it is exactly what the 2023 experiment lacked. If it fails, nothing downstream is worth running.

**Files:**
- Create: `scripts/verify_baselines.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: everything from Tasks 4–9.
- Produces: nothing later tasks import. It is a gate.

- [ ] **Step 1: Write a fast integration test**

```python
# tests/test_sweep.py
"""Integration checks that the task is learnable and the baselines are floored.

These train real (tiny) models, so they are slower than the unit tests. They are
the cheap version of verification checkpoint 2; scripts/verify_baselines.py is
the full version.
"""
import numpy as np
import pytest

from illusion_rnn import evaluate, make_dataset, make_env, train
from illusion_rnn.models import FrameOnlyNet, RNNNet


@pytest.mark.slow
def test_frame2_only_baseline_cannot_beat_chance():
    """The gate. Frame 2 is independent of the label by construction, so a
    model that sees only frame 2 must sit at 50% no matter how long it trains.
    """
    env = make_env("correspondence", split="train", img_size=32)
    env.seed(0)
    dataset = make_dataset(env, batch_size=32, seq_len=90)
    model = FrameOnlyNet(
        32 * 32, 128, 6, n_steps=9, frame_index=env.frame2_indices[0],
    )
    train(model, dataset, n_epochs=300, device="cpu", log_every=0)

    eval_env = make_env("correspondence", split="train", img_size=32)
    eval_env.seed(1)
    result = evaluate(model, eval_env, n_trials=400, device="cpu")
    committed = result.committed_accuracy
    assert np.isnan(committed) or committed < 0.60, (
        f"frame-2-only reached {committed:.3f}; the generator leaks and every "
        f"downstream result is void"
    )


@pytest.mark.slow
def test_the_task_is_learnable_by_a_recurrent_model():
    """The other half of the gate: if the CTRNN cannot learn it either, the
    task is impossible rather than merely well-controlled."""
    env = make_env("correspondence", split="train", img_size=32)
    env.seed(0)
    dataset = make_dataset(env, batch_size=32, seq_len=90)
    model = RNNNet(32 * 32, 256, 6, dt=50)
    train(model, dataset, n_epochs=600, device="cpu", log_every=0)

    eval_env = make_env("correspondence", split="train", img_size=32)
    eval_env.seed(1)
    result = evaluate(model, eval_env, n_trials=400, device="cpu")
    assert result.accuracy > 0.70, (
        f"CTRNN reached only {result.accuracy:.3f}; the task may be unlearnable "
        f"at this size rather than well-controlled"
    )
```

Register the marker in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: trains a model; excluded from the default CI run"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/python -m pytest tests/test_sweep.py -m slow -v`
Expected: both PASS.

**If `test_frame2_only_baseline_cannot_beat_chance` fails:** stop. Re-run `scripts/verify_generator.py`, find which pixels leak, and fix the sampler. Do not proceed.

**If `test_the_task_is_learnable_by_a_recurrent_model` fails:** the task may need a longer frame-1 presentation or a larger hidden size. Try `n_repeats=2` (shorter memory demand) and `hidden=512` before concluding it is unlearnable.

- [ ] **Step 3: Write the full verification script**

```python
# scripts/verify_baselines.py
"""Verification checkpoint 2: every baseline lands where theory says.

Trains each model once at 64px and prints accuracy with the abstention split.
Expected, on the balanced/growth condition:

    FF2Only  ~0.50   provable ceiling
    FF1Only  ~0.55   measured floor (see spec section 3)
    FFStack  high    has both frames, no recurrence
    RNNNet   high    the binding model

Run: .venv/bin/python scripts/verify_baselines.py
"""
import numpy as np
import torch

from illusion_rnn import evaluate, make_dataset, make_env, train
from illusion_rnn.models import FFStack, FrameOnlyNet, GRUNet, RNNNet

IMG, HIDDEN, EPOCHS = 64, 256, 1500


def build(name, env, n_steps):
    size = IMG * IMG
    if name == "RNNNet":
        return RNNNet(size, HIDDEN, 6, dt=50)
    if name == "GRUNet":
        return GRUNet(size, HIDDEN, 6)
    if name == "FFStack":
        return FFStack(size, HIDDEN, 6, n_steps=n_steps)
    if name == "FF1Only":
        return FrameOnlyNet(size, HIDDEN, 6, n_steps=n_steps,
                            frame_index=env.frame1_index)
    if name == "FF2Only":
        return FrameOnlyNet(size, HIDDEN, 6, n_steps=n_steps,
                            frame_index=env.frame2_indices[0])
    raise ValueError(f"Unknown model {name!r}")


def main():
    torch.manual_seed(0)
    env = make_env("correspondence", split="train", img_size=IMG)
    env.seed(0)
    n_steps = env.ob.shape[0] if env.ob is not None else 9
    env.new_trial()
    n_steps = env.ob.shape[0]

    for name in ("FF2Only", "FF1Only", "FFStack", "GRUNet", "RNNNet"):
        model = build(name, env, n_steps)
        dataset = make_dataset(make_env("correspondence", split="train",
                                        img_size=IMG), batch_size=64,
                               seq_len=n_steps * 10)
        train(model, dataset, n_epochs=EPOCHS, log_every=0)
        eval_env = make_env("correspondence", split="train", img_size=IMG)
        eval_env.seed(1)
        r = evaluate(model, eval_env, n_trials=500)
        print(f"{name:9s} acc={r.accuracy:.3f}  abstain={r.abstention_rate:.3f}  "
              f"committed={r.committed_accuracy:.3f}")


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/verify_baselines.py`

**Pass condition:** `FF2Only` committed accuracy in `[0.48, 0.52]`, `RNNNet` well above it.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_baselines.py tests/test_sweep.py pyproject.toml
git commit -m "$(cat <<'EOF'
test: add verification checkpoint 2 -- baselines land at their ceilings

The gate the 2023 experiment lacked. A frame-2-only model is trained to
convergence and must stay at chance; if it does not, the generator leaks
and every downstream number is void.

Paired with a learnability check, because a task no model can learn would
pass the first assertion trivially. Both are marked `slow` and excluded
from the default run since they train real models.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 11: The sweep runner

**Files:**
- Create: `illusion_rnn/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: model classes from Task 9, `make_env`/`train`/`evaluate` from Task 1.
- Produces: `ARCHITECTURES: tuple[str, ...]` = `("RNNNet", "GRUNet", "FFStack", "FF1Only", "FF2Only", "RNNNet-shuffled")`; `build_model(name, *, input_size, hidden_size, output_size, n_steps, env) -> nn.Module`; `run_cell(**cfg) -> dict`; `run_grid(architectures, families, hidden_sizes, seeds, **kw) -> list[dict]`; `aggregate(records) -> list[dict]` returning mean and 95% CI per `(architecture, family, hidden_size, split)`. Tasks 12 and 13 call `run_grid` and `aggregate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep.py -- append
from illusion_rnn.sweep import ARCHITECTURES, aggregate, build_model, run_grid


def test_every_named_architecture_builds():
    env = make_env("correspondence", split="train", img_size=16)
    env.seed(0); env.new_trial()
    for name in ARCHITECTURES:
        model = build_model(name, input_size=16 * 16, hidden_size=8,
                            output_size=6, n_steps=env.ob.shape[0], env=env)
        assert model is not None, name


def test_build_model_rejects_an_unknown_architecture():
    env = make_env("correspondence", split="train", img_size=16)
    env.seed(0); env.new_trial()
    with pytest.raises(ValueError, match="Unknown architecture 'transformer'"):
        build_model("transformer", input_size=256, hidden_size=8,
                    output_size=6, n_steps=9, env=env)


@pytest.mark.slow
def test_run_grid_produces_one_record_per_cell_and_split():
    records = run_grid(
        architectures=("FF2Only",), families=("balanced",),
        hidden_sizes=(8,), seeds=(0, 1),
        img_size=16, n_epochs=5, n_eval_trials=20,
    )
    assert len(records) == 2 * 4      # 2 seeds x 4 splits
    for r in records:
        assert set(r) >= {"architecture", "family", "hidden_size", "seed",
                          "split", "accuracy", "abstention_rate",
                          "committed_accuracy"}


def test_aggregate_computes_mean_and_ci_per_cell():
    records = [
        {"architecture": "A", "family": "balanced", "hidden_size": 8,
         "split": "train", "seed": s, "accuracy": a}
        for s, a in enumerate([0.5, 0.6, 0.7, 0.8, 0.9])
    ]
    out = aggregate(records)
    assert len(out) == 1
    row = out[0]
    assert row["accuracy_mean"] == pytest.approx(0.7)
    assert row["n_seeds"] == 5
    assert row["accuracy_ci_low"] < 0.7 < row["accuracy_ci_high"]


def test_aggregate_reports_a_degenerate_ci_for_a_single_seed():
    """One seed gives no interval; it must say so rather than invent one."""
    records = [{"architecture": "A", "family": "balanced", "hidden_size": 8,
                "split": "train", "seed": 0, "accuracy": 0.7}]
    row = aggregate(records)[0]
    assert row["n_seeds"] == 1
    assert np.isnan(row["accuracy_ci_low"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sweep.py -k "architecture or aggregate or grid" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'illusion_rnn.sweep'`

- [ ] **Step 3: Write the implementation**

```python
# illusion_rnn/sweep.py
"""Multi-seed grid runner for the correspondence experiment.

One cell is an (architecture, family, hidden_size, seed) tuple. Each cell trains
one model on the `train` split and evaluates it on all four splits, producing
one record per split. `aggregate` reduces records to mean and 95% CI across
seeds.

Seeds control weight initialisation and trial sampling together, so a cell is
reproducible end to end from an integer.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from illusion_rnn.generate import SPLITS
from illusion_rnn.models import FFStack, FrameOnlyNet, GRUNet, RNNNet, shuffle_frames
from illusion_rnn.training import evaluate, make_dataset, make_env, train

ARCHITECTURES = (
    "RNNNet", "GRUNet", "FFStack", "FF1Only", "FF2Only", "RNNNet-shuffled",
)


def build_model(name, *, input_size, hidden_size, output_size, n_steps, env):
    """Construct one architecture by name.

    ``RNNNet-shuffled`` is an ``RNNNet``; the shuffling is applied to its inputs
    during training and evaluation, not baked into the module.
    """
    if name in ("RNNNet", "RNNNet-shuffled"):
        return RNNNet(input_size, hidden_size, output_size, dt=50)
    if name == "GRUNet":
        return GRUNet(input_size, hidden_size, output_size)
    if name == "FFStack":
        return FFStack(input_size, hidden_size, output_size, n_steps=n_steps)
    if name == "FF1Only":
        return FrameOnlyNet(input_size, hidden_size, output_size,
                            n_steps=n_steps, frame_index=env.frame1_index)
    if name == "FF2Only":
        return FrameOnlyNet(input_size, hidden_size, output_size,
                            n_steps=n_steps, frame_index=env.frame2_indices[0])
    msg = f"Unknown architecture {name!r}; expected one of {ARCHITECTURES}"
    raise ValueError(msg)


def run_cell(*, architecture, family, hidden_size, seed, img_size=64,
             transform="growth", n_epochs=1500, batch_size=64,
             n_eval_trials=500, device=None):
    """Train one model and evaluate it on every split. Returns one record per
    split."""
    torch.manual_seed(seed)
    rng_seed = seed

    def env_for(split, eval_seed):
        env = make_env("correspondence", split=split, img_size=img_size,
                       family=family, transform=transform)
        env.seed(eval_seed)
        env.new_trial()
        return env

    train_env = env_for("train", rng_seed)
    n_steps = train_env.ob.shape[0]
    model = build_model(
        architecture, input_size=img_size * img_size, hidden_size=hidden_size,
        output_size=6, n_steps=n_steps, env=train_env,
    )

    shuffled = architecture.endswith("-shuffled")
    frame_indices = (train_env.frame1_index, *train_env.frame2_indices)
    gen = torch.Generator().manual_seed(seed) if shuffled else None

    def maybe_shuffle(x):
        return shuffle_frames(x, frame_indices, gen) if shuffled else x

    dataset = make_dataset(env_for("train", rng_seed), batch_size=batch_size,
                           seq_len=n_steps * 10)
    train(model, dataset, n_epochs=n_epochs, device=device, log_every=0,
          encoder=None, input_transform=maybe_shuffle)

    records = []
    for split in SPLITS:
        result = evaluate(model, env_for(split, 10_000 + rng_seed),
                          n_trials=n_eval_trials, device=device,
                          input_transform=maybe_shuffle)
        records.append({
            "architecture": architecture, "family": family,
            "hidden_size": hidden_size, "seed": seed, "split": split,
            "transform": transform,
            "accuracy": result.accuracy,
            "abstention_rate": result.abstention_rate,
            "committed_accuracy": result.committed_accuracy,
        })
    return records


def run_grid(*, architectures, families, hidden_sizes, seeds, **kwargs):
    """Run every (architecture, family, hidden_size, seed) combination."""
    out = []
    for architecture in architectures:
        for family in families:
            for hidden_size in hidden_sizes:
                for seed in seeds:
                    out.extend(run_cell(
                        architecture=architecture, family=family,
                        hidden_size=hidden_size, seed=seed, **kwargs,
                    ))
    return out


def aggregate(records, metric="accuracy"):
    """Mean and 95% CI of ``metric`` across seeds, per cell.

    The CI is the normal-approximation interval on the seed mean. With one seed
    the interval is NaN rather than zero -- a single run has no interval and
    must not be drawn as if it had a tight one.
    """
    cells = {}
    for r in records:
        key = (r["architecture"], r["family"], r["hidden_size"], r["split"])
        cells.setdefault(key, []).append(r[metric])

    out = []
    for (architecture, family, hidden_size, split), values in cells.items():
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        mean = float(np.nanmean(arr))
        if n < 2:
            low = high = float("nan")
        else:
            sem = float(np.nanstd(arr, ddof=1) / np.sqrt(n))
            low, high = mean - 1.96 * sem, mean + 1.96 * sem
        out.append({
            "architecture": architecture, "family": family,
            "hidden_size": hidden_size, "split": split, "n_seeds": n,
            f"{metric}_mean": mean, f"{metric}_ci_low": low,
            f"{metric}_ci_high": high,
        })
    return out
```

This needs `train()` and `evaluate()` to accept an `input_transform`. Add it:

```python
# illusion_rnn/training.py -- in _prepare_inputs' callers
# train(): add parameter `input_transform=None`, and after building `x`:
    if input_transform is not None:
        x = input_transform(x)
# evaluate(): same parameter, same placement after `x` is built.
```

Document it: `input_transform` is applied to the `(T, B, F)` tensor after encoding, and exists so the order control can permute frames without a separate training loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sweep.py -v` then `.venv/bin/python -m pytest tests/test_sweep.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/sweep.py illusion_rnn/training.py tests/test_sweep.py
git commit -m "$(cat <<'EOF'
feat: add the multi-seed sweep runner

One cell trains a model on the train split and evaluates on all four
splits. aggregate() reduces across seeds to mean and 95% CI, returning
NaN bounds for a single seed rather than a zero-width interval that would
plot as certainty.

train() and evaluate() gain an input_transform hook so the frame-order
control can permute inputs without duplicating the training loop.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 12: Local mini-sweep — verification checkpoints 3 and 4

**Files:**
- Create: `scripts/run_sweep.py`
- Modify: `.gitignore` if needed for intermediate artifacts

**Interfaces:**
- Consumes: `run_grid`, `aggregate` from Task 11.
- Produces: `results/sweep_local.json` (records) — Task 13 reuses the same writer for the HF run.

- [ ] **Step 1: Write the sweep script**

```python
# scripts/run_sweep.py
"""Run the correspondence sweep and write records to results/.

Local smoke run (minutes, MPS):
    .venv/bin/python scripts/run_sweep.py --quick --out results/sweep_local.json

Full grid (HF Jobs, ~1h on l4x1):
    .venv/bin/python scripts/run_sweep.py --out results/sweep.json
"""
import argparse
import json
from pathlib import Path

from illusion_rnn.sweep import ARCHITECTURES, aggregate, run_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="tiny grid for a local smoke test")
    parser.add_argument("--out", default="results/sweep.json")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    if args.quick:
        config = dict(
            architectures=("FF2Only", "RNNNet"), families=("balanced",),
            hidden_sizes=(64,), seeds=tuple(range(2)),
            img_size=32, n_epochs=200, n_eval_trials=100,
        )
    else:
        config = dict(
            architectures=ARCHITECTURES, families=("balanced", "classic"),
            hidden_sizes=(256,), seeds=tuple(range(args.seeds)),
            img_size=64, n_epochs=1500, n_eval_trials=500,
        )

    records = run_grid(**config)
    summary = aggregate(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {k: list(v) if isinstance(v, tuple) else v
                    for k, v in config.items()},
         "records": records, "summary": summary}, indent=2,
    ))
    print(f"wrote {out}  ({len(records)} records, {len(summary)} cells)")

    for row in sorted(summary, key=lambda r: (r["family"], r["split"],
                                              r["architecture"])):
        if row["split"] != "train":
            continue
        print(f"{row['family']:9s} {row['architecture']:16s} "
              f"{row['accuracy_mean']:.3f} "
              f"[{row['accuracy_ci_low']:.3f}, {row['accuracy_ci_high']:.3f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run checkpoint 3 — split disjointness**

Run:
```bash
.venv/bin/python -m pytest tests/test_generate.py -k "disjoint" -v
```
Expected: PASS. Already covered by Task 6, re-run here as the formal checkpoint.

- [ ] **Step 3: Run the quick sweep**

Run: `.venv/bin/python scripts/run_sweep.py --quick --out results/sweep_local.json`
Expected: completes in minutes; `FF2Only` near 0.50, `RNNNet` above it.

- [ ] **Step 4: Run checkpoint 4 — CI separation**

Run:
```bash
.venv/bin/python -c "
import json
s = json.load(open('results/sweep_local.json'))['summary']
rows = {r['architecture']: r for r in s if r['split'] == 'train' and r['family'] == 'balanced'}
rnn, ff2 = rows['RNNNet'], rows['FF2Only']
print(f\"RNNNet  {rnn['accuracy_mean']:.3f} [{rnn['accuracy_ci_low']:.3f}, {rnn['accuracy_ci_high']:.3f}]\")
print(f\"FF2Only {ff2['accuracy_mean']:.3f} [{ff2['accuracy_ci_low']:.3f}, {ff2['accuracy_ci_high']:.3f}]\")
print('SEPARATED' if rnn['accuracy_ci_low'] > ff2['accuracy_ci_high'] else 'OVERLAPPING -- do not proceed')
"
```
**Pass condition:** `SEPARATED`. With 2 seeds the CI is wide; if it overlaps only because of seed count, raise `--seeds` and rerun before concluding anything.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sweep.py results/sweep_local.json
git commit -m "$(cat <<'EOF'
feat: add the sweep entry point and run the local smoke grid

Quick mode runs a 2-architecture, 2-seed grid at 32px in minutes so the
pipeline can be validated before spending GPU time. Verification
checkpoints 3 (split disjointness) and 4 (CI separation between the
binding model and the frame-2 baseline) both pass locally.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

### Task 13: HF Jobs sweep, outline curve, and results

**Files:**
- Create: `scripts/hf_sweep.py`
- Create: `scripts/plot_results.py`
- Modify: `README.md`
- Test: manual — this task's deliverable is measured numbers

**Interfaces:**
- Consumes: `run_grid`, `aggregate` from Task 11; `ink_mass` from Task 3.
- Produces: `results/sweep.json`, `figures/correspondence_accuracy.png`, `figures/outline_ink_curve.png`.

- [ ] **Step 1: Write the UV script for HF Jobs**

```python
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
```

- [ ] **Step 2: Push the branch so the job can install from it**

```bash
git push -u origin correspondence
```

- [ ] **Step 3: Launch the job**

Run:
```bash
hf jobs uv run --flavor l4x1 --timeout 2h --detach \
  --name illusion-correspondence-sweep scripts/hf_sweep.py
```
Then follow with `hf jobs logs <JOB_ID> --follow`.

Expected: ~1 hour, ~$0.80. If it errors on install, check that the branch is pushed and public.

- [ ] **Step 4: Recover results into `results/sweep.json`**

```bash
hf jobs logs <JOB_ID> | grep '^RESULTS_JSON ' | cut -d' ' -f2- > results/sweep.json
.venv/bin/python -c "import json; d=json.load(open('results/sweep.json')); print(len(d['records']), 'records')"
```

- [ ] **Step 5: Run checkpoint 5 — the outline ink-mass curve**

```python
# scripts/plot_results.py
"""Figures for the correspondence experiment.

1. accuracy by architecture and family, with 95% CIs and the analytic floors
2. outline accuracy as a function of ink-mass ratio -- separating "cannot infer
   motion from outline form" from "input too far out of distribution"

Run: .venv/bin/python scripts/plot_results.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from illusion_rnn import evaluate, make_env
from illusion_rnn.generate import ink_mass, render_trial, sampler_for


def accuracy_figure(summary, path="figures/correspondence_accuracy.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, family in zip(axes, ("balanced", "classic")):
        rows = [r for r in summary
                if r["family"] == family and r["split"] == "train"]
        rows.sort(key=lambda r: r["accuracy_mean"])
        names = [r["architecture"] for r in rows]
        means = [r["accuracy_mean"] for r in rows]
        err = [[m - r["accuracy_ci_low"] for m, r in zip(means, rows)],
               [r["accuracy_ci_high"] - m for m, r in zip(means, rows)]]
        ax.barh(names, means, xerr=err, color="#4c72b0")
        ax.axvline(0.5, color="crimson", ls="--", lw=1,
                   label="chance / frame-2 ceiling")
        ax.set_title(f"{family} family")
        ax.set_xlabel("accuracy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def ink_curve(model_path=None, path="figures/outline_ink_curve.png"):
    """Accuracy against ink-mass ratio for outline renders at several stroke
    widths. A single outline number cannot distinguish a form-transfer failure
    from an out-of-distribution input; a curve can.
    """
    ratios, accuracies = [], []
    filled_ref = None
    for width in (1, 2, 3, 4):
        env = make_env("correspondence", split="test_style", img_size=64,
                       stroke_width=width)
        env.seed(0)
        sample = sampler_for("test_style", stroke_width=width)
        rng = np.random.default_rng(0)
        masses = [ink_mass(render_trial(sample(rng)).frame1) for _ in range(200)]
        if filled_ref is None:
            fill_sample = sampler_for("train")
            filled_ref = np.mean(
                [ink_mass(render_trial(fill_sample(rng)).frame1)
                 for _ in range(200)],
            )
        ratios.append(np.mean(masses) / filled_ref)
        # accuracy is filled in from results/sweep.json per stroke width if the
        # sweep was run with that width; otherwise re-evaluate here.
        accuracies.append(float("nan"))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ratios, accuracies, "o-")
    ax.axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
    ax.set_xlabel("outline ink mass / filled ink mass")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    data = json.load(open("results/sweep.json"))
    accuracy_figure(data["summary"])
    ink_curve()
```

Run: `.venv/bin/python scripts/plot_results.py`

- [ ] **Step 6: Update the README with the measured result**

Add a "Correspondence task" section reporting: the analytic frame-2 ceiling, the measured frame-1 floor, `RNNNet` accuracy with CI on both families, and the cross-family comparison that quantifies the 2023 defect. Link the new figures.

- [ ] **Step 7: Run the whole suite one last time**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pytest -m slow -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/hf_sweep.py scripts/plot_results.py results/sweep.json figures/ README.md
git commit -m "$(cat <<'EOF'
feat: run the correspondence sweep on HF Jobs and report results

Five seeds x six architectures x two families at 64px on an l4x1. Stimuli
are generated from a seed inside the job, so nothing is uploaded.

Reports the outline result as a curve over ink-mass ratio rather than a
single number, because one number cannot separate a form-transfer failure
from an input that is simply too far out of distribution to drive the
network.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HF2ejFihqgzzp4wWng2Ftf
EOF
)"
```

---

## Self-Review

**Spec coverage.** §1 findings → Tasks 1, 2. §3 balance → Tasks 4, 8. §5.1 generator → Tasks 3–6. §5.2 env → Task 7. §5.3 models → Task 9. §5.4 sweep → Task 11. §5.5 HF script → Task 13. §6 metrics → Task 1. §7 splits → Task 6. §8 checkpoints → Tasks 8, 10, 12, 13. §9 testing → every task. §10 integration → Tasks 2, 7. No gaps.

**Known risks flagged inline rather than hidden:**

- Task 6's disjointness test is expected to fail on the first implementation; the fix is written into the task rather than left for the implementer to discover.
- Task 7's `frame1_index` may be off by one because of neurogym's reset behaviour; the debugging step is written in.
- Task 11 requires an `input_transform` hook in `train()`/`evaluate()` that does not exist yet; adding it is part of the task.
- Task 13's `ink_curve` leaves accuracy as NaN unless the sweep is run per stroke width. If the checkpoint-5 curve is wanted in one pass, extend the Task 13 grid with `stroke_width` as a swept dimension.

**Type consistency.** `EvalResult` fields (Task 1) are read by name in Tasks 10, 11. `TrialStimulus.frame1/frame2/label/params` (Task 4) are read in Tasks 5, 7, 8, 13. `env.frame1_index`/`frame2_indices` (Task 7) are read in Tasks 9, 10, 11. `run_grid`/`aggregate` signatures (Task 11) match their calls in Tasks 12, 13. `sampler_for(split, **overrides)` (Task 6) matches its calls in Tasks 7, 8, 13.
