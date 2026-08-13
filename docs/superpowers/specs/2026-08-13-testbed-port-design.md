# illusion-rnn Testbed Port — Design Spec

**Date:** 2026-08-13
**Branch:** `testbed`
**Status:** Approved design, pending implementation plan

## 1. Goal

Turn the 2022–2023 research repo into a usable testbed for studying Transformational
Apparent Motion (TAM) in neural networks, ported to current neurogym. After this work,
a stranger can:

1. `pip install git+https://github.com/sh4r11f/illusion-rnn` (or clone + `uv sync`) and
   construct TAM and real-motion task environments with zero path configuration.
2. Plug any sequence model into a supervised train/evaluate pipeline
   (`neurogym.Dataset` batches in, per-condition accuracy and hidden activity out).
3. Reproduce the headline result — an RNN trained on unambiguous motion generalizing
   (or not) to TAM stimuli — from a shipped checkpoint via a quickstart notebook.

The reusable contribution is the **task paradigm + stimuli**, not the reference CTRNN.
The design treats models as replaceable and environments as the product.

## 2. Verified environment facts (from API spike, 2026-08-13)

- neurogym **2.3.1**, gymnasium **0.29.1**, numpy **2.2.6**, torch **2.12.0** coexist on
  Python 3.11 (repo floor: **>=3.10**, matching neurogym).
- `TrialEnv` now lives at `neurogym.core.TrialEnv` (gone from top-level namespace).
- `TrialEnv.add_period / add_ob / add_randn / set_groundtruth / in_period / view_ob`,
  `ob_now` / `gt_now`, `ngym.Dataset(env, batch_size=, seq_len=)`, and
  `ngym.spaces.Box / Discrete(..., name=...)` are **unchanged** from the 2022 usage.
- `reset()` is gymnasium-style: signature `(seed=None, options=None)`, returns
  `(ob, info)`. The old `reset(no_step=True)` kwarg is gone; `reset()` already builds
  the first trial (`env.ob` / `env.gt` populated after it).
- `_step` must return the 5-tuple `(ob, reward, terminated, truncated, info)`.
- `TrialEnv.seed(seed)` provides `self.rng` (`np.random.RandomState`); stimulus sampling
  must go through it, not global `np.random` / `random`.
- `ngym.Dataset` deep-copies the env (`dataset.env is not env`), same as before.
- Local machine: uv 0.10.10 available; torch MPS available (matters for retraining).
- Git LFS objects for all 4 `.pt` files + `shape_dataset.zip` exist on GitHub and are
  fetched into `.git/lfs/objects`, but `git lfs install` has never been run locally, so
  smudge/checkout is not configured. Implementation must run `git lfs install --local`
  and `git lfs checkout` before anything needs real checkpoint bytes.

## 3. Approach (decided)

**Self-contained package.** Environments, stimulus JPGs (~12 MB), and model code ship
inside an installable `illusion_rnn` package (flat layout at repo root). Checkpoints
stay outside the package in `checkpoints/` (git LFS — pip-install-from-git users without
LFS get pointer files; documented in README). Legacy notebooks are **archived** in
`notebooks/legacy/` untouched (user decision), with a disclaimer README.

## 4. Target layout

```
illusion_rnn/                  # installable package (flat layout)
  __init__.py                  # re-exports: TAMTask, MotionTask, loaders, models, train, evaluate
  envs.py                      # TAMTask, MotionTask (ported TrialEnvs)
  stimuli.py                   # stimulus loaders (importlib.resources; path override)
  models.py                    # CTRNN, RNNNet, ShapesCNN
  training.py                  # make_env, make_dataset, train, evaluate
  plotting.py                  # plot_trials, plot_training_curves
  stimuli_data/                # package data (moved from data/stimuli/)
    TAM_task/ TAM_basic/ TAM_outline/ motion_task/   # *.jpg
tests/
  test_envs.py test_stimuli.py test_models.py test_training.py test_checkpoints.py
checkpoints/                   # renamed .pt files (LFS) + MANIFEST.md
  legacy/                      # rnn_v13, rnn_v13_state_dict (early prototype)
notebooks/
  01_quickstart.ipynb          # env → visualize → load checkpoint → TAM generalization
  02_train.ipynb               # train small RNN from scratch, plot curves
  legacy/                      # 12 original notebooks + README.md disclaimer
scripts/
  train_shape_cnn.py           # cleaned, attributed shapes-CNN training (optional)
data/
  shape_dataset/               # shape_dataset.zip (LFS) + dataset.csv (script-only input)
docs/superpowers/specs/        # this spec
figures/                       # README images (keep; notebooks/training_curve moves here as training_curve.png)
.github/workflows/test.yml     # CI: pytest on 3.10 and 3.12
pyproject.toml                 # hatchling build; uv-managed
README.md  LICENSE  .gitignore  .gitattributes
```

**Deleted on this branch:** `src/` (superseded by package), `scripts/make_datasets.py`
and `scripts/train_models.py` (pickled-dataset workflow dies; envs regenerate cheaply),
`scripts/CNN_train.py`, `scripts/CNN_test*.ipynb` (superseded by cleaned script +
notebooks), `parameters.json` (defaults live in function signatures), committed `.idea/`
files, `TODO.md` (items either done here or captured in README "Limitations").
Moves use `git mv` where content is preserved (stimuli, notebooks, checkpoints).

## 5. Environments (`envs.py`)

Port `TAMTask` and `MotionTask` with behavior identical to what trained the shipped
checkpoints, except for two deliberate bug fixes.

**API deltas applied:** import from `neurogym.core`; drop `reset(no_step=True)` call
sites (plain `reset()`); `_step` returns 5-tuple; all stimulus/direction sampling uses
`self.rng`.

**Frozen label semantics** (checkpoint compatibility — do not change):

- `TAMTask`: `{fixation: 0, left: 1, middle: 2, right: 3, down: 4, up: 5}`,
  `Discrete(6)`. Horizontal trials draw direction from `{1, 2, 3}`; vertical from
  `{2, 4, 5}`. Ground truth is 0 during fixation + frame1, and the trial direction from
  frame2 through decision. Trial structure: fixation 100 ms, frames 1–5 at 50 ms each,
  decision 100 ms; dt = 50 ms.
- `MotionTask`: `{no_motion: 0, left: 1, middle: 2, right: 3, down: 4, up: 5,
  fixation: 6}`, `Discrete(7)`. Ground truth 6 outside decision, trial direction during
  decision (150 ms). Horizontal draws from `{0, 1, 2, 3}`; vertical from `{0, 2, 4, 5}`.

**Bug fix 1 — vertical rotation.** Old code rotated the stimulus dict in place on every
`_new_trial`, so successive vertical trials cycled through 90°/180°/270° rotations while
labels stayed fixed. New behavior: rotate once at env construction. Horizontal behavior
(all shipped checkpoints) is unaffected.

**Bug fix 2 — `TAMTask._step` fixation index.** Old code checked `action != 6` but
TAMTask's fixation action is 0. Fix to the env's own mapping. The supervised pipeline
never calls `_step`, so checkpoints are unaffected; the fix makes the RL interface
coherent.

**Constructor surface** (both classes): `dt`, `box_shape` (`square|circle|triangle`),
`stim_ori` (`horizontal|vertical`), `sigma` (frame-period Gaussian noise), `img_size`,
`rewards`, `timing`, plus `stimuli` (a loaded stimulus dict). `TAMTask` takes
`variant` (`standard|basic|outline` → stimulus sets `TAM_task|TAM_basic|TAM_outline`);
`MotionTask` takes `motion_type` (`continuous|tracking` → filename prefixes
`cnt|track`). When `stimuli` is None (the default), the env loads the packaged set for
its variant/motion_type via `stimuli.py`; passing a dict overrides (custom stimuli). The unused `cnn` flag, property/setter boilerplate, dead
`_make_second_order_horizontal_stim`, and empty `_trial_type_cnn` are deleted.

## 6. Stimuli (`stimuli.py`)

`load_tam(img_size, variant, source=None)` and
`load_motion(img_size, motion_type, source=None)` reproduce the old loaders' output
format (dict of grayscale float arrays in [0, 1], ink = 1, background = 0). Default
`source` resolves inside the package via `importlib.resources`; passing a `Path`
overrides (custom stimulus sets — this is the testbed extension point). Loaders raise
a clear error listing expected filename patterns when a directory doesn't match.

## 7. Models (`models.py`)

- `CTRNN` and `RNNNet` carried over as-is (docstrings intact; they are clean).
- `ShapesCNN` replaces `CNNNet`, **matching the shipped checkpoint**: conv stack →
  fc 10368→4096→1024→**64**→9, expecting 100×100 grayscale input. `feature_dim` and
  input size become documented constructor parameters with checkpoint-matching
  defaults; `forward` returns `(logits, features)`.
- Deleted: the `TAMNet` god-class, its broken pretrained-AlexNet path
  (`torch.hub` AlexNet + 3-channel normalize on grayscale + `hidden_size`-shaped
  feature buffer), and the hardcoded `reshape(1, 1, 100, 100)` hacks.

## 8. Training / evaluation (`training.py`)

Plain functions, no class state:

- `make_env(task, **kwargs) -> TrialEnv` — convenience dispatcher
  (`task in {"tam", "motion"}`) that loads stimuli and constructs the env.
- `make_dataset(env, batch_size=16, seq_len=100) -> ngym.Dataset`.
- `train(model, datasets, n_epochs, lr=5e-4, device=None, encoder=None, log_every=100)
  -> dict` with per-epoch loss and accuracy. Multi-dataset batches concatenate along
  the batch axis (as before). `encoder` is an optional callable
  `(T, B, H, W) array -> (T, B, F) array` applied to inputs before the model — the
  generalization of the old hardcoded CNN-features path. A provided
  `cnn_encoder(shapes_cnn, device)` factory implements it for `ShapesCNN`
  (batched over T×B, no per-image Python loops).
- `evaluate(model, env, n_trials, device=None, encoder=None) -> EvalResult` with
  per-trial ground truth / choice / correct, hidden activity arrays, and overall
  accuracy. Choice = argmax of the final decision timestep, as before.
- Loss: `CrossEntropyLoss` over all timesteps (labels flattened), matching how the
  checkpoints were trained. Early stopping stays out (was dead code).
- sklearn dependency dropped (accuracy computed with torch/numpy).

## 9. Checkpoints

Renames (git mv of LFS pointers) + `checkpoints/MANIFEST.md` documenting each file's
architecture, input format, label map, training data, and provenance:

| Old | New | What it is |
|---|---|---|
| `learned_horiz_tam_all.pt` | `rnn-pixel_h2048_tam-horiz.pt` | **Flagship.** RNN on raw 64×64 pixels (4096→2048→6), horizontal TAM |
| `CNN_horiz_tam.pt` | `rnn-cnnfeat64_h1024_tam-horiz.pt` | RNN on 64-d ShapesCNN features (64→1024→6) |
| `CNN_horiz_tam_2048.pt` | `rnn-cnnfeat64_h2048_tam-horiz.pt` | RNN on 64-d ShapesCNN features (64→2048→6) |
| `cnn_model_64.pt` | `cnn-shapes_feat64_100px.pt` | ShapesCNN (9-class, 64-d features, 100×100 input) |
| `rnn_v13`, `rnn_v13_state_dict` | `legacy/` (unchanged names) | Early 32×32 / 4-action prototype; kept for the legacy notebooks |

**Acceptance gate:** under the ported envs, the flagship must score **> 0.8 accuracy on
horizontal standard TAM** (its training condition, all three shapes, ≥ 300 trials).
Outline/basic generalization is reported, not gated. If the gate fails (port shifted the
stimulus pipeline), retrain the flagship on MPS with the known-good recipe
(img 64, batch 16, seq len 100, hidden 2048, lr 5e-4, ≤ 1000 epochs), ship the fresh
weights under the same name, and note the retrain in the manifest. The CNN-feature
checkpoints are evaluated and reported in the manifest but not gated.

## 10. Notebooks

- `01_quickstart.ipynb`: install pointer, build TAM + motion envs, visualize sample
  trials per condition, load flagship checkpoint, evaluate on standard TAM →
  outline → basic (the generalization story), plot per-condition accuracy and a
  hidden-activity glance. Runs in a few minutes on CPU.
- `02_train.ipynb`: train a small pixel RNN from scratch (reduced hidden size / epochs
  for demo speed), plot loss/accuracy curves, save + reload the state dict, evaluate.
- `notebooks/legacy/README.md`: one paragraph — these are the untouched 2022–2023
  research notebooks, written against neurogym 0.x, kept as the experimental record;
  they will not run against this package.
- Both new notebooks are committed **executed** (outputs in), and must run end-to-end
  via `jupyter nbconvert --execute` as part of final verification.

## 11. Tests & CI

Pytest, total runtime < ~1 min CPU:

- `test_stimuli.py`: loaders return expected keys/shapes/value ranges for every
  variant; path-override works; helpful error on bad directory.
- `test_envs.py`: construction across task × variant × orientation × shape grid;
  ob/gt shapes and dtype; label values match the frozen maps; period timing; seeded
  env reproduces identical trials; vertical envs produce stable (non-cycling) stimuli
  across trials; `step()` returns 5-tuple and gymnasium `Env` contract holds.
- `test_models.py`: forward shapes for RNNNet and ShapesCNN; CTRNN alpha math.
- `test_training.py`: one tiny `train()` run (few epochs, small hidden) decreases loss;
  `evaluate()` returns coherent structures; encoder hook applies.
- `test_checkpoints.py`: each shipped checkpoint loads into its documented
  architecture; flagship scores above chance on a small seeded eval. **Skips cleanly**
  (pytest.skip) when files are LFS pointers, so CI needs no LFS bandwidth.

CI: `.github/workflows/test.yml` — uv setup, `uv sync`, `pytest`, matrix on Python
3.10 and 3.12, no LFS fetch.

## 12. Packaging

- `pyproject.toml`, hatchling backend, flat layout, package data (JPGs) included.
- Runtime deps: `neurogym>=2.3`, `torch>=2.0`, `numpy`, `matplotlib`, `pillow`.
  Dropped: sklearn, seaborn, opencv, pandas, efficientnet_pytorch, docopt-style
  strings. Extras: `dev` (pytest, jupyter, nbconvert, pandas — pandas only for
  `train_shape_cnn.py`).
- `uv.lock` committed for reproducible dev setup.
- LFS patterns in `.gitattributes` unchanged (`*.pt`, `*.zip`).

## 13. Docs / README

README covers: what TAM is and what the testbed tests (2–3 paragraphs, one stimulus
figure); install (uv clone path and pip-from-git path, with the LFS note for
checkpoints); quickstart code snippet (build env → dataset → evaluate checkpoint);
the generalization result with a figure; how to plug in your own model (the
`encoder` / `RNNNet`-shaped contract) and your own stimuli (loader `source` override);
limitations (small hand-made stimulus set, few exemplars per condition); attributions —
neurogym (cite Molano-Mazón et al.), the 2D geometric shapes dataset (El Korchi &
Ghanou) used to train the ShapesCNN, and a note that `train_shape_cnn.py` derives from
an adapted classroom implementation; MIT license.

## 14. Out of scope

PyPI publication, retraining the full model zoo, new stimulus generation, the
second-order (random-dot) stimulus generator, RL-mode training (envs keep a coherent
`_step`, but the pipeline is supervised), and any change to `main`.

## 15. Acceptance criteria

1. Fresh clone of `testbed` + `uv sync` + `pytest` → all green (checkpoint tests
   active locally after `git lfs install --local && git lfs checkout`).
2. `01_quickstart.ipynb` and `02_train.ipynb` execute end-to-end via nbconvert.
3. Flagship checkpoint gate from §9 met (or flagship retrained and gate met).
4. README complete per §13; MANIFEST.md documents all shipped checkpoints.
5. Legacy notebooks and prototype checkpoints archived per §4/§9; no `src/`,
   `parameters.json`, `.idea/`, or dead code paths remain on the branch.
