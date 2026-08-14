# illusion-rnn Testbed Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 2022-era research repo into an installable `illusion_rnn` package (task environments + stimuli + reference models) ported to neurogym 2.x, with tests, two canonical notebooks, renamed checkpoints, and a real README.

**Architecture:** Flat-layout Python package `illusion_rnn/` at repo root containing neurogym `TrialEnv` subclasses (`TAMTask`, `MotionTask`), stimulus loaders reading packaged JPGs, reference models (CTRNN/RNNNet/ShapesCNN), and plain train/evaluate functions. Checkpoints live outside the package in `checkpoints/` (git LFS). Legacy research code is deleted; legacy notebooks archived untouched.

**Tech Stack:** Python >=3.10, neurogym >=2.3 (gymnasium-based), torch >=2.0, numpy, matplotlib, pillow; uv for env management; hatchling build backend; pytest; GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-13-testbed-port-design.md` (read it before starting a task if anything is unclear).

## Global Constraints

- Work on branch `testbed`. Never touch `main`.
- Python floor **>=3.10**. Runtime deps exactly: `neurogym>=2.3`, `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`, `pillow>=10.0`. No sklearn, seaborn, opencv, pandas, or efficientnet_pytorch at runtime.
- **Frozen label maps** (checkpoint compatibility — never change):
  - TAMTask: `{"fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5}`, `Discrete(6)`.
  - MotionTask: `{"no_motion": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5, "fixation": 6}`, `Discrete(7)`.
- neurogym 2.x facts (verified by spike): import `TrialEnv` from `neurogym.core`; `reset()` takes `(seed=None, options=None)` and returns `(ob, info)` — there is NO `no_step` kwarg; `_step` must return the 5-tuple `(ob, reward, terminated, truncated, info)`; `self.rng` (a `np.random.RandomState`) exists after construction and is replaced by `env.seed(n)`; `add_period/add_ob/add_randn/set_groundtruth/in_period`, `ob_now`/`gt_now`, `ngym.Dataset(env, batch_size=, seq_len=)`, and `ngym.spaces.Box/Discrete(..., name=...)` work exactly as in the old code.
- All randomness inside envs goes through `self.rng` — never `np.random.*` or `random.*`.
- Run everything through uv: `uv run pytest ...`, `uv run python ...`, `uv run jupyter ...`.
- `*.pt` and `*.zip` are LFS-tracked (`.gitattributes`) — never write throwaway `.pt` files inside the repo tree; use `tempfile` for scratch weights.
- Every commit message ends with these two trailer lines:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ
  ```
- Old reference code lives in `src/` until Task 10 deletes it; you may read it, never import it.

---

### Task 1: Packaging scaffold + LFS materialization

**Files:**
- Create: `pyproject.toml`, `illusion_rnn/__init__.py`
- Modify: (none)
- Test: none (scaffold; verified by `uv run python -c "import illusion_rnn"`)

**Interfaces:**
- Consumes: nothing.
- Produces: an installable (editable) package `illusion_rnn` with `__version__`; a synced `.venv` with dev tools; materialized LFS checkpoint bytes in `data/models/*.pt`; committed `uv.lock`.

- [ ] **Step 1: Configure git LFS locally and materialize checkpoint bytes**

```bash
cd /Users/devxi/projects/illusion-rnn
git lfs install --local
git lfs checkout
du -h data/models/*.pt data/stimuli/shape_dataset/shape_dataset.zip
```

Expected: `git lfs checkout` reports checking out 5 files; `du` shows real sizes (~1–5 MB per `.pt`, ~87 MB zip), not 4.0K pointers. `git status` must remain clean afterward (smudged LFS files still match the index).

- [ ] **Step 2: Create the package skeleton**

Create `illusion_rnn/__init__.py`:

```python
"""illusion-rnn: a neurogym testbed for Transformational Apparent Motion (TAM)."""

__version__ = "1.0.0"
```

(Module re-exports are added in Task 7 once the modules exist.)

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "illusion-rnn"
version = "1.0.0"
description = "A neurogym testbed for studying Transformational Apparent Motion (TAM) in recurrent neural networks"
readme = "README.md"
license = { file = "LICENSE" }
authors = [{ name = "Sharif Saleki" }]
requires-python = ">=3.10"
dependencies = [
    "neurogym>=2.3",
    "torch>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "pillow>=10.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "jupyter",
    "nbconvert",
    "ipykernel",
    "pandas",
]

[tool.hatch.build.targets.wheel]
packages = ["illusion_rnn"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Sync the environment and verify the package imports**

```bash
uv sync
uv run python -c "import illusion_rnn, neurogym; print(illusion_rnn.__version__, neurogym.__version__)"
```

Expected: prints `1.0.0 2.3.1` (or a newer 2.x neurogym). `uv sync` creates `.venv/` and `uv.lock`. If `uv sync` fails on the README reference, README.md already exists (one line) — that is fine; do not create it here.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock illusion_rnn/__init__.py
git commit -m "Add installable package scaffold (pyproject, uv)" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 2: Stimulus data move + loaders

**Files:**
- Move (git mv): `data/stimuli/TAM_task/`, `data/stimuli/TAM_basic/`, `data/stimuli/TAM_outline/`, `data/stimuli/motion_task/` → `illusion_rnn/stimuli_data/<same name>/`
- Create: `illusion_rnn/stimuli.py`, `tests/__init__.py` (empty), `tests/conftest.py`, `tests/test_stimuli.py`
- Test: `tests/test_stimuli.py`

**Interfaces:**
- Consumes: packaged JPGs under `illusion_rnn/stimuli_data/`.
- Produces:
  - `load_tam(img_size: int, variant: str = "standard", source: Path | None = None) -> dict[str, list[np.ndarray]]` — keys `f"{shape}-{prefix}-{motion}"` with `shape ∈ {square, circle, triangle}`, `prefix = "outline" if variant == "outline" else "tam"`, `motion ∈ {no_motion, left, right, middle, init}`; values are lists of `(img_size, img_size)` float arrays in [0, 1], ink = 1.
  - `load_motion(img_size: int, motion_type: str = "continuous", source: Path | None = None) -> dict[str, list[list[np.ndarray]]]` — keys `f"{shape}-{prefix}-{direction}"` with `prefix ∈ {cnt, track}` (from `continuous|tracking`), `direction ∈ {left, right, middle}`; values are lists of exemplars, each a list of 5 frames.
  - Constants: `SHAPES = ("square", "circle", "triangle")`, `TAM_MOTIONS = ("no_motion", "left", "right", "middle", "init")`, `MOTION_DIRECTIONS = ("left", "right", "middle")`, `TAM_VARIANTS = ("standard", "basic", "outline")`, `MOTION_TYPES = ("continuous", "tracking")`.

- [ ] **Step 1: Move the stimulus directories into the package**

```bash
mkdir -p illusion_rnn/stimuli_data
git mv data/stimuli/TAM_task illusion_rnn/stimuli_data/TAM_task
git mv data/stimuli/TAM_basic illusion_rnn/stimuli_data/TAM_basic
git mv data/stimuli/TAM_outline illusion_rnn/stimuli_data/TAM_outline
git mv data/stimuli/motion_task illusion_rnn/stimuli_data/motion_task
ls illusion_rnn/stimuli_data
```

Expected: four directories listed; `data/stimuli/` now contains only `shape_dataset/`.

- [ ] **Step 2: Write the failing tests**

Create `tests/__init__.py` (empty file), `tests/conftest.py`:

```python
import matplotlib

matplotlib.use("Agg")
```

Create `tests/test_stimuli.py`:

```python
import numpy as np
import pytest

from illusion_rnn.stimuli import (
    MOTION_DIRECTIONS,
    SHAPES,
    load_motion,
    load_tam,
)


@pytest.mark.parametrize(
    "variant,prefix",
    [("standard", "tam"), ("basic", "tam"), ("outline", "outline")],
)
def test_load_tam_variants(variant, prefix):
    stims = load_tam(24, variant=variant)
    for key, frames in stims.items():
        for frame in frames:
            assert frame.shape == (24, 24)
            assert frame.dtype == np.float64
            assert 0.0 <= frame.min() and frame.max() <= 1.0
    for shape in SHAPES:
        for motion in ("init", "left", "right", "middle"):
            assert len(stims[f"{shape}-{prefix}-{motion}"]) >= 1


@pytest.mark.parametrize(
    "motion_type,prefix",
    [("continuous", "cnt"), ("tracking", "track")],
)
def test_load_motion_types(motion_type, prefix):
    stims = load_motion(24, motion_type=motion_type)
    for shape in SHAPES:
        for direction in MOTION_DIRECTIONS:
            key = f"{shape}-{prefix}-{direction}"
            assert key in stims
            assert len(stims[key]) >= 1
            for exemplar in stims[key]:
                assert len(exemplar) == 5
                for frame in exemplar:
                    assert frame.shape == (24, 24)


def test_ink_is_one():
    # Stimulus JPGs are dark ink on white; loader inverts so ink ~= 1.
    stims = load_tam(48, variant="standard")
    frame = stims["square-tam-init"][0]
    # background (corner) should be near 0, some ink pixels near 1
    assert frame[0, 0] < 0.2
    assert frame.max() > 0.7


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        load_tam(24, variant="bogus")
    with pytest.raises(ValueError, match="motion_type"):
        load_motion(24, motion_type="bogus")


def test_empty_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .jpg stimuli"):
        load_tam(24, source=tmp_path)


def test_bad_filename_raises(tmp_path):
    (tmp_path / "bogus-file.jpg").write_bytes(b"junk")
    with pytest.raises(ValueError, match="expected"):
        load_tam(24, source=tmp_path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_stimuli.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'illusion_rnn.stimuli'`.

- [ ] **Step 4: Implement `illusion_rnn/stimuli.py`**

```python
"""Loaders for the packaged TAM and real-motion stimulus frames.

Stimulus files are grayscale JPGs, dark ink on white background. Loaders
resize to ``img_size`` x ``img_size``, normalize to [0, 1], and invert so
ink = 1 and background = 0 (the convention the environments and shipped
checkpoints were trained with).

Filename conventions:
    TAM sets:    ``<shape>-<tam|outline>-<motion>-<n>.jpg``
    Motion sets: ``<shape>-<cnt|track>-<direction>-<n>-f<m>.jpg``
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image

SHAPES = ("square", "circle", "triangle")
TAM_MOTIONS = ("no_motion", "left", "right", "middle", "init")
MOTION_DIRECTIONS = ("left", "right", "middle")
TAM_VARIANTS = ("standard", "basic", "outline")
MOTION_TYPES = ("continuous", "tracking")

# variant -> (packaged directory, filename prefix)
_TAM_VARIANT_MAP = {
    "standard": ("TAM_task", "tam"),
    "basic": ("TAM_basic", "tam"),
    "outline": ("TAM_outline", "outline"),
}
# motion_type -> filename prefix (all live in motion_task/)
_MOTION_TYPE_MAP = {"continuous": "cnt", "tracking": "track"}


def _default_source(dirname: str) -> Path:
    return Path(str(files("illusion_rnn") / "stimuli_data" / dirname))


def _read_frame(path: Path, img_size: int) -> np.ndarray:
    img = Image.open(path).resize((img_size, img_size))
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return 1.0 - arr / 255.0


def _list_jpgs(source: Path) -> list[Path]:
    jpgs = sorted(source.glob("*.jpg"))
    if not jpgs:
        msg = f"No .jpg stimuli found in {source}"
        raise FileNotFoundError(msg)
    return jpgs


def load_tam(
    img_size: int,
    variant: str = "standard",
    source: Path | str | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load a TAM stimulus set as ``{"<shape>-<prefix>-<motion>": [frames]}``."""
    if variant not in _TAM_VARIANT_MAP:
        msg = f"Unknown variant {variant!r}; expected one of {TAM_VARIANTS}"
        raise ValueError(msg)
    dirname, prefix = _TAM_VARIANT_MAP[variant]
    src = Path(source) if source is not None else _default_source(dirname)

    stims: dict[str, list[np.ndarray]] = {
        f"{shape}-{prefix}-{motion}": []
        for shape in SHAPES
        for motion in TAM_MOTIONS
    }
    for jpg in _list_jpgs(src):
        base = "-".join(jpg.stem.split("-")[:-1])
        if base not in stims:
            msg = (
                f"Unexpected stimulus file {jpg.name!r}; expected "
                f"'<shape>-{prefix}-<motion>-<n>.jpg' with shape in {SHAPES} "
                f"and motion in {TAM_MOTIONS}"
            )
            raise ValueError(msg)
        stims[base].append(_read_frame(jpg, img_size))
    return stims


def load_motion(
    img_size: int,
    motion_type: str = "continuous",
    source: Path | str | None = None,
) -> dict[str, list[list[np.ndarray]]]:
    """Load a motion stimulus set as ``{"<shape>-<prefix>-<dir>": [exemplars]}``,
    each exemplar an ordered list of frames."""
    if motion_type not in _MOTION_TYPE_MAP:
        msg = f"Unknown motion_type {motion_type!r}; expected one of {MOTION_TYPES}"
        raise ValueError(msg)
    prefix = _MOTION_TYPE_MAP[motion_type]
    src = Path(source) if source is not None else _default_source("motion_task")

    raw: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for jpg in _list_jpgs(src):
        parts = jpg.stem.split("-")
        if len(parts) != 5 or parts[1] not in _MOTION_TYPE_MAP.values():
            msg = (
                f"Unexpected stimulus file {jpg.name!r}; expected "
                f"'<shape>-<cnt|track>-<direction>-<n>-f<m>.jpg'"
            )
            raise ValueError(msg)
        if parts[1] != prefix:
            continue  # other motion type, same directory
        base = "-".join(parts[:3])
        try:
            exemplar = int(parts[3])
            frame = int(parts[4].lstrip("f"))
        except ValueError as err:
            msg = (
                f"Unexpected stimulus file {jpg.name!r}; expected "
                f"'<shape>-<cnt|track>-<direction>-<n>-f<m>.jpg'"
            )
            raise ValueError(msg) from err
        raw.setdefault(base, {})[(exemplar, frame)] = _read_frame(jpg, img_size)

    stims: dict[str, list[list[np.ndarray]]] = {}
    for base, frames in raw.items():
        exemplar_ids = sorted({key[0] for key in frames})
        stims[base] = [
            [frames[key] for key in sorted(frames) if key[0] == ex_id]
            for ex_id in exemplar_ids
        ]
    return stims
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_stimuli.py -v`
Expected: all PASS. If `test_load_motion_types` fails on `len(exemplar) == 5`, inspect actual frame counts with `uv run python -c "from illusion_rnn.stimuli import load_motion; s = load_motion(24); print({k: [len(e) for e in v] for k, v in s.items()})"` — every exemplar in the shipped set has frames f1–f5; a mismatch means the filename parsing is wrong, not the data.

- [ ] **Step 6: Commit**

```bash
git add -A illusion_rnn data tests
git commit -m "Move stimuli into package; add loaders with tests" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 3: Models module

**Files:**
- Create: `illusion_rnn/models.py`, `tests/test_models.py`
- Reference (read-only): `src/networks.py:33-197` (old CTRNN/RNNNet to carry over)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `CTRNN(input_size: int, hidden_size: int, dt: float | None = None, tau: float = 100)` — `forward(x, hidden=None) -> (output, hidden)` with `x: (T, B, input_size)`, `output: (T, B, hidden_size)`.
  - `RNNNet(input_size: int, hidden_size: int, output_size: int, dt: float | None = None, tau: float = 100)` — `forward(x) -> (out, rnn_activity)` with `out: (T, B, output_size)`, `rnn_activity: (T, B, hidden_size)`; submodules named `rnn` (CTRNN with `input2h`, `h2h`) and `fc` (checkpoint state-dict compatibility).
  - `ShapesCNN(input_size: int = 100, feature_dim: int = 64, n_classes: int = 9)` — `forward(x) -> (logits, features)` with `x: (B, 1, input_size, input_size)`, `logits: (B, n_classes)`, `features: (B, feature_dim)` (pre-ReLU fc3 output); static `_flat_dim(size: int) -> int`.
  - `load_rnn(path, dt: float = 50, map_location: str = "cpu") -> RNNNet` — infers sizes from the state dict, returns the model in eval mode.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
import torch

from illusion_rnn.models import CTRNN, RNNNet, ShapesCNN, load_rnn


def test_ctrnn_alpha():
    assert CTRNN(4, 8, dt=50).alpha == 0.5
    assert CTRNN(4, 8).alpha == 1


def test_rnnnet_shapes():
    model = RNNNet(input_size=64, hidden_size=32, output_size=6, dt=50)
    x = torch.randn(9, 4, 64)
    out, activity = model(x)
    assert out.shape == (9, 4, 6)
    assert activity.shape == (9, 4, 32)


def test_rnnnet_state_dict_layout():
    # Frozen layout: shipped checkpoints use these exact key names.
    keys = set(RNNNet(64, 32, 6, dt=50).state_dict())
    assert keys == {
        "rnn.input2h.weight",
        "rnn.input2h.bias",
        "rnn.h2h.weight",
        "rnn.h2h.bias",
        "fc.weight",
        "fc.bias",
    }


def test_shapes_cnn_flat_dim():
    assert ShapesCNN._flat_dim(100) == 10368


def test_shapes_cnn_shapes():
    model = ShapesCNN()
    x = torch.randn(2, 1, 100, 100)
    logits, features = model(x)
    assert logits.shape == (2, 9)
    assert features.shape == (2, 64)


def test_shapes_cnn_small_input():
    model = ShapesCNN(input_size=32, feature_dim=16)
    logits, features = model(torch.randn(3, 1, 32, 32))
    assert logits.shape == (3, 9)
    assert features.shape == (3, 16)


def test_load_rnn_roundtrip(tmp_path):
    model = RNNNet(16, 8, 6, dt=50)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_rnn(path)
    assert not loaded.training  # eval mode
    assert loaded.rnn.input2h.in_features == 16
    assert loaded.rnn.h2h.in_features == 8
    assert loaded.fc.out_features == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'illusion_rnn.models'`.

- [ ] **Step 3: Implement `illusion_rnn/models.py`**

Carry `CTRNN` and `RNNNet` over from `src/networks.py:33-197` with these deltas: keep the docstrings; add a `tau` constructor parameter (default 100) replacing the hardcoded `self.tau = 100`; type hints `torch.Tensor` instead of `torch.tensor`; no other behavior changes. Then add `ShapesCNN` and `load_rnn` as below. The full module:

```python
"""Reference models: continuous-time RNN and the shapes-CNN feature extractor."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class CTRNN(nn.Module):
    """Continuous-time RNN.

    The hidden state decays toward the driven activation with rate
    ``alpha = dt / tau`` (``alpha = 1`` when ``dt`` is None).

    Parameters
    ----------
    input_size : int
        Number of input units.
    hidden_size : int
        Number of hidden units.
    dt : float, optional
        Simulation time step in ms. If None, ``alpha = 1``.
    tau : float
        Membrane time constant in ms.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float | None = None,
        tau: float = 100,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.alpha = 1 if dt is None else dt / self.tau

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape) -> torch.Tensor:
        """Zero initial hidden state for a ``(T, B, input_size)`` input shape."""
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, act_input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """One time step: leaky integration of the ReLU-driven activation."""
        h_new = torch.relu(self.input2h(act_input) + self.h2h(hidden))
        return hidden * (1 - self.alpha) + h_new * self.alpha

    def forward(self, act_input: torch.Tensor, hidden: torch.Tensor | None = None):
        """Propagate a ``(T, B, input_size)`` sequence; returns
        ``(output (T, B, hidden), final hidden (B, hidden))``."""
        if hidden is None:
            hidden = self.init_hidden(act_input.shape).to(act_input.device)

        output = []
        for t in range(act_input.size(0)):
            hidden = self.recurrence(act_input[t], hidden)
            output.append(hidden)
        return torch.stack(output, dim=0), hidden


class RNNNet(nn.Module):
    """CTRNN with a linear readout.

    ``forward(x)`` takes ``(T, B, input_size)`` and returns
    ``(out (T, B, output_size), rnn_activity (T, B, hidden_size))``.

    Submodule names (``rnn``, ``fc``) are frozen: the shipped checkpoints'
    state dicts use them.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dt: float | None = None,
        tau: float = 100,
    ):
        super().__init__()
        self.rnn = CTRNN(input_size, hidden_size, dt=dt, tau=tau)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity


class ShapesCNN(nn.Module):
    """CNN feature extractor trained on the 2D geometric shapes dataset
    (El Korchi & Ghanou, 2020; 9 classes).

    Defaults match the shipped checkpoint
    ``checkpoints/cnn-shapes_feat64_100px.pt``: 100x100 grayscale input,
    64-d features, 9 classes. ``forward`` returns ``(logits, features)``
    where ``features`` is the pre-ReLU fc3 output — the RNN-on-features
    checkpoints were trained on exactly this tensor.
    """

    def __init__(self, input_size: int = 100, feature_dim: int = 64, n_classes: int = 9):
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self._flat_dim(input_size), 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, feature_dim)
        self.fc4 = nn.Linear(feature_dim, n_classes)

    @staticmethod
    def _flat_dim(size: int) -> int:
        """Flattened conv-stack output size for a square input of ``size``."""
        for kernel in (3, 3, 5):
            size = (size - (kernel - 1)) // 2
        return 128 * size * size

    def forward(self, x: torch.Tensor):
        for conv in (self.conv1, self.conv2, self.conv3):
            x = F.max_pool2d(F.relu(conv(x)), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        features = self.fc3(x)
        logits = self.fc4(F.relu(features))
        return logits, features


def load_rnn(path: Path | str, dt: float = 50, map_location: str = "cpu") -> RNNNet:
    """Load an ``RNNNet`` state dict, inferring layer sizes; returns eval-mode model."""
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    hidden_size, input_size = state_dict["rnn.input2h.weight"].shape
    output_size = state_dict["fc.weight"].shape[0]
    model = RNNNet(input_size, hidden_size, output_size, dt=dt)
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

Note the conv order in `forward`: `conv -> relu -> pool` per conv, matching the original `src/networks.py` CNNNet ordering exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/models.py tests/test_models.py
git commit -m "Add models: CTRNN, RNNNet, ShapesCNN, load_rnn" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 4: TAMTask environment

**Files:**
- Create: `illusion_rnn/envs.py`, `tests/test_envs.py`
- Reference (read-only): `src/environments.py:24-398` (old TAMTask being ported)
- Test: `tests/test_envs.py`

**Interfaces:**
- Consumes: `load_tam`, `SHAPES`, `TAM_VARIANTS` from `illusion_rnn.stimuli` (Task 2).
- Produces:
  - `TAM_CHOICES = {"fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5}` (module constant).
  - `rotate_stimuli(stimuli: dict) -> dict` — returns a new dict with every frame rotated 90° clockwise (`np.rot90(a, k=-1)`), recursing through nested lists.
  - `TAMTask(dt=50, box_shape="square", variant="standard", stim_ori="horizontal", stimuli=None, sigma=0.0, img_size=64, rewards=None, timing=None)` — a `neurogym.core.TrialEnv`. Attributes: `box_shape`, `variant`, `stim_ori`, `sigma`, `img_size`, `ob_shape`, `choice_names` (copy of `TAM_CHOICES`), `observation_space`, `action_space` (`Discrete(6)`). Default trial: 9 steps at dt=50 (fixation 100, frame1–5 at 50 each, decision 100). gt is 0 for fixation+frame1, trial direction for frame2–decision. Horizontal directions {1,2,3}; vertical {2,4,5}.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_envs.py`:

```python
import numpy as np
import pytest

from illusion_rnn.envs import TAM_CHOICES, TAMTask, rotate_stimuli
from illusion_rnn.stimuli import SHAPES, TAM_VARIANTS


def _make_tam(**kwargs):
    defaults = dict(box_shape="square", variant="standard",
                    stim_ori="horizontal", img_size=32)
    defaults.update(kwargs)
    env = TAMTask(**defaults)
    env.seed(0)
    return env


@pytest.mark.parametrize("variant", TAM_VARIANTS)
@pytest.mark.parametrize("stim_ori", ["horizontal", "vertical"])
@pytest.mark.parametrize("shape", SHAPES)
def test_tam_grid(variant, stim_ori, shape):
    env = _make_tam(variant=variant, stim_ori=stim_ori, box_shape=shape)
    ob, _ = env.reset()
    # fixation 100 + 5 frames x 50 + decision 100 = 450 ms; dt 50 -> 9 steps
    assert env.ob.shape == (9, 32, 32)
    assert env.gt.shape == (9,)
    assert env.ob.dtype == np.float32
    # first 3 steps (fixation x2 + frame1) are gt=0; rest are the direction
    assert set(env.gt[:3]) == {TAM_CHOICES["fixation"]}
    allowed = (
        {TAM_CHOICES["left"], TAM_CHOICES["middle"], TAM_CHOICES["right"]}
        if stim_ori == "horizontal"
        else {TAM_CHOICES["middle"], TAM_CHOICES["down"], TAM_CHOICES["up"]}
    )
    assert set(env.gt[3:]) <= allowed
    assert len(set(env.gt[3:])) == 1


def test_action_space_and_labels_frozen():
    env = _make_tam()
    assert env.action_space.n == 6
    assert env.choice_names == {
        "fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5,
    }


def test_seeded_trials_reproduce():
    a = _make_tam()
    b = _make_tam()
    a.seed(123)
    b.seed(123)
    a.reset()
    b.reset()
    for _ in range(3):
        ta = a.new_trial()
        tb = b.new_trial()
        assert ta["ground_truth"] == tb["ground_truth"]
        np.testing.assert_array_equal(a.ob, b.ob)


def test_direction_variability():
    env = _make_tam()
    env.reset()
    gts = {env.new_trial()["ground_truth"] for _ in range(50)}
    assert gts == {1, 2, 3}


def test_vertical_stimuli_do_not_mutate():
    # Regression: old code rotated the stimulus dict in place every trial.
    env = _make_tam(stim_ori="vertical")
    env.reset()
    before = env._stimuli["square-tam-init"][0].copy()
    for _ in range(5):
        env.new_trial()
    np.testing.assert_array_equal(env._stimuli["square-tam-init"][0], before)


def test_vertical_rotates_once():
    h = _make_tam(stim_ori="horizontal")
    v = _make_tam(stim_ori="vertical")
    np.testing.assert_array_equal(
        v._stimuli["square-tam-init"][0],
        np.rot90(h._stimuli["square-tam-init"][0], k=-1),
    )


def test_rotate_stimuli_nested():
    stims = {"a": [np.eye(3)], "b": [[np.eye(3), np.ones((3, 3))]]}
    out = rotate_stimuli(stims)
    np.testing.assert_array_equal(out["a"][0], np.rot90(np.eye(3), k=-1))
    np.testing.assert_array_equal(out["b"][0][1], np.ones((3, 3)))
    # original untouched
    np.testing.assert_array_equal(stims["a"][0], np.eye(3))


def test_step_contract_and_fixation_index():
    # neurogym 2.x reset() consumes the trial's first timestep internally,
    # so exactly one fixation step remains before frame1.
    env = _make_tam()
    env.reset()
    # Fixating (action 0) during the fixation period must NOT be punished.
    out = env.step(TAM_CHOICES["fixation"])
    assert len(out) == 5
    _, reward, terminated, truncated, info = out
    assert reward == 0.0
    assert terminated is False and truncated is False
    # Fresh trial: breaking fixation (any non-fixation action) on the
    # remaining fixation step draws the abort penalty.
    env.reset()
    _, reward, _, _, _ = env.step(TAM_CHOICES["left"])
    assert reward == pytest.approx(env.rewards["abort"])


def test_decision_reward():
    env = _make_tam()
    env.reset()
    gt_final = int(env.gt[-1])
    for _ in range(7):  # t1..t7: remaining fixation step, frames 1-5, first decision step
        env.step(TAM_CHOICES["fixation"])
    _, reward, _, _, info = env.step(gt_final)  # t8: final decision step
    assert reward == pytest.approx(env.rewards["correct"])
    assert info["new_trial"] is True


def test_custom_stimuli_override():
    frames = {
        f"{s}-tam-{m}": [np.zeros((32, 32))]
        for s in SHAPES
        for m in ("no_motion", "left", "right", "middle", "init")
    }
    env = TAMTask(box_shape="square", variant="standard",
                  stim_ori="horizontal", stimuli=frames, img_size=32)
    env.seed(0)
    env.reset()
    assert env.ob.shape == (9, 32, 32)


def test_invalid_args_raise():
    with pytest.raises(ValueError, match="box_shape"):
        TAMTask(box_shape="hexagon")
    with pytest.raises(ValueError, match="stim_ori"):
        TAMTask(stim_ori="diagonal")
    with pytest.raises(ValueError, match="variant"):
        TAMTask(variant="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_envs.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'illusion_rnn.envs'`.

- [ ] **Step 3: Implement `illusion_rnn/envs.py` (TAMTask half)**

```python
"""Neurogym trial environments for TAM and real-motion tasks.

Label maps are frozen: the shipped checkpoints' output heads were trained
against these exact indices.
"""

import numpy as np

import neurogym as ngym
from neurogym.core import TrialEnv

from illusion_rnn.stimuli import (
    MOTION_TYPES,
    SHAPES,
    TAM_VARIANTS,
    load_motion,
    load_tam,
)

TAM_CHOICES = {"fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5}
MOTION_CHOICES = {
    "no_motion": 0, "left": 1, "middle": 2, "right": 3,
    "down": 4, "up": 5, "fixation": 6,
}

# direction index -> stimulus-file motion id (vertical stimuli are the
# horizontal files rotated 90° clockwise, so up/down reuse left/right files)
_MOTION_ID = {0: "no_motion", 1: "left", 2: "middle", 3: "right", 4: "right", 5: "left"}


def _rotate(obj):
    if isinstance(obj, np.ndarray):
        return np.rot90(obj, k=-1)
    return [_rotate(item) for item in obj]


def rotate_stimuli(stimuli: dict) -> dict:
    """Return a copy of a stimulus dict with every frame rotated 90° clockwise."""
    return {key: _rotate(value) for key, value in stimuli.items()}


def _validate(name, value, allowed):
    if value not in allowed:
        msg = f"Unknown {name} {value!r}; expected one of {tuple(allowed)}"
        raise ValueError(msg)


class TAMTask(TrialEnv):
    """3AFC Transformational Apparent Motion task.

    Trial structure: fixation (100 ms) -> init frame (50 ms) -> a repeated
    second frame (4 x 50 ms) -> decision (100 ms), dt = 50 ms. Ground truth
    is ``fixation`` (0) through frame1 and the motion direction from frame2
    onward. Horizontal trials sample directions {left, middle, right};
    vertical trials {middle, down, up} on stimuli rotated 90° clockwise
    once at construction.

    Parameters
    ----------
    dt : int
        Time step in ms.
    box_shape : str
        One of ``illusion_rnn.stimuli.SHAPES``.
    variant : str
        Stimulus set: ``standard`` | ``basic`` | ``outline``.
    stim_ori : str
        ``horizontal`` | ``vertical``.
    stimuli : dict, optional
        Pre-loaded stimulus dict (see ``stimuli.load_tam``). Loaded from the
        packaged set for ``variant`` when None. Keys must use the variant's
        filename prefix (``tam`` or ``outline``).
    sigma : float
        Std of Gaussian noise added to the frame periods.
    img_size : int
        Height/width of the (square) observation.
    rewards, timing : dict, optional
        Overrides merged into the defaults.
    """

    def __init__(
        self,
        dt: int = 50,
        box_shape: str = "square",
        variant: str = "standard",
        stim_ori: str = "horizontal",
        stimuli: dict | None = None,
        sigma: float = 0.0,
        img_size: int = 64,
        rewards: dict | None = None,
        timing: dict | None = None,
    ):
        super().__init__(dt=dt)
        _validate("box_shape", box_shape, SHAPES)
        _validate("variant", variant, TAM_VARIANTS)
        _validate("stim_ori", stim_ori, ("horizontal", "vertical"))

        self.box_shape = box_shape
        self.variant = variant
        self.stim_ori = stim_ori
        self.sigma = sigma
        self.img_size = img_size
        self._prefix = "outline" if variant == "outline" else "tam"

        if stimuli is None:
            stimuli = load_tam(img_size, variant=variant)
        if stim_ori == "vertical":
            stimuli = rotate_stimuli(stimuli)
        self._stimuli = stimuli

        self.abort = False
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 100,
            "frame1": 50, "frame2": 50, "frame3": 50, "frame4": 50, "frame5": 50,
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.ob_shape = (img_size, img_size)
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=self.ob_shape, dtype=np.float32,
        )
        self.choice_names = dict(TAM_CHOICES)
        self.action_space = ngym.spaces.Discrete(6, name=self.choice_names)

    def _sample_direction(self) -> int:
        if self.stim_ori == "horizontal":
            options = [TAM_CHOICES["left"], TAM_CHOICES["middle"], TAM_CHOICES["right"]]
        else:
            options = [TAM_CHOICES["middle"], TAM_CHOICES["down"], TAM_CHOICES["up"]]
        return int(self.rng.choice(options))

    def _fixation_ob(self) -> np.ndarray:
        fix = np.zeros(self.ob_shape)
        center, half = self.img_size // 2, 1
        fix[center - half:center + half, center - half:center + half] = 1.0
        return fix

    def _new_trial(self, **kwargs):
        direction = self._sample_direction()
        trial = {
            "ground_truth": direction,
            "box_shape": self.box_shape,
            "variant": self.variant,
            "stim_ori": self.stim_ori,
            "noise": self.sigma,
        }
        trial.update(kwargs)

        motion_id = _MOTION_ID[trial["ground_truth"]]
        key = f"{self.box_shape}-{self._prefix}-{motion_id}"
        second_frames = self._stimuli[key]
        init_frame = self._stimuli[f"{self.box_shape}-{self._prefix}-init"][0]
        second = second_frames[int(self.rng.randint(len(second_frames)))]
        frames = [init_frame] + [second] * 4

        self.add_period(
            ["fixation", "frame1", "frame2", "frame3", "frame4", "frame5", "decision"],
        )
        self.add_ob(self._fixation_ob(), period=["fixation"])
        for i, frame in enumerate(frames):
            self.add_ob(frame, period=[f"frame{i + 1}"])
        self.add_ob(np.zeros(self.ob_shape), period=["decision"])
        self.add_randn(
            0, self.sigma,
            period=["frame1", "frame2", "frame3", "frame4", "frame5"],
        )

        self.set_groundtruth(self.choice_names["fixation"], period=["fixation", "frame1"])
        self.set_groundtruth(
            trial["ground_truth"],
            period=["frame2", "frame3", "frame4", "frame5", "decision"],
        )
        return trial

    def _step(self, action):
        new_trial = False
        reward = 0.0
        gt = self.gt_now
        fixation_action = self.choice_names["fixation"]

        if self.in_period("fixation"):
            if action != fixation_action:
                new_trial = self.abort
                reward += self.rewards["abort"]
        elif self.in_period("decision") and action != fixation_action:
            new_trial = True
            if action == gt:
                reward += self.rewards["correct"]
                self.performance = 1
            else:
                reward += self.rewards["fail"]

        return self.ob_now, reward, False, False, {"new_trial": new_trial, "gt": gt}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_envs.py -v`
Expected: all PASS. Known trap: if `test_seeded_trials_reproduce` fails, some sampling still uses `np.random`/`random` instead of `self.rng`. If `test_tam_grid` fails on step count, check that `add_period` received the period *names* list (durations resolve from `self.timing`).

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/envs.py tests/test_envs.py
git commit -m "Port TAMTask to neurogym 2.x (seeded rng, 5-tuple step, rotation fix)" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 5: MotionTask environment

**Files:**
- Modify: `illusion_rnn/envs.py` (append MotionTask), `tests/test_envs.py` (append tests)
- Reference (read-only): `src/environments.py:401-633` (old MotionTask being ported)
- Test: `tests/test_envs.py`

**Interfaces:**
- Consumes: `load_motion`, `MOTION_TYPES`, `SHAPES` from Task 2; `MOTION_CHOICES`, `rotate_stimuli`, `_validate`, `_MOTION_ID` from Task 4 (same module).
- Produces: `MotionTask(dt=50, box_shape="square", motion_type="continuous", stim_ori="horizontal", stimuli=None, sigma=0.0, img_size=64, rewards=None, timing=None)` — a `TrialEnv` with `Discrete(7)` action space, `choice_names` = copy of `MOTION_CHOICES`. Default trial: 10 steps at dt=50 (fixation 100, frames 1–5 at 50, decision 150). gt is 6 (fixation) everywhere except the trial direction during decision. Horizontal directions {0,1,2,3}; vertical {0,2,4,5}. `no_motion` trials repeat one randomly chosen frame 5 times.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_envs.py`:

```python
from illusion_rnn.envs import MOTION_CHOICES, MotionTask  # top of file
from illusion_rnn.stimuli import MOTION_TYPES  # top of file


def _make_motion(**kwargs):
    defaults = dict(box_shape="square", motion_type="continuous",
                    stim_ori="horizontal", img_size=32)
    defaults.update(kwargs)
    env = MotionTask(**defaults)
    env.seed(0)
    return env


@pytest.mark.parametrize("motion_type", MOTION_TYPES)
@pytest.mark.parametrize("stim_ori", ["horizontal", "vertical"])
@pytest.mark.parametrize("shape", SHAPES)
def test_motion_grid(motion_type, stim_ori, shape):
    env = _make_motion(motion_type=motion_type, stim_ori=stim_ori, box_shape=shape)
    env.reset()
    # fixation 100 + 5 frames x 50 + decision 150 = 500 ms; dt 50 -> 10 steps
    assert env.ob.shape == (10, 32, 32)
    assert env.gt.shape == (10,)
    # gt is fixation (6) everywhere except the 3 decision steps
    assert set(env.gt[:7]) == {MOTION_CHOICES["fixation"]}
    allowed = {0, 1, 2, 3} if stim_ori == "horizontal" else {0, 2, 4, 5}
    assert set(env.gt[7:]) <= allowed
    assert len(set(env.gt[7:])) == 1


def test_motion_action_space_frozen():
    env = _make_motion()
    assert env.action_space.n == 7
    assert env.choice_names == {
        "no_motion": 0, "left": 1, "middle": 2, "right": 3,
        "down": 4, "up": 5, "fixation": 6,
    }


def test_motion_direction_variability():
    env = _make_motion()
    env.reset()
    gts = {env.new_trial()["ground_truth"] for _ in range(80)}
    assert gts == {0, 1, 2, 3}


def test_no_motion_trials_are_static():
    env = _make_motion()
    env.reset()
    for _ in range(80):
        trial = env.new_trial()
        if trial["ground_truth"] == MOTION_CHOICES["no_motion"]:
            # frames 1-5 are steps 2..6 of the ob; all identical
            frames = env.ob[2:7]
            for i in range(1, 5):
                np.testing.assert_array_equal(frames[i], frames[0])
            break
    else:
        pytest.fail("no no_motion trial sampled in 80 draws")


def test_motion_trials_change_frames():
    env = _make_motion()
    env.reset()
    for _ in range(80):
        trial = env.new_trial()
        if trial["ground_truth"] != MOTION_CHOICES["no_motion"]:
            frames = env.ob[2:7]
            assert any(
                not np.array_equal(frames[i], frames[0]) for i in range(1, 5)
            )
            break
    else:
        pytest.fail("no motion trial sampled in 80 draws")


def test_motion_step_fixation_index():
    # neurogym 2.x reset() consumes the trial's first timestep internally,
    # so exactly one fixation step remains before frame1.
    env = _make_motion()
    env.reset()
    out = env.step(MOTION_CHOICES["fixation"])
    assert len(out) == 5
    assert out[1] == 0.0  # fixating during fixation: no penalty
    env.reset()
    _, reward, _, _, _ = env.step(MOTION_CHOICES["left"])
    assert reward == pytest.approx(env.rewards["abort"])


def test_motion_vertical_stimuli_do_not_mutate():
    env = _make_motion(stim_ori="vertical")
    env.reset()
    before = env._stimuli["square-cnt-left"][0][0].copy()
    for _ in range(5):
        env.new_trial()
    np.testing.assert_array_equal(env._stimuli["square-cnt-left"][0][0], before)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_envs.py -v`
Expected: previous TAM tests PASS; new tests ERROR with `ImportError: cannot import name 'MotionTask'`.

- [ ] **Step 3: Append MotionTask to `illusion_rnn/envs.py`**

```python
class MotionTask(TrialEnv):
    """Real-motion 4AFC task (control condition for TAM).

    Trial structure: fixation (100 ms) -> 5 motion frames (50 ms each) ->
    decision (150 ms), dt = 50 ms. Ground truth is ``fixation`` (6) outside
    the decision period and the motion direction during it. ``no_motion``
    trials repeat a single randomly drawn frame. Horizontal trials sample
    {no_motion, left, middle, right}; vertical {no_motion, middle, down, up}
    on stimuli rotated 90° clockwise once at construction.

    Parameters are as in ``TAMTask`` except ``motion_type``
    (``continuous`` | ``tracking``) replacing ``variant``.
    """

    def __init__(
        self,
        dt: int = 50,
        box_shape: str = "square",
        motion_type: str = "continuous",
        stim_ori: str = "horizontal",
        stimuli: dict | None = None,
        sigma: float = 0.0,
        img_size: int = 64,
        rewards: dict | None = None,
        timing: dict | None = None,
    ):
        super().__init__(dt=dt)
        _validate("box_shape", box_shape, SHAPES)
        _validate("motion_type", motion_type, MOTION_TYPES)
        _validate("stim_ori", stim_ori, ("horizontal", "vertical"))

        self.box_shape = box_shape
        self.motion_type = motion_type
        self.stim_ori = stim_ori
        self.sigma = sigma
        self.img_size = img_size
        self._prefix = {"continuous": "cnt", "tracking": "track"}[motion_type]

        if stimuli is None:
            stimuli = load_motion(img_size, motion_type=motion_type)
        if stim_ori == "vertical":
            stimuli = rotate_stimuli(stimuli)
        self._stimuli = stimuli

        self.abort = False
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 100,
            "frame1": 50, "frame2": 50, "frame3": 50, "frame4": 50, "frame5": 50,
            "decision": 150,
        }
        if timing:
            self.timing.update(timing)

        self.ob_shape = (img_size, img_size)
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=self.ob_shape, dtype=np.float32,
        )
        self.choice_names = dict(MOTION_CHOICES)
        self.action_space = ngym.spaces.Discrete(7, name=self.choice_names)

    def _sample_direction(self) -> int:
        if self.stim_ori == "horizontal":
            options = [0, 1, 2, 3]  # no_motion, left, middle, right
        else:
            options = [0, 2, 4, 5]  # no_motion, middle, down, up
        return int(self.rng.choice(options))

    def _fixation_ob(self) -> np.ndarray:
        fix = np.zeros(self.ob_shape)
        center, half = self.img_size // 2, 1
        fix[center - half:center + half, center - half:center + half] = 1.0
        return fix

    def _sample_frames(self, direction: int) -> list:
        if direction == MOTION_CHOICES["no_motion"]:
            file_dir = ("left", "right")[int(self.rng.randint(2))]
            exemplars = self._stimuli[f"{self.box_shape}-{self._prefix}-{file_dir}"]
            exemplar = exemplars[int(self.rng.randint(len(exemplars)))]
            frame = exemplar[int(self.rng.randint(len(exemplar)))]
            return [frame] * 5
        motion_id = _MOTION_ID[direction]
        exemplars = self._stimuli[f"{self.box_shape}-{self._prefix}-{motion_id}"]
        return exemplars[int(self.rng.randint(len(exemplars)))]

    def _new_trial(self, **kwargs):
        direction = self._sample_direction()
        trial = {
            "ground_truth": direction,
            "box_shape": self.box_shape,
            "motion_type": self.motion_type,
            "stim_ori": self.stim_ori,
            "noise": self.sigma,
        }
        trial.update(kwargs)
        frames = self._sample_frames(trial["ground_truth"])

        self.add_period(
            ["fixation", "frame1", "frame2", "frame3", "frame4", "frame5", "decision"],
        )
        self.add_ob(self._fixation_ob(), period=["fixation"])
        for i, frame in enumerate(frames):
            self.add_ob(frame, period=[f"frame{i + 1}"])
        self.add_ob(np.zeros(self.ob_shape), period=["decision"])
        self.add_randn(
            0, self.sigma,
            period=["frame1", "frame2", "frame3", "frame4", "frame5"],
        )

        self.set_groundtruth(
            self.choice_names["fixation"],
            period=["fixation", "frame1", "frame2", "frame3", "frame4", "frame5"],
        )
        self.set_groundtruth(trial["ground_truth"], period=["decision"])
        return trial

    def _step(self, action):
        new_trial = False
        reward = 0.0
        gt = self.gt_now
        fixation_action = self.choice_names["fixation"]

        if self.in_period("fixation"):
            if action != fixation_action:
                new_trial = self.abort
                reward += self.rewards["abort"]
        elif self.in_period("decision") and action != fixation_action:
            new_trial = True
            if action == gt:
                reward += self.rewards["correct"]
                self.performance = 1
            else:
                reward += self.rewards["fail"]

        return self.ob_now, reward, False, False, {"new_trial": new_trial, "gt": gt}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_envs.py -v`
Expected: all PASS (TAM + Motion).

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/envs.py tests/test_envs.py
git commit -m "Port MotionTask to neurogym 2.x" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 6: Training and evaluation functions

**Files:**
- Create: `illusion_rnn/training.py`, `tests/test_training.py`
- Reference (read-only): `src/networks.py:399-631` (old train_rnn/test_rnn being replaced)
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `TAMTask`, `MotionTask` (Tasks 4–5); `ShapesCNN` (Task 3); `ngym.Dataset`.
- Produces:
  - `make_env(task: str, **kwargs) -> TrialEnv` — `task ∈ {"tam", "motion"}` dispatching to `TAMTask` / `MotionTask`.
  - `make_dataset(env, batch_size: int = 16, seq_len: int = 100) -> ngym.Dataset`.
  - `train(model, datasets, n_epochs: int = 1000, lr: float = 5e-4, device=None, encoder=None, log_every: int = 100) -> dict` — returns `{"loss": [...], "accuracy": [...]}` (one entry per epoch). `datasets` is one `ngym.Dataset` or a list (batches concatenated along batch axis). `encoder` is `Callable[[np.ndarray], np.ndarray]` mapping `(T, B, H, W) -> (T, B, F)`; when None, images are flattened to `(T, B, H*W)`. Loss is `CrossEntropyLoss` over all timesteps.
  - `evaluate(model, env, n_trials: int = 100, device=None, encoder=None) -> EvalResult` — dataclass with `accuracy: float`, `trials: list[dict]` (keys `ground_truth`, `choice`, `correct`), `activity: list[np.ndarray]` (each `(T, hidden_size)`). Choice = argmax of the final timestep's output.
  - `cnn_encoder(cnn: ShapesCNN, device=None) -> Callable` — batched feature extractor implementing the encoder contract.
  - `resolve_device(device=None) -> torch.device` — explicit arg > cuda > mps > cpu.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training.py`:

```python
import numpy as np
import torch

from illusion_rnn.models import RNNNet, ShapesCNN
from illusion_rnn.training import (
    EvalResult,
    cnn_encoder,
    evaluate,
    make_dataset,
    make_env,
    resolve_device,
    train,
)


def _tiny_env():
    env = make_env(
        "tam", box_shape="square", variant="standard",
        stim_ori="horizontal", img_size=16,
    )
    env.seed(0)
    return env


def test_make_env_dispatch():
    assert type(_tiny_env()).__name__ == "TAMTask"
    motion = make_env("motion", box_shape="circle", motion_type="tracking",
                      stim_ori="horizontal", img_size=16)
    assert type(motion).__name__ == "MotionTask"
    try:
        make_env("bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_make_dataset_shapes():
    ds = make_dataset(_tiny_env(), batch_size=4, seq_len=18)
    inputs, labels = ds()
    assert inputs.shape == (18, 4, 16, 16)
    assert labels.shape == (18, 4)


def test_train_decreases_loss():
    torch.manual_seed(0)
    ds = make_dataset(_tiny_env(), batch_size=4, seq_len=18)
    model = RNNNet(input_size=16 * 16, hidden_size=32, output_size=6, dt=50)
    history = train(model, ds, n_epochs=40, lr=1e-3, device="cpu", log_every=0)
    assert len(history["loss"]) == 40
    assert len(history["accuracy"]) == 40
    assert np.mean(history["loss"][-5:]) < np.mean(history["loss"][:5])


def test_train_multi_dataset():
    torch.manual_seed(0)
    datasets = [
        make_dataset(_tiny_env(), batch_size=2, seq_len=9),
        make_dataset(_tiny_env(), batch_size=3, seq_len=9),
    ]
    model = RNNNet(16 * 16, 16, 6, dt=50)
    history = train(model, datasets, n_epochs=2, lr=1e-3, device="cpu", log_every=0)
    assert len(history["loss"]) == 2


def test_evaluate_structure():
    model = RNNNet(16 * 16, 32, 6, dt=50)
    result = evaluate(model, _tiny_env(), n_trials=5, device="cpu")
    assert isinstance(result, EvalResult)
    assert len(result.trials) == 5
    assert len(result.activity) == 5
    assert result.activity[0].shape == (9, 32)
    assert set(result.trials[0]) == {"ground_truth", "choice", "correct"}
    assert 0.0 <= result.accuracy <= 1.0


def test_encoder_hook_in_train_and_evaluate():
    def mean_encoder(inputs):  # (T, B, H, W) -> (T, B, 1)
        return inputs.mean(axis=(2, 3))[..., np.newaxis]

    torch.manual_seed(0)
    ds = make_dataset(_tiny_env(), batch_size=2, seq_len=9)
    model = RNNNet(input_size=1, hidden_size=8, output_size=6, dt=50)
    history = train(model, ds, n_epochs=2, lr=1e-3, device="cpu",
                    encoder=mean_encoder, log_every=0)
    assert len(history["loss"]) == 2
    result = evaluate(model, _tiny_env(), n_trials=3, device="cpu",
                      encoder=mean_encoder)
    assert result.activity[0].shape == (9, 8)


def test_cnn_encoder_shapes():
    # NOTE: 32 is the smallest input ShapesCNN's conv stack supports.
    cnn = ShapesCNN(input_size=32, feature_dim=12)
    encoder = cnn_encoder(cnn, device="cpu")
    out = encoder(np.random.rand(3, 2, 32, 32).astype(np.float32))
    assert out.shape == (3, 2, 12)


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'illusion_rnn.training'`.

- [ ] **Step 3: Implement `illusion_rnn/training.py`**

```python
"""Supervised training and evaluation for TAM/motion tasks.

The pipeline is the neurogym supervised one: ``ngym.Dataset`` yields
``(inputs (T, B, H, W), labels (T, B))`` batches of concatenated trials;
the model is trained with cross-entropy over every timestep. An optional
``encoder`` callable maps raw image batches ``(T, B, H, W)`` to feature
batches ``(T, B, F)`` before they reach the model — pass
``cnn_encoder(ShapesCNN(...), ...)`` to reproduce the CNN-features
pipeline the ``rnn-cnnfeat64_*`` checkpoints were trained with.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

import neurogym as ngym

from illusion_rnn.envs import MotionTask, TAMTask


def resolve_device(device=None) -> torch.device:
    """Explicit argument > cuda > mps > cpu."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(task: str, **kwargs):
    """Construct a task env: ``task`` is ``"tam"`` (TAMTask) or ``"motion"``
    (MotionTask); ``kwargs`` pass through to the constructor."""
    if task == "tam":
        return TAMTask(**kwargs)
    if task == "motion":
        return MotionTask(**kwargs)
    msg = f"Unknown task {task!r}; expected 'tam' or 'motion'"
    raise ValueError(msg)


def make_dataset(env, batch_size: int = 16, seq_len: int = 100) -> ngym.Dataset:
    """Wrap an env in a neurogym supervised Dataset (env is deep-copied)."""
    return ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)


def _prepare_inputs(inputs: np.ndarray, encoder) -> np.ndarray:
    if encoder is not None:
        return encoder(inputs)
    return inputs.reshape(*inputs.shape[:2], -1)


def train(
    model,
    datasets,
    n_epochs: int = 1000,
    lr: float = 5e-4,
    device=None,
    encoder=None,
    log_every: int = 100,
) -> dict:
    """Train ``model`` on one or more ``ngym.Dataset``s; returns per-epoch
    ``{"loss": [...], "accuracy": [...]}``."""
    device = resolve_device(device)
    model = model.to(device)
    model.train()
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": []}

    for epoch in range(n_epochs):
        batches = [dataset() for dataset in datasets]
        inputs = np.concatenate([b[0] for b in batches], axis=1)
        labels = np.concatenate([b[1] for b in batches], axis=1).flatten()

        x = torch.from_numpy(_prepare_inputs(inputs, encoder)).float().to(device)
        y = torch.from_numpy(labels).long().to(device)

        optimizer.zero_grad()
        out, _ = model(x)
        out = out.view(-1, out.shape[-1])
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        accuracy = (out.argmax(dim=1) == y).float().mean().item()
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)
        if log_every and (epoch + 1) % log_every == 0:
            print(
                f"epoch {epoch + 1}/{n_epochs}  "
                f"loss {loss.item():.4f}  acc {accuracy:.3f}",
            )
    return history


@dataclass
class EvalResult:
    """Result of ``evaluate``: overall accuracy, per-trial records, and
    per-trial hidden activity ``(T, hidden_size)`` arrays."""

    accuracy: float
    trials: list = field(default_factory=list)
    activity: list = field(default_factory=list)


def evaluate(model, env, n_trials: int = 100, device=None, encoder=None) -> EvalResult:
    """Run ``n_trials`` single trials through ``model``; choice is the argmax
    of the final timestep's output."""
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    env.reset()

    trials, activity = [], []
    with torch.no_grad():
        for _ in range(n_trials):
            env.new_trial()
            ob, gt = env.ob, env.gt
            inputs = _prepare_inputs(ob[:, np.newaxis], encoder)
            x = torch.from_numpy(inputs).float().to(device)
            pred, hidden = model(x)
            choice = int(pred[-1, 0].argmax().item())
            ground_truth = int(gt[-1])
            trials.append(
                {
                    "ground_truth": ground_truth,
                    "choice": choice,
                    "correct": choice == ground_truth,
                },
            )
            activity.append(hidden[:, 0].cpu().numpy())

    accuracy = float(np.mean([t["correct"] for t in trials]))
    return EvalResult(accuracy=accuracy, trials=trials, activity=activity)


def cnn_encoder(cnn, device=None):
    """Batched encoder: runs every frame of a ``(T, B, H, W)`` batch through
    ``cnn`` in one forward pass, returning ``(T, B, feature_dim)``."""
    device = resolve_device(device)
    cnn = cnn.to(device)
    cnn.eval()

    def encode(inputs: np.ndarray) -> np.ndarray:
        n_steps, n_batch, height, width = inputs.shape
        x = torch.from_numpy(
            inputs.reshape(n_steps * n_batch, 1, height, width),
        ).float().to(device)
        with torch.no_grad():
            _, features = cnn(x)
        return features.cpu().numpy().reshape(n_steps, n_batch, -1)

    return encode
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`
Expected: all PASS. `test_train_decreases_loss` takes ~10–30 s on CPU; that is normal.

- [ ] **Step 5: Commit**

```bash
git add illusion_rnn/training.py tests/test_training.py
git commit -m "Add train/evaluate pipeline with encoder hook" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 7: Plotting + package exports

**Files:**
- Create: `illusion_rnn/plotting.py`, `tests/test_plotting.py`
- Modify: `illusion_rnn/__init__.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: envs (Tasks 4–5), `EvalResult`-shaped history dicts (Task 6).
- Produces:
  - `plot_trials(env, n_trials: int = 2, save_path=None) -> matplotlib.figure.Figure` — grid of trials × timesteps showing each observation frame with `t` and `gt` in the title; resets the env itself.
  - `plot_training_curves(history: dict, save_path=None) -> Figure` — two panels (loss, accuracy) from a `train()` history dict.
  - Package root re-exports (the public API): `TAMTask`, `MotionTask`, `TAM_CHOICES`, `MOTION_CHOICES`, `rotate_stimuli`, `load_tam`, `load_motion`, `SHAPES`, `TAM_VARIANTS`, `MOTION_TYPES`, `CTRNN`, `RNNNet`, `ShapesCNN`, `load_rnn`, `make_env`, `make_dataset`, `train`, `evaluate`, `EvalResult`, `cnn_encoder`, `resolve_device`, `plot_trials`, `plot_training_curves`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plotting.py`:

```python
import matplotlib.pyplot as plt

from illusion_rnn.plotting import plot_training_curves, plot_trials
from illusion_rnn.training import make_env


def test_plot_trials_grid():
    env = make_env("tam", box_shape="square", variant="standard",
                   stim_ori="horizontal", img_size=16)
    env.seed(0)
    fig = plot_trials(env, n_trials=2)
    assert len(fig.axes) == 2 * 9  # trials x timesteps
    plt.close(fig)


def test_plot_trials_save(tmp_path):
    env = make_env("motion", box_shape="circle", motion_type="continuous",
                   stim_ori="horizontal", img_size=16)
    env.seed(0)
    out = tmp_path / "trials.png"
    fig = plot_trials(env, n_trials=1, save_path=out)
    assert out.exists()
    plt.close(fig)


def test_plot_training_curves():
    history = {"loss": [1.5, 0.8, 0.4], "accuracy": [0.2, 0.5, 0.9]}
    fig = plot_training_curves(history)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_package_exports():
    import illusion_rnn

    for name in (
        "TAMTask", "MotionTask", "TAM_CHOICES", "MOTION_CHOICES",
        "rotate_stimuli", "load_tam", "load_motion", "SHAPES",
        "TAM_VARIANTS", "MOTION_TYPES", "CTRNN", "RNNNet", "ShapesCNN",
        "load_rnn", "make_env", "make_dataset", "train", "evaluate",
        "EvalResult", "cnn_encoder", "resolve_device", "plot_trials",
        "plot_training_curves",
    ):
        assert hasattr(illusion_rnn, name), name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'illusion_rnn.plotting'`.

- [ ] **Step 3: Implement `illusion_rnn/plotting.py`**

```python
"""Minimal matplotlib visualization helpers (no seaborn dependency)."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trials(env, n_trials: int = 2, save_path=None):
    """Plot ``n_trials`` sampled trials as a trials x timesteps image grid."""
    env.reset()
    observations, ground_truths = [], []
    for _ in range(n_trials):
        env.new_trial()
        observations.append(env.ob.copy())
        ground_truths.append(env.gt.copy())

    n_steps = observations[0].shape[0]
    fig, axes = plt.subplots(
        n_trials, n_steps,
        squeeze=False,
        figsize=(1.4 * n_steps, 1.7 * n_trials),
    )
    for row in range(n_trials):
        for col in range(n_steps):
            ax = axes[row][col]
            ax.imshow(observations[row][col], cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"t={col * env.dt}ms\ngt={int(ground_truths[row][col])}",
                fontsize=7,
            )
            ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history: dict, save_path=None):
    """Plot loss and accuracy from a ``train()`` history dict."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, history["accuracy"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 4: Wire up `illusion_rnn/__init__.py`**

Replace its contents with:

```python
"""illusion-rnn: a neurogym testbed for Transformational Apparent Motion (TAM)."""

from illusion_rnn.envs import (
    MOTION_CHOICES,
    TAM_CHOICES,
    MotionTask,
    TAMTask,
    rotate_stimuli,
)
from illusion_rnn.models import CTRNN, RNNNet, ShapesCNN, load_rnn
from illusion_rnn.plotting import plot_training_curves, plot_trials
from illusion_rnn.stimuli import (
    MOTION_TYPES,
    SHAPES,
    TAM_VARIANTS,
    load_motion,
    load_tam,
)
from illusion_rnn.training import (
    EvalResult,
    cnn_encoder,
    evaluate,
    make_dataset,
    make_env,
    resolve_device,
    train,
)

__version__ = "1.0.0"

__all__ = [
    "CTRNN",
    "MOTION_CHOICES",
    "MOTION_TYPES",
    "EvalResult",
    "MotionTask",
    "RNNNet",
    "SHAPES",
    "ShapesCNN",
    "TAM_CHOICES",
    "TAM_VARIANTS",
    "TAMTask",
    "cnn_encoder",
    "evaluate",
    "load_motion",
    "load_rnn",
    "load_tam",
    "make_dataset",
    "make_env",
    "plot_training_curves",
    "plot_trials",
    "resolve_device",
    "rotate_stimuli",
    "train",
    "__version__",
]
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (stimuli, envs, models, training, plotting).

- [ ] **Step 6: Commit**

```bash
git add illusion_rnn/plotting.py illusion_rnn/__init__.py tests/test_plotting.py
git commit -m "Add plotting helpers and public package API" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 8: Checkpoint renames, manifest, and the flagship gate

**Files:**
- Move (git mv): `data/models/*` → `checkpoints/` with renames (table below)
- Create: `checkpoints/MANIFEST.md`, `tests/test_checkpoints.py`
- Test: `tests/test_checkpoints.py`

**Interfaces:**
- Consumes: `load_rnn`, `ShapesCNN` (Task 3); `make_env`, `evaluate` (Task 6).
- Produces: canonical checkpoint paths used by notebooks/README:
  - `checkpoints/rnn-pixel_h2048_tam-horiz.pt` (flagship: 4096→2048→6)
  - `checkpoints/rnn-cnnfeat64_h1024_tam-horiz.pt` (64→1024→6)
  - `checkpoints/rnn-cnnfeat64_h2048_tam-horiz.pt` (64→2048→6)
  - `checkpoints/cnn-shapes_feat64_100px.pt` (ShapesCNN defaults)
  - `checkpoints/legacy/rnn_v13`, `checkpoints/legacy/rnn_v13_state_dict`

- [ ] **Step 1: Move and rename the checkpoints**

Precondition: Task 1 materialized LFS content (`du -h data/models/*.pt` shows MB sizes, not 4.0K). If not, run `git lfs install --local && git lfs fetch origin main && git lfs checkout` first.

```bash
mkdir -p checkpoints/legacy
git mv data/models/learned_horiz_tam_all.pt checkpoints/rnn-pixel_h2048_tam-horiz.pt
git mv data/models/CNN_horiz_tam.pt checkpoints/rnn-cnnfeat64_h1024_tam-horiz.pt
git mv data/models/CNN_horiz_tam_2048.pt checkpoints/rnn-cnnfeat64_h2048_tam-horiz.pt
git mv data/models/cnn_model_64.pt checkpoints/cnn-shapes_feat64_100px.pt
git mv data/models/rnn_v13 checkpoints/legacy/rnn_v13
git mv data/models/rnn_v13_state_dict checkpoints/legacy/rnn_v13_state_dict
ls -la checkpoints checkpoints/legacy
```

Expected: `data/models/` is gone; the four `.pt` files show MB sizes.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_checkpoints.py`:

```python
from pathlib import Path

import pytest
import torch

from illusion_rnn.models import ShapesCNN, load_rnn
from illusion_rnn.training import evaluate, make_env

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

RNN_SPECS = [
    ("rnn-pixel_h2048_tam-horiz.pt", 4096, 2048, 6),
    ("rnn-cnnfeat64_h1024_tam-horiz.pt", 64, 1024, 6),
    ("rnn-cnnfeat64_h2048_tam-horiz.pt", 64, 2048, 6),
]


def _materialized(name: str) -> bool:
    """False for missing files and un-smudged LFS pointers (~130 bytes)."""
    path = CHECKPOINT_DIR / name
    return path.exists() and path.stat().st_size > 1024


def _require(name: str) -> Path:
    if not _materialized(name):
        pytest.skip(f"{name} not materialized (run: git lfs install --local && git lfs checkout)")
    return CHECKPOINT_DIR / name


@pytest.mark.parametrize("name,input_size,hidden_size,output_size", RNN_SPECS)
def test_rnn_checkpoints_load(name, input_size, hidden_size, output_size):
    model = load_rnn(_require(name))
    assert model.rnn.input2h.in_features == input_size
    assert model.rnn.h2h.in_features == hidden_size
    assert model.fc.out_features == output_size


def test_shapes_cnn_checkpoint_loads():
    path = _require("cnn-shapes_feat64_100px.pt")
    model = ShapesCNN()  # defaults must match the checkpoint
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)  # strict — raises on any mismatch


def test_flagship_above_chance():
    model = load_rnn(_require("rnn-pixel_h2048_tam-horiz.pt"))
    env = make_env("tam", box_shape="square", variant="standard",
                   stim_ori="horizontal", img_size=64)
    env.seed(7)
    result = evaluate(model, env, n_trials=30, device="cpu")
    assert result.accuracy > 0.5  # chance is ~1/3
```

- [ ] **Step 3: Run tests to verify current state**

Run: `uv run pytest tests/test_checkpoints.py -v`
Expected: load tests PASS immediately (files exist from Step 1). `test_flagship_above_chance` is the real check — it is expected to PASS if the port preserved the stimulus pipeline. If it FAILS, do not weaken the test; continue to Step 4's gate and the contingency in Step 5.

- [ ] **Step 4: Run the flagship acceptance gate (spec §9)**

```bash
uv run python - <<'EOF'
import numpy as np
from illusion_rnn.models import load_rnn
from illusion_rnn.training import evaluate, make_env

model = load_rnn("checkpoints/rnn-pixel_h2048_tam-horiz.pt")
rows = {}
for variant in ("standard", "outline", "basic"):
    accs = []
    for shape in ("square", "circle", "triangle"):
        env = make_env("tam", box_shape=shape, variant=variant,
                       stim_ori="horizontal", img_size=64)
        env.seed(0)
        accs.append(evaluate(model, env, n_trials=100, device="cpu").accuracy)
    rows[variant] = dict(zip(("square", "circle", "triangle"), np.round(accs, 3)))
    print(variant, rows[variant], "mean", round(float(np.mean(accs)), 3))
gate = float(np.mean(list(rows["standard"].values())))
print("GATE (standard mean, 300 trials):", round(gate, 3), "PASS" if gate > 0.8 else "FAIL")
EOF
```

Record every printed number — they go into MANIFEST.md in Step 6.

- [ ] **Step 5: Contingency — ONLY if the gate printed FAIL**

Retrain the flagship with the known-good recipe (~10–30 min on MPS):

```bash
uv run python - <<'EOF'
import torch
from illusion_rnn.models import RNNNet
from illusion_rnn.training import make_dataset, make_env, train

torch.manual_seed(0)
datasets = [
    make_dataset(
        make_env("tam", box_shape=shape, variant="standard",
                 stim_ori="horizontal", img_size=64),
        batch_size=16, seq_len=100,
    )
    for shape in ("square", "circle", "triangle")
]
model = RNNNet(input_size=64 * 64, hidden_size=2048, output_size=6, dt=50)
history = train(model, datasets, n_epochs=1000, lr=5e-4, device="mps", log_every=50)
torch.save(model.state_dict(), "checkpoints/rnn-pixel_h2048_tam-horiz.pt")
print("final loss", history["loss"][-1], "final acc", history["accuracy"][-1])
EOF
```

Then re-run Step 4; the gate must PASS. Note the retrain in the manifest's provenance column (Step 6). If MPS errors, use `device="cpu"` (slower but correct).

- [ ] **Step 6: Write `checkpoints/MANIFEST.md`**

Fill the `Eval` column with the Step 4 numbers (per-variant means). The provenance line for the flagship depends on whether Step 5 ran.

```markdown
# Checkpoint manifest

All RNN checkpoints are `illusion_rnn.models.RNNNet` state dicts; load them
with `illusion_rnn.models.load_rnn(path)`. Label map (frozen):
fixation=0, left=1, middle=2, right=3, down=4, up=5.

| File | Model | Input | Trained on | Eval (horizontal, mean of 3 shapes x 100 trials, seed 0) |
|---|---|---|---|---|
| `rnn-pixel_h2048_tam-horiz.pt` | RNNNet 4096->2048->6, dt=50 | raw 64x64 frames, flattened | horizontal standard TAM, all shapes (2023) | standard <fill>, outline <fill>, basic <fill> |
| `rnn-cnnfeat64_h1024_tam-horiz.pt` | RNNNet 64->1024->6, dt=50 | 64-d ShapesCNN features of 100x100 frames | horizontal standard TAM, all shapes (2023) | see note below |
| `rnn-cnnfeat64_h2048_tam-horiz.pt` | RNNNet 64->2048->6, dt=50 | 64-d ShapesCNN features of 100x100 frames | horizontal standard TAM, all shapes (2023) | see note below |
| `cnn-shapes_feat64_100px.pt` | ShapesCNN (100px, 64-d, 9 classes) | 100x100 grayscale shape images | 2D geometric shapes dataset (El Korchi & Ghanou, 2020) | n/a (feature extractor) |

The `rnn-cnnfeat64_*` models expect inputs produced by
`cnn_encoder(shapes_cnn, ...)` with the ShapesCNN checkpoint above and
**img_size=100** environments:

    cnn = ShapesCNN()
    cnn.load_state_dict(torch.load("checkpoints/cnn-shapes_feat64_100px.pt", weights_only=True))
    encoder = cnn_encoder(cnn)
    env = make_env("tam", img_size=100, ...)
    evaluate(load_rnn("checkpoints/rnn-cnnfeat64_h2048_tam-horiz.pt"), env, encoder=encoder)

## Legacy

`legacy/rnn_v13` (full pickled model) and `legacy/rnn_v13_state_dict`
(1024->128->4) are an early 32x32 / 4-action prototype kept only as the
companion of the archived notebooks; they do not fit the current envs.

## Provenance

Trained in 2022-2023 with the original `src/` code (see `notebooks/legacy/`).
Verified to load and evaluated under the ported neurogym 2.x environments
on 2026-08-13.
```

Replace each `<fill>` with the measured means from Step 4 (three decimals). Also run this quick eval for the two `rnn-cnnfeat64_*` rows and put the standard-variant mean in their Eval cells (replacing "see note below" with the number; keep the note text):

```bash
uv run python - <<'EOF'
import numpy as np, torch
from illusion_rnn.models import ShapesCNN, load_rnn
from illusion_rnn.training import cnn_encoder, evaluate, make_env

cnn = ShapesCNN()
cnn.load_state_dict(torch.load("checkpoints/cnn-shapes_feat64_100px.pt",
                               map_location="cpu", weights_only=True))
encoder = cnn_encoder(cnn, device="cpu")
for name in ("rnn-cnnfeat64_h1024_tam-horiz.pt", "rnn-cnnfeat64_h2048_tam-horiz.pt"):
    model = load_rnn(f"checkpoints/{name}")
    accs = []
    for shape in ("square", "circle", "triangle"):
        env = make_env("tam", box_shape=shape, variant="standard",
                       stim_ori="horizontal", img_size=100)
        env.seed(0)
        accs.append(evaluate(model, env, n_trials=100, device="cpu", encoder=encoder).accuracy)
    print(name, "standard mean:", round(float(np.mean(accs)), 3))
EOF
```

- [ ] **Step 7: Run tests and commit**

Run: `uv run pytest tests/test_checkpoints.py -v` — all PASS (no skips locally).

```bash
git add -A checkpoints data tests/test_checkpoints.py
git commit -m "Rename checkpoints, add manifest with ported-env evaluations" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 9: Shapes-CNN training script + shape-dataset move

**Files:**
- Move (git mv): `data/stimuli/shape_dataset/` → `data/shape_dataset/`
- Create: `scripts/train_shape_cnn.py`
- Test: `uv run python scripts/train_shape_cnn.py --help` (smoke; no pytest — optional offline script)

**Interfaces:**
- Consumes: `ShapesCNN` (Task 3); `data/shape_dataset/shape_dataset.zip` (LFS, 90 MB).
- Produces: an optional, attributed script that reproduces `cnn-shapes_feat64_100px.pt`. Nothing downstream imports it.

- [ ] **Step 1: Move the dataset directory**

```bash
git mv data/stimuli/shape_dataset data/shape_dataset
ls data
```

Expected: `data/` contains only `shape_dataset/`; `data/stimuli/` is gone.

- [ ] **Step 2: Write `scripts/train_shape_cnn.py`**

```python
#!/usr/bin/env python3
"""Retrain the ShapesCNN feature extractor (optional).

The packaged checkpoint ``checkpoints/cnn-shapes_feat64_100px.pt`` was
produced by an earlier version of this pipeline in 2022; retraining is only
needed if you want to change the encoder.

Data: the 2D geometric shapes dataset (El Korchi & Ghanou, 2020,
https://doi.org/10.17632/wzr2yv7r53.1) — 9 shape classes x 10k RGB images.
``data/shape_dataset/shape_dataset.zip`` (git LFS) holds the 200x200 originals
under ``output/``. This script is a cleaned-up derivative of a classroom
implementation adapted for this project in 2022.

Usage:
    uv run python scripts/train_shape_cnn.py \
        --zip data/shape_dataset/shape_dataset.zip --epochs 30
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from illusion_rnn.models import ShapesCNN  # noqa: E402

# Frozen label map — the shipped checkpoint was trained with these indices.
CLASS_IDS = {
    "Circle": 0, "Square": 1, "Octagon": 2, "Heptagon": 3, "Nonagon": 4,
    "Star": 5, "Hexagon": 6, "Pentagon": 7, "Triangle": 8,
}


class ShapeImages(Dataset):
    def __init__(self, image_dir: Path, img_size: int = 100):
        self.paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if not self.paths:
            msg = f"No images found in {image_dir}"
            raise FileNotFoundError(msg)
        self.img_size = img_size
        self.labels = []
        for path in self.paths:
            label = next(
                (idx for name, idx in CLASS_IDS.items() if name in path.name), None,
            )
            if label is None:
                msg = f"Cannot infer class from filename {path.name!r}"
                raise ValueError(msg)
            self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("L")
        img = img.resize((self.img_size, self.img_size))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        return x.unsqueeze(0), self.labels[index]


def accuracy(loader, model, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path,
                        default=Path("data/shape_dataset/shape_dataset.zip"))
    parser.add_argument("--workdir", type=Path,
                        default=Path("data/shape_dataset/extracted"))
    parser.add_argument("--out", type=Path,
                        default=Path("checkpoints/cnn-shapes_feat64_100px.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    image_dir = args.workdir / "output"
    if not image_dir.exists():
        if args.zip.stat().st_size < 1024:
            msg = (
                f"{args.zip} is an LFS pointer; run "
                "'git lfs install --local && git lfs checkout' first"
            )
            raise SystemExit(msg)
        print(f"Extracting {args.zip} -> {args.workdir}")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(args.workdir)

    dataset = ShapeImages(image_dir)
    n_test = len(dataset) // 5
    train_set, test_set = random_split(
        dataset, [len(dataset) - n_test, n_test],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = ShapesCNN().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        losses = []
        for x, y in train_loader:
            x, y = x.to(args.device), torch.as_tensor(y).to(args.device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(
            f"epoch {epoch}/{args.epochs}  loss {np.mean(losses):.4f}  "
            f"test acc {accuracy(test_loader, model, args.device):.3f}",
        )

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the CLI (no training run)**

Run: `uv run python scripts/train_shape_cnn.py --help`
Expected: usage text prints, exit 0. Do NOT run a full training — it needs the 90 MB zip extracted and ~30 min; the shipped checkpoint already exists.

- [ ] **Step 4: Guard the extraction dir from git**

Append to `.gitignore`:

```
# extracted shape dataset (from data/shape_dataset/shape_dataset.zip)
data/shape_dataset/extracted/
```

- [ ] **Step 5: Commit**

```bash
git add -A data scripts/train_shape_cnn.py .gitignore
git commit -m "Add attributed shapes-CNN training script; move shape dataset" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 10: Delete legacy code, archive legacy notebooks

**Files:**
- Move (git mv): all 12 `notebooks/*.ipynb` → `notebooks/legacy/`; `notebooks/training_curve` → `figures/training_curve.png`
- Create: `notebooks/legacy/README.md`
- Delete (git rm): `src/`, `scripts/make_datasets.py`, `scripts/train_models.py`, `scripts/CNN_train.py`, `scripts/CNN_test.ipynb`, `scripts/CNN_test_on_tam.ipynb`, `parameters.json`, `TODO.md`, `.idea/`
- Test: full suite must stay green

**Interfaces:**
- Consumes: nothing.
- Produces: a repo tip with no dead code; `notebooks/legacy/` archive. Later tasks create the new notebooks alongside it.

- [ ] **Step 1: Archive the notebooks**

```bash
mkdir -p notebooks/legacy
for nb in notebooks/*.ipynb; do git mv "$nb" notebooks/legacy/; done
git mv notebooks/training_curve figures/training_curve.png
ls notebooks/legacy | wc -l   # expect 12
```

- [ ] **Step 2: Write `notebooks/legacy/README.md`**

```markdown
# Legacy notebooks (2022-2023)

These are the original research notebooks, kept untouched as the
experimental record of the project. They were written against neurogym 0.x
and a `src/` layout that no longer exists, and they will NOT run against
the current `illusion_rnn` package. The companion prototype checkpoints
live in `../../checkpoints/legacy/`.

For working examples, see `../01_quickstart.ipynb` and `../02_train.ipynb`.
```

- [ ] **Step 3: Delete the dead code**

```bash
git rm -r src .idea
git rm scripts/make_datasets.py scripts/train_models.py scripts/CNN_train.py \
       "scripts/CNN_test.ipynb" "scripts/CNN_test_on_tam.ipynb" \
       parameters.json TODO.md
```

- [ ] **Step 4: Verify nothing broke and the tree is what we expect**

```bash
uv run pytest -q
git status --short
ls
```

Expected: all tests pass; top level now shows `checkpoints data docs figures illusion_rnn notebooks scripts tests` plus config files; `scripts/` contains only `train_shape_cnn.py`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Archive legacy notebooks; remove superseded src/, scripts, configs" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 11: Quickstart notebook + README figures

**Files:**
- Create: `notebooks/01_quickstart.ipynb` (committed executed), `figures/sample_trials.png`, `figures/generalization.png` (produced by the notebook run)
- Temp (create, run, delete): `notebooks/_build_quickstart.py`
- Test: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_quickstart.ipynb`

**Interfaces:**
- Consumes: the full public API (`import illusion_rnn as ir`), `checkpoints/rnn-pixel_h2048_tam-horiz.pt`.
- Produces: the two PNGs the README (Task 13) embeds.

- [ ] **Step 1: Create the builder script `notebooks/_build_quickstart.py`**

```python
"""One-shot builder for 01_quickstart.ipynb (run once, then delete)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        "# TAM Testbed — Quickstart\n\n"
        "Build Transformational Apparent Motion (TAM) task environments, "
        "look at trials, load the shipped reference RNN, and test how it "
        "generalizes across TAM stimulus variants.\n\n"
        "Setup: see the README (clone with git LFS, then `uv sync`)."
    ),
    code(
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "import illusion_rnn as ir\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n"
        "CKPT = ROOT / \"checkpoints\"\n"
        "FIG = ROOT / \"figures\"\n"
        "print(\"illusion_rnn\", ir.__version__)"
    ),
    md(
        "## 1. Task environments\n\n"
        "A TAM trial: fixation → an *init* frame (two end shapes) → a second "
        "frame where a bar connects them, with an extra element that "
        "disambiguates the motion direction → a blank decision period. "
        "Ground truth is `fixation` (0) until the second frame appears, then "
        "the direction (left=1, middle=2, right=3)."
    ),
    code(
        "env = ir.make_env(\"tam\", box_shape=\"square\", variant=\"standard\",\n"
        "                  stim_ori=\"horizontal\", img_size=64)\n"
        "env.seed(0)\n"
        "fig = ir.plot_trials(env, n_trials=2, save_path=FIG / \"sample_trials.png\")"
    ),
    code(
        "for variant in (\"outline\", \"basic\"):\n"
        "    env = ir.make_env(\"tam\", box_shape=\"square\", variant=variant,\n"
        "                      stim_ori=\"horizontal\", img_size=64)\n"
        "    env.seed(0)\n"
        "    ir.plot_trials(env, n_trials=1)"
    ),
    md(
        "The motion control task shows *real* frame-by-frame motion "
        "(`continuous` or `tracking`), used to train networks on unambiguous "
        "motion before testing them on TAM."
    ),
    code(
        "env = ir.make_env(\"motion\", box_shape=\"circle\", motion_type=\"continuous\",\n"
        "                  stim_ori=\"horizontal\", img_size=64)\n"
        "env.seed(0)\n"
        "ir.plot_trials(env, n_trials=2)"
    ),
    md("## 2. Load the shipped reference RNN\n\n"
       "A continuous-time RNN (4096 → 2048 → 6) trained on raw 64×64 pixels "
       "of the horizontal *standard* TAM set (all three shapes)."),
    code(
        "flagship = CKPT / \"rnn-pixel_h2048_tam-horiz.pt\"\n"
        "assert flagship.stat().st_size > 1024, (\n"
        "    \"Checkpoint is an LFS pointer; run: git lfs install --local && git lfs checkout\"\n"
        ")\n"
        "model = ir.load_rnn(flagship)\n"
        "sum(p.numel() for p in model.parameters())"
    ),
    md("## 3. Generalization across TAM variants\n\n"
       "Evaluate on the training condition (*standard*) and on the unseen "
       "*outline* and *basic* variants — 100 trials per shape per variant."),
    code(
        "variants = (\"standard\", \"outline\", \"basic\")\n"
        "results = {}\n"
        "for variant in variants:\n"
        "    accs = []\n"
        "    for shape in ir.SHAPES:\n"
        "        env = ir.make_env(\"tam\", box_shape=shape, variant=variant,\n"
        "                          stim_ori=\"horizontal\", img_size=64)\n"
        "        env.seed(0)\n"
        "        accs.append(ir.evaluate(model, env, n_trials=100, device=\"cpu\").accuracy)\n"
        "    results[variant] = accs\n"
        "    print(f\"{variant:9s}\", {s: round(a, 3) for s, a in zip(ir.SHAPES, accs)})"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "x = np.arange(len(variants))\n"
        "width = 0.25\n"
        "for i, shape in enumerate(ir.SHAPES):\n"
        "    ax.bar(x + (i - 1) * width, [results[v][i] for v in variants],\n"
        "           width, label=shape)\n"
        "ax.axhline(1 / 3, color=\"gray\", ls=\"--\", lw=1, label=\"chance\")\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(variants)\n"
        "ax.set_ylabel(\"Accuracy (100 trials/bar)\")\n"
        "ax.set_title(\"Reference RNN: TAM variant generalization (horizontal)\")\n"
        "ax.legend(frameon=False)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / \"generalization.png\", dpi=150, bbox_inches=\"tight\")"
    ),
    md("## 4. A glance at the hidden dynamics"),
    code(
        "env = ir.make_env(\"tam\", box_shape=\"square\", variant=\"standard\",\n"
        "                  stim_ori=\"horizontal\", img_size=64)\n"
        "env.seed(3)\n"
        "res = ir.evaluate(model, env, n_trials=1, device=\"cpu\")\n"
        "act = res.activity[0]  # (T, hidden)\n"
        "order = np.argsort(act.mean(axis=0))[-60:]\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "im = ax.imshow(act[:, order].T, aspect=\"auto\", cmap=\"viridis\")\n"
        "ax.set_xlabel(f\"Time step (dt = {env.dt} ms)\")\n"
        "ax.set_ylabel(\"Hidden unit (top 60 by mean rate)\")\n"
        "fig.colorbar(im, label=\"activity\")\n"
        "fig.tight_layout()"
    ),
    md(
        "## 5. Plug in your own model\n\n"
        "Anything with the interface `model(x: (T, B, F) tensor) -> (out (T, B, 6), "
        "activity (T, B, H))` drops into `ir.evaluate` / `ir.train`. To test a "
        "frame encoder instead of raw pixels, pass "
        "`encoder=ir.cnn_encoder(your_cnn)` — see `checkpoints/MANIFEST.md` for "
        "the shipped CNN-features models, and the README for custom stimuli."
    ),
]

nbf.write(nb, "notebooks/01_quickstart.ipynb")
print("wrote notebooks/01_quickstart.ipynb")
```

- [ ] **Step 2: Build, execute, and clean up**

```bash
uv run python notebooks/_build_quickstart.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/01_quickstart.ipynb
rm notebooks/_build_quickstart.py
ls -la figures/sample_trials.png figures/generalization.png
```

Expected: nbconvert exits 0; both PNGs exist and are non-trivial (>20 KB). The eval cells take a few minutes on CPU.

- [ ] **Step 3: Sanity-check the executed notebook**

```bash
uv run python - <<'EOF'
import json
nb = json.load(open("notebooks/01_quickstart.ipynb"))
outs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
errors = [o for o in outs if o.get("output_type") == "error"]
print("code cells:", sum(c["cell_type"] == "code" for c in nb["cells"]),
      "| outputs:", len(outs), "| errors:", len(errors))
assert not errors, errors
EOF
```

Expected: `errors: 0`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_quickstart.ipynb figures/sample_trials.png figures/generalization.png
git commit -m "Add executed quickstart notebook and README figures" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 12: Training notebook

**Files:**
- Create: `notebooks/02_train.ipynb` (committed executed)
- Temp (create, run, delete): `notebooks/_build_train.py`
- Test: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_train.ipynb`

**Interfaces:**
- Consumes: public API only.
- Produces: nothing downstream; a runnable training demo.

- [ ] **Step 1: Create the builder script `notebooks/_build_train.py`**

```python
"""One-shot builder for 02_train.ipynb (run once, then delete)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        "# Train a TAM RNN from scratch\n\n"
        "A small-scale demo of the supervised pipeline: ~2 minutes on CPU. "
        "The shipped reference model used the same recipe at full scale "
        "(64×64 input, 2048 hidden, 1000 epochs — see "
        "`checkpoints/MANIFEST.md`)."
    ),
    code(
        "from pathlib import Path\n"
        "import tempfile\n"
        "\n"
        "import torch\n"
        "\n"
        "import illusion_rnn as ir\n"
        "\n"
        "torch.manual_seed(0)\n"
        "env = ir.make_env(\"tam\", box_shape=\"square\", variant=\"standard\",\n"
        "                  stim_ori=\"horizontal\", img_size=32)\n"
        "env.seed(0)\n"
        "dataset = ir.make_dataset(env, batch_size=8, seq_len=27)\n"
        "model = ir.RNNNet(input_size=32 * 32, hidden_size=128, output_size=6, dt=50)\n"
        "sum(p.numel() for p in model.parameters())"
    ),
    md("`seq_len=27` packs three 9-step trials per sequence; the loss is "
       "cross-entropy over every timestep (fixation steps included)."),
    code(
        "history = ir.train(model, dataset, n_epochs=300, lr=1e-3,\n"
        "                   device=\"cpu\", log_every=50)"
    ),
    code("ir.plot_training_curves(history)"),
    md("## Save, reload, evaluate\n\nState dicts round-trip through "
       "`ir.load_rnn`, which infers the architecture from the weights."),
    code(
        "path = Path(tempfile.mkdtemp()) / \"demo_rnn.pt\"\n"
        "torch.save(model.state_dict(), path)\n"
        "reloaded = ir.load_rnn(path)\n"
        "\n"
        "test_env = ir.make_env(\"tam\", box_shape=\"square\", variant=\"standard\",\n"
        "                       stim_ori=\"horizontal\", img_size=32)\n"
        "test_env.seed(99)\n"
        "result = ir.evaluate(reloaded, test_env, n_trials=200, device=\"cpu\")\n"
        "print(f\"accuracy on the trained condition: {result.accuracy:.3f}\")"
    ),
    md("## Does it transfer to shapes it never saw?"),
    code(
        "for shape in (\"circle\", \"triangle\"):\n"
        "    e = ir.make_env(\"tam\", box_shape=shape, variant=\"standard\",\n"
        "                    stim_ori=\"horizontal\", img_size=32)\n"
        "    e.seed(99)\n"
        "    acc = ir.evaluate(reloaded, e, n_trials=100, device=\"cpu\").accuracy\n"
        "    print(f\"{shape:9s} {acc:.3f}\")"
    ),
    md(
        "This is deliberately tiny. Scale `hidden_size`, `img_size`, "
        "`n_epochs`, and the number of datasets (pass a list to `ir.train`) "
        "for real experiments; add `sigma=` to the env for noisy frames, or "
        "`encoder=` to train on CNN features instead of pixels."
    ),
]

nbf.write(nb, "notebooks/02_train.ipynb")
print("wrote notebooks/02_train.ipynb")
```

- [ ] **Step 2: Build, execute, sanity-check, clean up**

```bash
uv run python notebooks/_build_train.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebooks/02_train.ipynb
rm notebooks/_build_train.py
uv run python - <<'EOF'
import json
nb = json.load(open("notebooks/02_train.ipynb"))
errors = [o for c in nb["cells"] if c["cell_type"] == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert not errors, errors
print("ok, no cell errors")
EOF
```

Expected: nbconvert exits 0 (the 300-epoch cell takes ~1–3 min on CPU); no cell errors. The printed trained-condition accuracy should be well above 0.5 — if it is near chance (~0.33), something is wrong with the pipeline; stop and investigate rather than committing.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_train.ipynb
git commit -m "Add executed from-scratch training notebook" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 13: README

**Files:**
- Modify: `README.md` (replace the one-liner)
- Test: none (prose); verify relative links/paths exist

**Interfaces:**
- Consumes: `figures/sample_trials.png`, `figures/generalization.png` (Task 11); checkpoint names (Task 8).
- Produces: the repo's front page.

- [ ] **Step 1: Replace `README.md` with:**

````markdown
# illusion-rnn

A [neurogym](https://github.com/neurogym/neurogym)-based testbed for studying
**Transformational Apparent Motion (TAM)** in neural networks: task
environments, matched real-motion controls, a hand-made stimulus set, and
reference recurrent models — packaged so you can plug in your own model.

![Sample TAM trials](figures/sample_trials.png)

## What is TAM?

In transformational apparent motion, two *static* frames — two shapes, then
the same shapes joined by a bar — are perceived as a single object smoothly
transforming and moving in a definite direction. The percept requires solving
a shape-correspondence problem: which contours of frame 2 "came from" frame 1.
That makes TAM a compact probe of how visual systems infer motion from form.

This testbed frames TAM as a supervised trial task: networks are trained to
report motion direction (left / middle / right, or up / down in the vertical
variant) either on TAM stimuli directly or on unambiguous frame-by-frame
motion controls, and are then tested on stimulus variants they never saw
(outline-only shapes, simplified layouts). The question: does motion inferred
from form transfer?

## Install

Requires Python ≥ 3.10. Checkpoints and the shapes dataset use git LFS.

```bash
git lfs install
git clone git@github.com:sh4r11f/illusion-rnn.git
cd illusion-rnn
uv sync            # or: pip install -e .
```

Library-only install (envs + models, no checkpoints):

```bash
pip install git+https://github.com/sh4r11f/illusion-rnn.git
```

## Quickstart

```python
import illusion_rnn as ir

# a TAM environment (stimuli ship inside the package)
env = ir.make_env("tam", box_shape="square", variant="standard",
                  stim_ori="horizontal", img_size=64)
env.seed(0)
ir.plot_trials(env, n_trials=2)

# evaluate the shipped reference RNN on it
model = ir.load_rnn("checkpoints/rnn-pixel_h2048_tam-horiz.pt")
print(ir.evaluate(model, env, n_trials=100).accuracy)

# train your own
dataset = ir.make_dataset(env, batch_size=16, seq_len=100)
rnn = ir.RNNNet(input_size=64 * 64, hidden_size=256, output_size=6, dt=50)
history = ir.train(rnn, dataset, n_epochs=200)
```

See `notebooks/01_quickstart.ipynb` (tour + generalization result) and
`notebooks/02_train.ipynb` (training from scratch).

## The generalization result

The reference RNN — trained only on the *standard* TAM set — transfers (or
fails to, per variant) as follows:

![Generalization across TAM variants](figures/generalization.png)

Exact numbers and every shipped model: [`checkpoints/MANIFEST.md`](checkpoints/MANIFEST.md).

## Plug in your own model

`ir.train` / `ir.evaluate` accept any callable module with the contract
`model(x: (T, B, F)) -> (out: (T, B, n_actions), activity: (T, B, H))`.
Two input regimes:

- **Raw pixels** (default): frames are flattened to `F = img_size²`.
- **Feature encoder**: pass `encoder=ir.cnn_encoder(your_cnn)` where the CNN
  maps `(N, 1, H, W) -> (logits, features)`; the RNN then sees `F =
  feature_dim`. The shipped `rnn-cnnfeat64_*` checkpoints use the packaged
  `ShapesCNN` this way (100×100 inputs — see the manifest).

Custom stimuli: give the loaders a directory of your own JPGs
(`ir.load_tam(64, source="path/to/frames")`) following the filename
conventions in `illusion_rnn/stimuli.py`, and pass the result to the env via
`stimuli=`.

## Layout

```
illusion_rnn/       the package: envs, stimulus loaders, models, train/eval, plotting
checkpoints/        trained models (git LFS) + MANIFEST.md
notebooks/          01_quickstart, 02_train; legacy/ holds the 2022-23 research record
scripts/            optional: retrain the ShapesCNN feature extractor
data/shape_dataset/ training data for that CNN (git LFS, 90 MB)
tests/              pytest suite (checkpoint tests skip without LFS content)
```

## Limitations

The stimulus set is small and hand-made (a few exemplars per shape ×
condition); results should be read as a proof-of-concept testbed, not a
benchmark. Trials follow one fixed timing template (fixation → 5 frames →
decision at dt = 50 ms), configurable via the `timing` argument.

## Attributions

- Task framework: [neurogym](https://github.com/neurogym/neurogym)
  (Molano-Mazón et al., 2022).
- Shape-classification data for the CNN encoder: 2D geometric shapes dataset,
  El Korchi & Ghanou (2020), <https://doi.org/10.17632/wzr2yv7r53.1>.
- `scripts/train_shape_cnn.py` derives from a classroom implementation
  adapted for this project in 2022.
- The 2022–2023 research notebooks this grew from are preserved in
  `notebooks/legacy/`.

## License

MIT — see [LICENSE](LICENSE).
````

- [ ] **Step 2: Verify referenced paths exist**

```bash
uv run python - <<'EOF'
import re
text = open("README.md").read()
paths = set(re.findall(r"\]\(([^)#>h][^)]*)\)|!\[[^\]]*\]\(([^)]+)\)", text))
from pathlib import Path
missing = [p for tup in paths for p in tup if p and not Path(p).exists()]
print("missing:", missing)
assert not missing
EOF
```

Expected: `missing: []`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Write real README: TAM intro, install, quickstart, attributions" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
```

---

### Task 14: CI + final verification

**Files:**
- Create: `.github/workflows/test.yml`
- Test: full local suite + wheel build + acceptance checklist

**Interfaces:**
- Consumes: everything.
- Produces: CI config; the finished branch.

- [ ] **Step 1: Create `.github/workflows/test.yml`**

```yaml
name: tests

on:
  push:
    branches: [main, testbed]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
        # no LFS fetch: checkpoint tests skip when content is absent
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync
      - run: uv run pytest -q
```

- [ ] **Step 2: Prove the LFS-skip path works (simulates CI)**

```bash
uv run python - <<'EOF'
# Point the checkpoint tests at a dir of fake pointers and confirm they skip.
import subprocess, tempfile, shutil
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
repo = Path.cwd()
fake = tmp / "checkpoints"
fake.mkdir()
for f in (repo / "checkpoints").glob("*.pt"):
    (fake / f.name).write_text("version https://git-lfs...\n")  # ~40-byte pointer stand-in
# Monkeypatch via env: run pytest with CHECKPOINT_DIR swapped by a conftest shim
shim = tmp / "conftest.py"
shim.write_text(
    "import tests.test_checkpoints as tc\n"
    f"tc.CHECKPOINT_DIR = __import__('pathlib').Path({str(fake)!r})\n"
)
r = subprocess.run(
    ["uv", "run", "pytest", "tests/test_checkpoints.py", "-v", "-p", "no:cacheprovider",
     "--rootdir", str(repo), "--confcutdir", str(repo)],
    env={**__import__('os').environ, "PYTHONPATH": str(tmp)},
    capture_output=True, text=True, cwd=repo,
)
print(r.stdout[-2000:])
shutil.rmtree(tmp)
assert "skipped" in r.stdout
EOF
```

Expected: the checkpoint tests report SKIPPED, none fail. (If the shim approach misbehaves on your pytest version, an acceptable fallback: temporarily `mv checkpoints checkpoints.bak`, run `uv run pytest tests/test_checkpoints.py -v` expecting skips, then `mv` back. Do NOT leave the tree modified.)

- [ ] **Step 3: Wheel sanity — stimuli ship in the package**

```bash
uv build
unzip -l dist/illusion_rnn-1.0.0-py3-none-any.whl | grep -c "stimuli_data.*\.jpg"
rm -rf dist
```

Expected: count > 400 (all stimulus JPGs packaged).

- [ ] **Step 4: Full local verification (spec §15 acceptance criteria)**

```bash
uv run pytest -q                    # 1. all green, checkpoint tests active
git status --short                  # 5. clean tree
```

Walk the checklist and confirm each item:
1. `uv sync` + `pytest` green with checkpoint tests running (not skipped).
2. Both notebooks executed end-to-end (Tasks 11–12 did this via nbconvert).
3. Flagship gate met (Task 8 Step 4 printed PASS; if Step 5 retrained, manifest says so).
4. README complete; `checkpoints/MANIFEST.md` documents all shipped checkpoints.
5. No `src/`, `parameters.json`, `.idea/`, or legacy scripts at the tip; legacy notebooks in `notebooks/legacy/`; prototype checkpoints in `checkpoints/legacy/`.

- [ ] **Step 5: Commit and wrap up**

```bash
git add .github
git commit -m "Add CI workflow (pytest on 3.10/3.12, no LFS)" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_01XbMes3E58A1zxjDopGSGjQ"
git log --oneline main..testbed
```

Expected: a tidy series of commits from scaffold to CI. Do not push or merge — that is the user's call.




