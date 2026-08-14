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
        if parts[0] not in SHAPES:
            msg = (
                f"Unexpected stimulus file {jpg.name!r}; expected "
                f"'<shape>-<cnt|track>-<direction>-<n>-f<m>.jpg' with shape in {SHAPES}"
            )
            raise ValueError(msg)
        if parts[2] not in MOTION_DIRECTIONS:
            msg = (
                f"Unexpected stimulus file {jpg.name!r}; expected "
                f"'<shape>-<cnt|track>-<direction>-<n>-f<m>.jpg' with direction in {MOTION_DIRECTIONS}"
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
