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
