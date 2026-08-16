"""Programmatic stimulus generator for the correspondence TAM task.

Stimuli are drawn analytically rather than loaded from disk, so every geometric
parameter of a trial is recoverable and the train/test splits can be defined on
parameters instead of on filenames. numpy only -- scipy is not a project
dependency.

Ink convention matches ``illusion_rnn.stimuli``: float arrays in [0, 1] with
ink = 1 on background = 0.
"""

from dataclasses import dataclass

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


DIRECTIONS = ("left", "right")
RENDERS = ("filled", "outline")
INK_MATCHES = ("none", "energy")
FAMILIES = ("balanced", "classic")
TRANSFORMS = ("growth", "shrink")
TRANSFORM_MODES = ("growth", "shrink", "growth+shrink")
CLASSIC_RAISE = 2.0   # raised end is this multiple of the bar's height


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


def render_trial(params: dict, img_size: int = 64) -> TrialStimulus:
    """Render one trial from a complete parameter dict.

    For ``transform="growth"`` (the primary condition), frame 2 is built from
    ``bar_left``, ``bar_length`` and ``shape_size`` alone -- never from
    ``direction``. That is what makes the frame-2-only baseline provably 50%,
    and ``test_frame2_is_bit_identical_across_directions`` pins it for this
    condition.

    This guarantee is scoped to ``transform="growth"`` and does NOT extend to
    ``transform="shrink"``: the frame1/frame2 swap below deliberately makes
    the *shrink* trial's ``frame2`` direction-dependent, because a
    growth-labelled-"right" trial and a shrink-labelled-"left" trial are meant
    to share the same underlying frame pair in reversed temporal order (the
    order-control condition Task 5 builds on top of this). Do not assume the
    frame-2-identity invariant holds for shrink trials.
    """
    _validate_choice("shape", params["shape"], SHAPE_NAMES)
    _validate_choice("direction", params["direction"], DIRECTIONS)
    _validate_choice("render", params["render"], RENDERS)
    _validate_choice("ink_match", params["ink_match"], INK_MATCHES)
    _validate_choice("transform", params["transform"], TRANSFORMS)
    _validate_choice("family", params["family"], FAMILIES)

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

    if params["family"] == "classic":
        frame1, frame2 = _render_classic(params, img_size)
    else:
        centre = img_size // 2
        top = centre - size // 2
        rows = slice(top, top + size)

        # --- frame 2: the bar. Depends only on (left, length, size).
        bar = np.ones((size, length), dtype=bool)
        frame2 = np.zeros((img_size, img_size))
        frame2[rows, left:left + length] = _render_element(
            bar, params, float(bar.sum()),
        )

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
    _validate_choice("transform", transform, TRANSFORM_MODES)

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

    # 4. growth+shrink draws a CONCRETE transform per trial, after direction.
    #    "growth+shrink" is a sampling mode, not a real value of the
    #    render_trial-facing "transform" field -- it must never leak into the
    #    returned params dict. Drawing it here (after direction) mirrors the
    #    "label drawn last" discipline: the transform draw does not feed back
    #    into anything above it, so its position relative to `direction`
    #    cannot introduce a correlation, but drawing it after keeps the
    #    invariant-relevant draws (which do have to precede direction)
    #    grouped together and unambiguous to audit.
    if transform == "growth+shrink":
        trial_transform = str(rng.choice(TRANSFORMS))
    else:
        trial_transform = transform

    return dict(
        shape=shape, shape_size=shape_size, bar_left=left, bar_length=length,
        direction=direction, family=family, transform=trial_transform,
        render=render, stroke_width=stroke_width, ink_match=ink_match,
    )
