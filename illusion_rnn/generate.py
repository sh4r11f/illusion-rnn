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


def _opposite_direction(direction: str) -> str:
    return "left" if direction == "right" else "right"


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
    if params["family"] == "classic" and length < 2 * size:
        msg = (
            f"classic family needs bar_length >= 2*shape_size to keep the two "
            f"end squares from overlapping; got bar_length={length}, "
            f"shape_size={size}"
        )
        raise ValueError(msg)

    if params["family"] == "classic":
        frame1, frame2 = _render_classic(params, img_size)
        # _render_classic does not consult transform -- it always builds the
        # (constant-frame1, raised-frame2) pair. "shrink" reverses the two
        # frames' temporal order here, same as it always has; this family
        # does not participate in the growth+shrink order-control guarantee
        # (that is scoped to the balanced family below), so a plain swap is
        # sufficient and correct.
        if params["transform"] == "shrink":
            frame1, frame2 = frame2, frame1
    else:
        centre = img_size // 2
        top = centre - size // 2
        rows = slice(top, top + size)

        bar = np.ones((size, length), dtype=bool)
        frame_bar = np.zeros((img_size, img_size))
        frame_bar[rows, left:left + length] = _render_element(
            bar, params, float(bar.sum()),
        )

        # For "shrink" trials, the shape occupies the end a "growth" trial of
        # the OPPOSITE direction would use -- this is what lets a growth
        # trial and a shrink trial share IDENTICAL frame content in reversed
        # temporal order with opposite labels (the growth+shrink order
        # control). A growth-right trial places the shape at the left end;
        # a shrink-left trial must place its (frame2) shape at that SAME
        # left end, not at the position its own "left" direction would
        # naively imply -- only then do the two trials' frame1/frame2 pairs
        # actually match when the array roles are swapped.
        placement_direction = (
            params["direction"] if params["transform"] == "growth"
            else _opposite_direction(params["direction"])
        )
        mask = shape_mask(params["shape"], size)
        shape_left = (
            left if placement_direction == "right" else left + length - size
        )
        frame_shape = np.zeros((img_size, img_size))
        frame_shape[rows, shape_left:shape_left + size] = _render_element(
            mask, params, float(mask.sum()),
        )

        if params["transform"] == "growth":
            frame1, frame2 = frame_shape, frame_bar
        else:
            frame1, frame2 = frame_bar, frame_shape

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
    #
    # Map the centre range onto an ABSOLUTE pixel band, anchored to the
    # longest bar the range can produce, rather than onto a span that moves
    # with the sampled length. A fractional-of-`span` mapping ties `left`'s
    # usable range to `length`: a long train-split bar and a short
    # test-split bar can then land on the very same `bar_left` pixel even
    # though their bar_centre_range fractions differ, silently overlapping
    # the two splits. Anchoring the band to `bar_length_range[1]` (the
    # longest bar this range will ever draw) and then capping `length` to
    # what still fits inside `high` keeps every bar's centre within the
    # band regardless of its individual length, so the bands the splits
    # are built from stay genuinely disjoint pixel-for-pixel.
    usable = img_size - bar_length_range[1]
    if usable < 1:
        msg = (
            f"bar_length_range[1]={bar_length_range[1]} leaves no room on a "
            f"{img_size}px canvas"
        )
        raise ValueError(msg)
    # Both bounds use ceil (not floor for `low`), so that two splits sharing
    # a boundary fraction (e.g. train's 0.6 upper edge and test_position's
    # 0.6 lower edge) partition the pixel band cleanly instead of both
    # claiming the same pixel. With usable=36 and fraction 0.6, the boundary
    # sits at a non-integer 21.6: ceil(21.6)=22 is used as both train's
    # exclusive `high` (so train covers pixels 0..21, all < 21.6) and
    # test_position's inclusive `low` (so test_position starts at pixel 22,
    # the first integer >= 21.6). Using floor for `low` instead would give
    # 21, which duplicates train's last pixel -- a genuine one-pixel overlap
    # this project's tests must catch, not paper over.
    low = int(np.ceil(bar_centre_range[0] * usable))
    high = int(np.ceil(bar_centre_range[1] * usable))
    # Cap bar_length to what fits within `high` so the later `left + length`
    # placement cannot spill past the band's right edge -- a clamp on `left`
    # instead would reintroduce the same length-dependent overlap this fix
    # removes.
    length = int(rng.integers(
        bar_length_range[0], min(bar_length_range[1], img_size - high) + 1,
    ))
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
    #
    # A consequence of that position split: restricting bar_centre_range
    # narrows the range of `bar_left` (and hence of `shape_left`, frame 1's
    # only observable), which makes shape position MORE diagnostic of
    # direction, not less. Concretely, a frame-1-only Bayes-optimal
    # classifier (majority-vote-by-shape-position over sample_params draws;
    # see scripts/verify_baselines.py) scores:
    #
    #   bar_centre_range=(0.0, 1.0)  full track        0.697
    #   bar_centre_range=(0.0, 0.6)  this "train" split 0.818  <-- narrower
    #   bar_centre_range=(0.6, 1.0)  "test_position"    0.942  <-- narrower still
    #
    # So the FF1Only baseline landing at ~0.818 on this split is NOT a case
    # of a static/frame-1-only model mysteriously beating theory -- it is
    # simply sitting at the Bayes-optimal ceiling for the position band
    # `train` restricts itself to. An earlier note attributed the ~0.82
    # figure to "a smooth neural classifier beating a per-pixel
    # majority-vote heuristic"; that explanation is wrong and has been
    # superseded by this measured, position-split explanation. The honest
    # framing for any binding model evaluated on `train`: it must beat
    # ~0.82, and that ~0.82 is an artifact of our own position split, not
    # evidence that frame-1-only information is unusually rich.
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


def pixel_label_mi(frames: np.ndarray, labels: np.ndarray, n_bins: int = 8):
    """Per-pixel mutual information in bits between pixel value and label.

    Pixel values are binned into ``n_bins`` equal-width bins; the label is
    binary. Returns an ``(H, W)`` array. Used to verify that frame 2 carries no
    direction information in the balanced family -- and that it DOES in the
    classic diagnostic family.

    Statistical note: this is the plain "plug-in" (maximum-likelihood) MI
    estimator -- MI(X;Y) = sum_{x,y} p(x,y) log2(p(x,y) / (p(x)p(y))) with
    p(x,y), p(x), p(y) all estimated as sample frequencies. That estimator is
    known to carry a small positive bias for finite samples (roughly
    (bins-1)(labels-1) / (2 N ln 2) bits), which is why the pass condition for
    a genuinely-independent pixel is "near the noise floor" rather than
    exactly zero -- with n_bins=8, 2 labels and N=3000 trials the expected
    bias is about 0.0017 bits, comfortably under the 0.02-bit checkpoint
    threshold.
    """
    frames = np.asarray(frames, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_trials, height, width = frames.shape
    flat = frames.reshape(n_trials, -1)

    # Per-pixel equal-width binning over that pixel's OWN observed range
    # (not a fixed 0..1 range) so pixels that are constant across every
    # trial -- e.g. background corners a shape never reaches -- collapse to
    # a single bin instead of being spread thin across bins that would
    # inflate their apparent variance (and hence estimation noise).
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)   # avoid /0 for constant pixels
    # (value - lo) / span lands in [0, 1]; the pixel at the maximum observed
    # value maps to exactly n_bins before flooring, which the clip pulls back
    # into the last bin instead of leaving it as an out-of-range index.
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
