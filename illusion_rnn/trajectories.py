"""Paired apparent-motion movies: vary tangential acceleration and path independently."""
from dataclasses import dataclass

import numpy as np

from illusion_rnn.apparent import MotionBatch, _mask

ACCEL = ("steady", "speed_up", "slow_down")
PATH = ("straight", "arc_left", "arc_right")
SHAPES = ("square", "circle", "triangle")
SIDE, FRAMES = 32, 8
ANGLE = 2 * np.pi / 3


@dataclass
class Chords:
    origin: np.ndarray  # midpoint of the chord, in shape top-left coordinates
    direction: np.ndarray
    length: np.ndarray
    sizes: np.ndarray
    shapes: np.ndarray


def sample_chords(rng, n):
    """Choose a common placement that fits *every* path, without clipping."""
    sizes = rng.integers(4, 9, n)
    shapes = rng.choice(SHAPES, n)
    length = rng.uniform(10, 18, (n, 1))
    phi = rng.uniform(-np.pi, np.pi, n)
    direction = np.stack((np.cos(phi), np.sin(phi)), axis=1)
    sagitta = length / 2 * np.tan(ANGLE / 4)
    half_box = length / 2 * abs(direction) + sagitta * abs(direction[:, ::-1])
    origin = rng.uniform(half_box, SIDE - sizes[:, None] - half_box)
    return Chords(origin, direction, length, sizes, shapes)


def trajectory(chords, accel, path):
    """Continuous top-left (x,y), before pixel rounding; factors can vary by trial."""
    a = np.broadcast_to(accel, len(chords.sizes))[:, None]
    p = np.broadcast_to(path, a.shape[0])[:, None]
    if not np.isin(a, ACCEL).all() or not np.isin(p, PATH).all():
        raise ValueError("unknown acceleration or path")
    t = np.linspace(0, 1, FRAMES)
    s = np.where(a == "speed_up", t**2, np.where(a == "slow_down", 2*t - t**2, t))
    r = chords.length / (2 * np.sin(ANGLE / 2))
    phase = ANGLE * (s - .5)
    along = np.where(p == "straight", chords.length * (s - .5), r * np.sin(phase))
    offset = ((p == "arc_left").astype(float) - (p == "arc_right")) * r * (
        np.cos(phase) - np.cos(ANGLE / 2))
    along[:, 0], along[:, -1] = -chords.length[:, 0] / 2, chords.length[:, 0] / 2
    offset[:, [0, -1]] = 0.
    # Screen coordinates: left of a rightward heading is negative y.
    normal = chords.direction[:, ::-1] * [1, -1]
    return (chords.origin[:, None] + along[..., None] * chords.direction[:, None]
            + offset[..., None] * normal[:, None])


def render(chords, accel, path):
    return render_tracks(trajectory(chords, accel, path), chords.sizes, chords.shapes)


def render_tracks(tracks, sizes, shapes):
    top_left = np.rint(tracks).astype(int)
    frames = np.zeros((len(sizes), FRAMES, SIDE, SIDE), dtype=np.float32)
    centres = top_left.astype(np.float32)
    for i, (size, shape) in enumerate(zip(sizes, shapes)):
        ink = _mask(shape, size)
        y, x = np.nonzero(ink)
        centres[i] += [x.mean(), y.mean()]
        for t, (left, top) in enumerate(top_left[i]):
            frames[i, t, top:top+size, left:left+size] = ink
    return MotionBatch(frames, centres, sizes, tuple(shapes))


def sample_trajectories(rng, batch_size, n_frames=FRAMES):
    if n_frames != FRAMES:
        raise ValueError(f"expected {FRAMES} frames")
    return render(sample_chords(rng, batch_size), rng.choice(ACCEL, batch_size),
                  rng.choice(PATH, batch_size))


PREFIX = 4


def forecast_tracks(rng, n):
    """(group, acceleration, path, time, xy), matched on the last two visible frames.

    k=0 is the last flash. The segment ending at k has arclength v+a*k,
    so at k=0 *all* alternatives have the same displacement along a given path.
    """
    k = np.arange(FRAMES) - (PREFIX - 1)
    rate = rng.uniform(.16, .23, (n, 1, 1)) * np.array([0., 1., -1.])[None, :, None]
    s = k + rate * k * (k+1) / 2  # arclength in units of the last observed step
    radius = (FRAMES - 1) / ANGLE
    along = radius * np.sin(s / radius)
    bend = radius * (1 - np.cos(s / radius))
    u = np.stack([s, along, along], axis=2)
    w = np.stack([np.zeros_like(s), bend, -bend], axis=2)
    phi = rng.uniform(-np.pi, np.pi, n)
    unit = np.stack([np.cos(phi), np.sin(phi)], axis=-1)[:, None, None, None]
    tracks = u[..., None] * unit + w[..., None] * (unit[..., ::-1] * [1, -1])
    sizes = rng.integers(4, 9, n)
    lo, hi = tracks.min((1, 2, 3)), tracks.max((1, 2, 3))
    # Fit all nine options with *one* scale and translation per group.
    limit = ((SIDE - sizes)[:, None] / (hi - lo)).min(1)
    speed = rng.uniform(1.7, np.minimum(2.35, limit))[:, None]
    anchor = rng.uniform(-lo * speed, SIDE - sizes[:, None] - hi * speed)
    tracks = tracks * speed[:, None, None, None] + anchor[:, None, None, None]
    return tracks, sizes, rng.choice(SHAPES, n)
