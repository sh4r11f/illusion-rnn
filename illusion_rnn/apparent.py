"""Classical apparent motion as frame imputation.

A shape translates at constant velocity from point ``a`` to point ``b`` over
``n_frames`` frames. Training shows the network the movie with some frames
blanked and asks it to reconstruct the *whole* movie; the test is the limiting
case in which only the two endpoint frames are shown (apparent motion). The
question is whether the reconstruction of the blanked frames contains the
intermediate positions.

This is not a neurogym ``TrialEnv``: the target is a movie rather than a class
label per timestep, so batches are sampled directly with numpy.

Ink convention matches ``illusion_rnn.generate``: float arrays in [0, 1] with
ink = 1 on background = 0. Positions are ``(x, y)`` shape centres in pixels.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn, optim

from illusion_rnn.generate import SHAPE_NAMES, shape_mask
from illusion_rnn.models import CTRNN
from illusion_rnn.training import resolve_device

READOUTS = ("final", "online")
CELLS = ("ctrnn", "gru")


@dataclass(frozen=True)
class MotionBatch:
    """``frames`` (B, T, H, W) is the unmasked movie, ``centres`` (B, T, 2) the
    shape's ink centroid in every frame, ``sizes`` (B,) the shape box side in pixels."""

    frames: np.ndarray
    centres: np.ndarray
    sizes: np.ndarray
    shapes: tuple


@lru_cache(maxsize=None)
def _mask(shape: str, size: int) -> np.ndarray:
    return shape_mask(shape, size).astype(np.float32)


def _render(shape: str, size: int, centre, img_size: int) -> np.ndarray:
    frame = np.zeros((img_size, img_size), dtype=np.float32)
    left = int(round(centre[0] - (size - 1) / 2))
    top = int(round(centre[1] - (size - 1) / 2))
    frame[top:top + size, left:left + size] = _mask(shape, size)
    return frame


def sample_motion(
    rng: np.random.Generator,
    batch_size: int,
    n_frames: int = 8,
    img_size: int = 32,
    shapes=("square", "circle", "triangle"),
    size_range=(4, 8),
    disp_range=(0.25, 0.75),
    p_static: float = 0.1,
) -> MotionBatch:
    """Sample constant-velocity trajectories.

    ``disp_range`` bounds the a-to-b distance as a fraction of ``img_size``;
    direction is uniform over the circle. With probability ``p_static`` the
    shape does not move (``a == b``), which is what the same-place flicker
    control needs the network to have seen.
    """
    for shape in shapes:
        if shape not in SHAPE_NAMES:
            msg = f"Unknown shape {shape!r}; expected one of {SHAPE_NAMES}"
            raise ValueError(msg)

    frames = np.zeros((batch_size, n_frames, img_size, img_size), dtype=np.float32)
    centres = np.zeros((batch_size, n_frames, 2), dtype=np.float32)
    sizes = np.zeros(batch_size, dtype=np.int64)
    names = []
    fractions = np.linspace(0.0, 1.0, n_frames)[:, None]

    for i in range(batch_size):
        shape = shapes[int(rng.integers(len(shapes)))]
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        lo, hi = size / 2, img_size - size / 2 - 0.5
        static = rng.random() < p_static
        if not static and disp_range[0] * img_size > np.sqrt(2) * (hi - lo):
            msg = (f"a size-{size} shape cannot move {disp_range[0]:g} x {img_size} px "
                   f"on a {img_size} px canvas; lower disp_range or size_range")
            raise ValueError(msg)
        for attempt in range(10_000):
            a = rng.uniform(lo, hi, size=2)
            if static:
                b = a
                break
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(*disp_range) * img_size
            b = a + dist * np.array([np.cos(angle), np.sin(angle)])
            if np.all(b >= lo) and np.all(b <= hi):
                break
        else:
            msg = "could not fit a trajectory on the canvas; check disp_range and size_range"
            raise ValueError(msg)
        path = a + fractions * (b - a)
        # The rendered box centre is the rounded one.
        offset = (size - 1) / 2
        path = np.round(path - offset) + offset
        for t in range(n_frames):
            frames[i, t] = _render(shape, size, path[t], img_size)
        # Ground truth is the ink centroid, which is what the metrics measure
        # on reconstructions; for a triangle it sits below the box centre.
        mask = _mask(shape, size)
        ys, xs = np.nonzero(mask)
        centres[i] = path + (np.array([xs.mean(), ys.mean()]) - offset)
        sizes[i] = size
        names.append(shape)
    return MotionBatch(frames, centres, sizes, tuple(names))


def _longest_run(blank: np.ndarray) -> tuple:
    """(start, length) of the longest run of True in a 1-D bool array."""
    best, start, length = (0, 0), 0, 0
    for t, value in enumerate(blank):
        if value:
            if length == 0:
                start = t
            length += 1
            if length > best[1]:
                best = (start, length)
        else:
            length = 0
    return best


def sample_masks(
    rng: np.random.Generator,
    batch_size: int,
    n_frames: int,
    max_gap: int | None = None,
    max_rate: float = 0.8,
) -> np.ndarray:
    """Boolean (B, T) array, True = frame is shown.

    Each trial blanks frames independently at a rate drawn from
    ``U(0, max_rate)``. Two constraints are then enforced: at least two frames
    stay visible (one frame does not determine a velocity), and no run of
    consecutive blanked frames is longer than ``max_gap``. ``max_gap`` is what
    keeps the apparent-motion test (``n_frames - 2`` consecutive blanks) out of
    the training distribution; ``None`` leaves runs unbounded.
    """
    rates = rng.uniform(0, max_rate, size=(batch_size, 1))
    blank = rng.random((batch_size, n_frames)) < rates
    for row in blank:
        while (~row).sum() < 2:
            row[rng.choice(np.flatnonzero(row))] = False
        if max_gap is not None:
            while True:
                start, length = _longest_run(row)
                if length <= max_gap:
                    break
                row[start + int(rng.integers(length))] = False
    return ~blank


def endpoint_mask(batch_size: int, n_frames: int) -> np.ndarray:
    """The apparent-motion condition: only the first and last frame shown."""
    visible = np.zeros((batch_size, n_frames), dtype=bool)
    visible[:, [0, -1]] = True
    return visible


def gap_mask(batch_size: int, n_frames: int, gap: int) -> np.ndarray:
    """A single centred run of ``gap`` blanked frames, everything else shown."""
    if not 0 <= gap <= n_frames - 2:
        msg = f"gap must be in [0, {n_frames - 2}], got {gap}"
        raise ValueError(msg)
    visible = np.ones((batch_size, n_frames), dtype=bool)
    start = (n_frames - gap) // 2
    visible[:, start:start + gap] = False
    return visible


def apply_mask(frames: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Blank (zero) the frames of a (B, T, H, W) movie where ``visible`` is False."""
    return frames * visible[:, :, None, None]


class ImputationNet(nn.Module):
    """Recurrent net that reads a masked movie and reconstructs the full one.

    ``forward(x)`` takes ``(T, B, F)`` and returns ``(logits (T, B, F),
    activity (T, B, hidden))`` -- the repo's usual model contract, with pixel
    logits in place of class logits.

    ``readout="final"`` decodes all ``T`` frames from the hidden state after
    the last frame, so the reconstruction of a blanked frame can depend on
    frames shown after it. ``readout="online"`` decodes frame ``t`` from the
    hidden state at ``t`` and can only extrapolate.
    """

    def __init__(self, input_size: int, hidden_size: int, n_frames: int,
                 cell: str = "ctrnn", readout: str = "final",
                 dt: float | None = 50, tau: float = 100):
        super().__init__()
        if cell not in CELLS:
            msg = f"Unknown cell {cell!r}; expected one of {CELLS}"
            raise ValueError(msg)
        if readout not in READOUTS:
            msg = f"Unknown readout {readout!r}; expected one of {READOUTS}"
            raise ValueError(msg)
        self.n_frames = n_frames
        self.input_size = input_size
        self.readout = readout
        if cell == "ctrnn":
            self.rnn = CTRNN(input_size, hidden_size, dt=dt, tau=tau)
        else:
            self.rnn = nn.GRU(input_size, hidden_size)
        n_out = input_size * (n_frames if readout == "final" else 1)
        self.fc = nn.Linear(hidden_size, n_out)

    def forward(self, x: torch.Tensor):
        n_steps, batch, _ = x.shape
        if n_steps != self.n_frames:
            msg = f"expected {self.n_frames} timesteps, got {n_steps}"
            raise ValueError(msg)
        activity, _ = self.rnn(x)
        if self.readout == "online":
            return self.fc(activity), activity
        out = self.fc(activity[-1]).reshape(batch, n_steps, self.input_size)
        return out.permute(1, 0, 2), activity


def _to_model_input(frames: np.ndarray, device) -> torch.Tensor:
    """(B, T, H, W) numpy -> (T, B, H*W) tensor."""
    flat = frames.reshape(*frames.shape[:2], -1).transpose(1, 0, 2)
    return torch.from_numpy(np.ascontiguousarray(flat)).float().to(device)


def train_imputation(
    model,
    n_steps: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_gap: int | None = None,
    seed: int = 0,
    device=None,
    log_every: int = 500,
    motion_sampler=sample_motion,
    **motion_kwargs,
) -> dict:
    """Train ``model`` to reconstruct full movies from masked ones.

    The loss is pixel-wise binary cross-entropy over *every* frame, shown or
    blanked. Returns ``{"loss": [...]}`` per step. ``motion_kwargs`` pass
    through to ``motion_sampler`` (``sample_motion`` by default).
    """
    device = resolve_device(device)
    model = model.to(device)
    model.train()
    rng = np.random.default_rng(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    history = {"loss": []}

    for step in range(n_steps):
        batch = motion_sampler(rng, batch_size, n_frames=model.n_frames, **motion_kwargs)
        visible = sample_masks(rng, batch_size, model.n_frames, max_gap=max_gap)
        x = _to_model_input(apply_mask(batch.frames, visible), device)
        target = _to_model_input(batch.frames, device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1}/{n_steps}  loss {loss.item():.4f}")
    return history


@torch.no_grad()
def reconstruct(model, frames: np.ndarray, visible: np.ndarray, device=None) -> np.ndarray:
    """Pixel probabilities (B, T, H, W) for a masked (B, T, H, W) movie."""
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    logits, _ = model(_to_model_input(apply_mask(frames, visible), device))
    probs = torch.sigmoid(logits).permute(1, 0, 2).cpu().numpy()
    return probs.reshape(frames.shape)


def _iou(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """IoU over the last two axes; 1.0 where both are empty."""
    inter = (pred & target).sum(axis=(-2, -1))
    union = (pred | target).sum(axis=(-2, -1))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def imputation_metrics(recon: np.ndarray, batch: MotionBatch, visible: np.ndarray) -> dict:
    """Score the reconstruction of the blanked frames.

    Per blanked frame the reconstruction is compared with three hypotheses
    about what the network could have drawn there:

    - ``iou_true``     the shape at its true interpolated position (fill-in),
    - ``iou_nearest``  the shape where it was in the nearest shown frame
      (a jump, no intermediate positions),
    - ``iou_union``    both neighbouring shown frames superimposed.

    ``path_slope`` is the regression slope of the reconstructed ink centroid's
    position along the a-to-b axis against the true position, over blanked
    frames of moving trials: 1 means the centroid travels the path, 0 means it
    does not move. A centroid alone cannot tell a moving shape from a
    cross-fade between the endpoints, which is why the IoUs are reported too.
    ``pos_err`` is the centroid's distance from the true centre in pixels.
    """
    n_batch, n_frames = visible.shape
    hard = recon > 0.5
    truth = batch.frames > 0.5
    shown = [np.flatnonzero(row) for row in visible]

    iou_true, iou_nearest, iou_union, pos_err, mass = [], [], [], [], []
    true_frac, recon_frac = [], []
    yy, xx = np.mgrid[0:recon.shape[-2], 0:recon.shape[-1]]

    for i in range(n_batch):
        a, b = batch.centres[i, 0], batch.centres[i, -1]
        axis = b - a
        norm = float(axis @ axis)
        for t in np.flatnonzero(~visible[i]):
            before = shown[i][shown[i] < t]
            after = shown[i][shown[i] > t]
            neighbours = [s[k] for s, k in ((before, -1), (after, 0)) if len(s)]
            nearest = min(neighbours, key=lambda s: abs(s - t))
            union = np.any(truth[i, neighbours], axis=0)

            iou_true.append(_iou(hard[i, t], truth[i, t]))
            iou_nearest.append(_iou(hard[i, t], truth[i, nearest]))
            iou_union.append(_iou(hard[i, t], union))

            total = float(recon[i, t].sum())
            mass.append(total / float(batch.frames[i, t].sum()))
            if total < 1e-6:
                continue
            centroid = np.array([(recon[i, t] * xx).sum(), (recon[i, t] * yy).sum()]) / total
            pos_err.append(float(np.linalg.norm(centroid - batch.centres[i, t])))
            if norm > 0:
                true_frac.append(float((batch.centres[i, t] - a) @ axis) / norm)
                recon_frac.append(float((centroid - a) @ axis) / norm)

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    slope = None  # undefined when nothing moves; None keeps the results valid JSON
    if len(set(np.round(true_frac, 6))) > 1:
        slope = float(np.polyfit(true_frac, recon_frac, 1)[0])
    return {
        "n_blanked": len(iou_true),
        "iou_true": mean(iou_true),
        "iou_nearest": mean(iou_nearest),
        "iou_union": mean(iou_union),
        "pos_err": mean(pos_err),
        "ink_ratio": mean(mass),
        "path_slope": slope,
    }


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a square structuring element; no wrap-around."""
    if radius < 1:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius)
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + height, dx:dx + width]
    return out


def _largest_patch(weights: np.ndarray, threshold: float):
    """Ink-weighted centre ``(x, y)`` of the 4-connected region of
    ``weights > threshold`` holding the most ink, or None if there is none."""
    active = weights > threshold
    height, width = active.shape
    seen = np.zeros_like(active)
    best_mass, best = 0.0, None
    for y0, x0 in zip(*np.nonzero(active)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, pixels = [(y0, x0)], []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < height and 0 <= nx < width and active[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = np.array(pixels).T
        w = weights[ys, xs]
        if w.sum() > best_mass:
            best_mass = float(w.sum())
            best = np.array([(w * xs).sum(), (w * ys).sum()]) / w.sum()
    return best


def patch_position_metrics(recon: np.ndarray, batch: MotionBatch, visible: np.ndarray,
                           threshold: float = 0.25, margin: int = 1) -> dict:
    """Where is the patch the network drew, ignoring shape and ink amount?

    In every blanked frame, ink inside the start-frame footprint (the shape in
    frame 0, dilated by ``margin`` pixels) is discarded, so ink that is merely
    held at the start position does not count. The predicted position is the
    ink-weighted centre of the largest remaining connected region above
    ``threshold``. All distances are in pixels, over moving trials only.

    - ``patch_found``   fraction of blanked frames with any such patch,
    - ``patch_err``     mean distance from the patch centre to the true centre
      (``patch_err_median``: median; ``patch_err_frac``: mean error divided by
      the trial's a-to-b distance),
    - ``truth_err``     the same measurement applied to the true movie. It is
      not 0: discarding the start footprint clips the true shape in the first
      frames after the start, so this is the best ``patch_err`` attainable,
    - ``stay_err``, ``jump_err``, ``midpoint_err``  what a centre sitting at
      the start, at the nearer endpoint, or halfway between the endpoints
      would score on the frames where a patch was found,
    - ``by_frame``      ``patch_found``, ``patch_err``, ``truth_err`` and
      ``jump_err`` per frame index (None where no frame qualifies).
    """
    n_frames = visible.shape[1]
    rows = {key: [] for key in ("frame", "err", "frac", "truth", "stay", "jump", "mid")}
    n_blanked = np.zeros(n_frames, dtype=int)

    for i in range(len(visible)):
        a, b = batch.centres[i, 0], batch.centres[i, -1]
        dist = float(np.linalg.norm(b - a))
        if dist == 0:
            continue
        keep = ~_dilate(batch.frames[i, 0] > 0.5, margin)
        for t in np.flatnonzero(~visible[i]):
            n_blanked[t] += 1
            centre = _largest_patch(recon[i, t] * keep, threshold)
            if centre is None:
                continue
            true = batch.centres[i, t]
            truth = _largest_patch(batch.frames[i, t] * keep, threshold)
            rows["frame"].append(t)
            rows["err"].append(np.linalg.norm(centre - true))
            rows["frac"].append(np.linalg.norm(centre - true) / dist)
            rows["truth"].append(np.nan if truth is None else np.linalg.norm(truth - true))
            rows["stay"].append(np.linalg.norm(a - true))
            rows["jump"].append(min(np.linalg.norm(a - true), np.linalg.norm(b - true)))
            rows["mid"].append(np.linalg.norm((a + b) / 2 - true))

    rows = {key: np.array(values, dtype=float) for key, values in rows.items()}
    total = int(n_blanked.sum())

    def mean(values):
        values = values[~np.isnan(values)]
        return float(values.mean()) if len(values) else None

    by_frame = {}
    for t in np.flatnonzero(n_blanked):
        sel = rows["frame"] == t
        by_frame[int(t)] = {
            "patch_found": float(sel.sum() / n_blanked[t]),
            "patch_err": mean(rows["err"][sel]),
            "truth_err": mean(rows["truth"][sel]),
            "jump_err": mean(rows["jump"][sel]),
        }
    return {
        "patch_found": len(rows["err"]) / total if total else None,
        "patch_err": mean(rows["err"]),
        "patch_err_median": float(np.median(rows["err"])) if len(rows["err"]) else None,
        "patch_err_frac": mean(rows["frac"]),
        "truth_err": mean(rows["truth"]),
        "stay_err": mean(rows["stay"]),
        "jump_err": mean(rows["jump"]),
        "midpoint_err": mean(rows["mid"]),
        "by_frame": by_frame,
    }


def evaluate_imputation(
    model,
    condition: str = "apparent",
    n_trials: int = 500,
    gap: int | None = None,
    seed: int = 1,
    device=None,
    **motion_kwargs,
) -> dict:
    """Evaluate on a named test condition.

    - ``"apparent"``  endpoints only, moving shapes: apparent motion.
    - ``"flicker"``   endpoints only, ``a == b``: same-place flicker control.
    - ``"gap"``       one centred run of ``gap`` blanked frames, moving shapes.
    """
    rng = np.random.default_rng(seed)
    n_frames = model.n_frames
    if condition == "apparent":
        motion_kwargs.setdefault("p_static", 0.0)
        visible = endpoint_mask(n_trials, n_frames)
    elif condition == "flicker":
        motion_kwargs["p_static"] = 1.0
        visible = endpoint_mask(n_trials, n_frames)
    elif condition == "gap":
        motion_kwargs.setdefault("p_static", 0.0)
        visible = gap_mask(n_trials, n_frames, gap)
    else:
        msg = f"Unknown condition {condition!r}; expected 'apparent', 'flicker' or 'gap'"
        raise ValueError(msg)
    batch = sample_motion(rng, n_trials, n_frames=n_frames, **motion_kwargs)
    recon = reconstruct(model, batch.frames, visible, device=device)
    result = imputation_metrics(recon, batch, visible)
    result["position"] = patch_position_metrics(recon, batch, visible)
    return result
