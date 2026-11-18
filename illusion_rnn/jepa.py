"""A small frame-level joint-embedding predictor for masked motion movies."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from illusion_rnn.trajectories import FRAMES, SIDE


def _check_training_mask(visible):
    if visible.all() or not visible.any():
        raise ValueError("a training batch needs both visible and masked frames")


class FrameEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project = nn.Sequential(nn.Linear((SIDE//2)**2, 96), nn.GELU(), nn.Linear(96, dim),
                                     nn.LayerNorm(dim, elementwise_affine=False))

    def forward(self, frames):
        return self.project(F.avg_pool2d(frames, 2).flatten(2))


class MoviePredictor(nn.Module):
    """One ordered slot per time point. All predictions see the whole visible context."""
    def __init__(self, dim=48, hidden=192):
        super().__init__()
        self.encoder = FrameEncoder(dim)
        self.context = nn.Sequential(nn.Flatten(1), nn.Linear(FRAMES*(dim+1), hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.predictor = nn.Linear(hidden, FRAMES*dim)

    def forward(self, frames, visible):
        x = self.encoder(frames.masked_fill(~visible[..., None, None], 0))
        slots = torch.cat((x.masked_fill(~visible[..., None], 0), visible[..., None]), dim=-1)
        h = self.context(slots)
        return h, self.predictor(h).reshape_as(x), x


class MotionJEPA(MoviePredictor):
    def __init__(self, dim=48, hidden=192):
        super().__init__(dim, hidden)
        self.target = deepcopy(self.encoder).requires_grad_(False)

    @torch.no_grad()
    def update_target(self, decay):
        for teacher, online in zip(self.target.parameters(), self.encoder.parameters()):
            teacher.lerp_(online, 1-decay)

    def objective(self, frames, visible):
        _check_training_mask(visible)
        _, predicted, x = self(frames, visible)
        with torch.no_grad():
            target = self.target(frames)
        prediction = F.mse_loss(predicted[~visible], target[~visible])
        # Prevent the encoder from satisfying the prediction task with a constant.
        z = x[visible]
        z = z - z.mean(0)
        cov = z.T @ z / max(len(z)-1, 1)
        variance = F.relu(.8 - (cov.diag()+1e-4).sqrt()).mean()
        off_diag = cov - torch.diag_embed(cov.diag())
        return prediction + variance + .04 * off_diag.square().sum()/cov.shape[0]


class PixelControl(MoviePredictor):
    """Same online encoder, temporal context and predictor; supervise pixels instead."""
    def __init__(self, dim=48, hidden=192):
        super().__init__(dim, hidden)
        self.pixels = nn.Linear(dim, SIDE**2)

    def objective(self, frames, visible):
        _check_training_mask(visible)
        _, predicted, _ = self(frames, visible)
        return F.binary_cross_entropy_with_logits(self.pixels(predicted[~visible]),
                                                  frames[~visible].flatten(1))
