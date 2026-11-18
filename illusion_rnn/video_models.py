"""Spatial models for masked RGB video, decoded after the last flash."""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VideoConfig:
    frames: int = 8
    side: int = 128
    width: int = 256
    dim: int = 384
    heads: int = 6
    spatial_depth: int = 6
    temporal_depth: int = 4

    def __post_init__(self):
        if self.frames < 4 or self.side < 16 or self.side % 16:
            raise ValueError("need at least 4 frames and a crop divisible by 16")
        if self.width < 32 or self.width % 32:
            raise ValueError("ConvGRU width must be a positive multiple of 32")
        if min(self.dim, self.heads, self.spatial_depth, self.temporal_depth) < 1 or self.dim % self.heads:
            raise ValueError("depths must be positive and dim divisible by heads")


def _check(clip, visible, cfg):
    if clip.shape[1:] != (cfg.frames, 3, cfg.side, cfg.side) or clip.ndim != 5:
        raise ValueError("expected a B,T,3,H,W RGB clip matching the model config")
    if visible.shape != clip.shape[:2] or visible.dtype != torch.bool:
        raise ValueError("expected a boolean B,T visibility mask")


def interpolate_endpoints(values):
    """Linear crossfade of the endpoints, in the model's scoring space."""
    t = values.shape[1]
    weights = torch.linspace(0, 1, t, device=values.device, dtype=values.dtype)
    return torch.lerp(values[:, :1], values[:, -1:], weights.view(1, t, *([1] * (values.ndim - 2))))


class ConvGRUCell(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gates = nn.Conv2d(2 * width, 2 * width, 3, padding=1)
        self.candidate = nn.Conv2d(2 * width, width, 3, padding=1)

    def forward(self, x, h):
        reset, update = self.gates(torch.cat([x, h], 1)).sigmoid().chunk(2, 1)
        cand = self.candidate(torch.cat([x, reset * h], 1)).tanh()
        return (1 - update) * h + update * cand


class VideoConvGRU(nn.Module):
    """Two recurrent layers on a spatial grid. Only the final state feeds the RGB decoder."""
    def __init__(self, cfg: VideoConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        blocks = []
        channels = 4  # RGB plus an explicit visibility plane
        for out in (width // 4, width // 2, width, width):
            blocks.extend([nn.Conv2d(channels, out, 4, 2, 1), nn.GroupNorm(8, out), nn.GELU()])
            channels = out
        self.encoder = nn.Sequential(*blocks)
        self.first, self.second = ConvGRUCell(width), ConvGRUCell(width)
        self.time = nn.Parameter(.02 * torch.randn(cfg.frames, width, 1, 1))
        self.decoder = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, 3 * 16**2, 1), nn.PixelShuffle(16),
                                     nn.Sigmoid())

    def forward(self, clip, visible):
        _check(clip, visible, self.cfg)
        b, t, _, h, w = clip.shape
        mask = visible[:, :, None, None, None]
        inp = torch.cat((clip.masked_fill(~mask, 0), mask.expand(b, t, 1, h, w).to(clip.dtype)), 2)
        x = self.encoder(inp.flatten(0, 1)).unflatten(0, (b, t))
        h1 = torch.zeros_like(x[:, 0])
        h2 = torch.zeros_like(h1)
        for frame in x.unbind(1):
            h1 = self.first(frame, h1)
            h2 = self.second(h1, h2)
        slots = h2[:, None] + self.time.to(h2.dtype)
        return self.decoder(slots.flatten(0, 1)).unflatten(0, (b, t))

    def objective(self, clip, visible):
        pred = self(clip, visible)
        return F.mse_loss(pred[~visible].float(), clip[~visible])

    @torch.no_grad()
    def validation_pair(self, clip, visible):
        return self(clip, visible), clip


def _blocks(cfg, depth):
    return nn.Sequential(*(nn.TransformerEncoderLayer(
        cfg.dim, cfg.heads, dim_feedforward=4 * cfg.dim, dropout=0., activation="gelu",
        batch_first=True, norm_first=True) for _ in range(depth)), nn.LayerNorm(cfg.dim))


class PatchEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch = nn.Conv2d(3, cfg.dim, 16, 16)
        self.position = nn.Parameter(.02 * torch.randn(1, (cfg.side // 16)**2, cfg.dim))
        self.blocks = _blocks(cfg, cfg.spatial_depth)

    def forward(self, frames):
        patches = self.patch(frames).flatten(2).transpose(1, 2)
        return self.blocks(patches + self.position.to(patches.dtype))


class VideoJEPA(nn.Module):
    """Attend over space within visible frames, then over time at each spatial patch."""
    def __init__(self, cfg: VideoConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(cfg)
        self.target = deepcopy(self.encoder).requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(cfg.dim))
        self.time = nn.Parameter(.02 * torch.randn(cfg.frames, 1, cfg.dim))
        self.temporal = _blocks(cfg, cfg.temporal_depth)
        self.predictor = nn.Linear(cfg.dim, cfg.dim)

    def _predict(self, clip, visible):
        _check(clip, visible, self.cfg)
        b, t = visible.shape
        x = self.encoder(clip[visible])
        _, p, d = x.shape
        slots = self.mask_token.to(x.dtype).expand(b, t, p, d).clone()
        slots[visible] = x
        slots = slots + self.time.to(x.dtype)
        # B*patch independent temporal sequences; spatial context is already in each token.
        z = slots.transpose(1, 2).reshape(b * p, t, d)
        pred = self.predictor(self.temporal(z)).reshape(b, p, t, d).transpose(1, 2)
        return pred, x

    def forward(self, clip, visible):
        return self._predict(clip, visible)[0]

    def objective(self, clip, visible):
        pred, x = self._predict(clip, visible)
        with torch.no_grad():
            target = self.target(clip[~visible])
        loss = F.mse_loss(pred[~visible].float(), target.float())
        # Variation across visible images at each fixed patch, not across patch coordinates.
        std = (x.float().var(dim=0, unbiased=False) + 1e-4).sqrt()
        return loss + F.relu(0.8 - std).mean()

    @torch.no_grad()
    def update_target(self, decay=0.99):
        for target, online in zip(self.target.parameters(), self.encoder.parameters()):
            target.lerp_(online, 1 - decay)

    @torch.no_grad()
    def validation_pair(self, clip, visible):
        target = self.target(clip.flatten(0, 1)).unflatten(0, clip.shape[:2])
        return self(clip, visible), target
