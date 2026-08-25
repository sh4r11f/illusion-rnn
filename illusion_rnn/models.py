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


class GRUNet(nn.Module):
    """GRU with a linear readout -- a recurrence-type control for ``RNNNet``.

    Same contract as ``RNNNet``: ``(T, B, input_size)`` in,
    ``(out (T, B, output_size), activity (T, B, hidden_size))`` out.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        activity, _ = self.rnn(x)
        return self.fc(activity), activity


class FFStack(nn.Module):
    """Feedforward MLP over every timestep concatenated -- no recurrence.

    Isolates whether *recurrence* matters or merely *access to both frames*:
    this model sees the whole trial at once but has no state.

    It emits a distinct prediction per timestep (the MLP maps to
    ``n_steps * output_size``) so it can answer ``fixation`` early and a
    direction late. Broadcasting one prediction across all timesteps would
    handicap it against the recurrent models under the per-timestep loss and
    make the comparison unfair.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size * n_steps, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, n_steps * output_size)

    def forward(self, x: torch.Tensor):
        n_steps, batch, _ = x.shape
        flat = x.permute(1, 0, 2).reshape(batch, -1)
        hidden = self.net(flat)
        out = self.fc(hidden).reshape(batch, self.n_steps, self.output_size)
        activity = hidden.unsqueeze(0).expand(n_steps, batch, hidden.shape[-1])
        return out.permute(1, 0, 2), activity


class FrameOnlyNet(nn.Module):
    """Feedforward MLP that sees exactly ONE timestep of the trial.

    ``frame_index=env.frame2_indices[0]`` gives the frame-2-only baseline, whose
    ceiling on the balanced family is provably 50%.
    ``frame_index=env.frame1_index`` gives the frame-1-only baseline.

    The single-timestep restriction is enforced by indexing, not by masking, so
    there is no path by which other timesteps can reach the output --
    ``test_frame_only_net_reads_exactly_one_timestep`` verifies it empirically.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 n_steps: int, frame_index: int):
        super().__init__()
        if not 0 <= frame_index < n_steps:
            msg = f"frame_index {frame_index} out of range for {n_steps} steps"
            raise ValueError(msg)
        self.n_steps = n_steps
        self.frame_index = frame_index
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, n_steps * output_size)

    def forward(self, x: torch.Tensor):
        n_steps, batch, _ = x.shape
        # ``out`` is reshaped using ``self.n_steps`` (fixed at construction)
        # while ``activity`` would be expanded using the actual ``x.shape[0]``
        # below -- if the caller passes a trial with a different T, those two
        # returned tensors would silently disagree on their T dimension. Fail
        # loudly instead, matching FFStack's fixed-size first layer, which
        # already raises on a T mismatch.
        if n_steps != self.n_steps:
            msg = f"expected {self.n_steps} timesteps, got {n_steps}"
            raise ValueError(msg)
        hidden = self.net(x[self.frame_index])
        out = self.fc(hidden).reshape(batch, self.n_steps, self.output_size)
        activity = hidden.unsqueeze(0).expand(n_steps, batch, hidden.shape[-1])
        return out.permute(1, 0, 2), activity


def shuffle_frames(x: torch.Tensor, frame_indices, generator=None) -> torch.Tensor:
    """Randomly permute the named timesteps of ``x`` independently per trial.

    Used for the order control. On the ``growth+shrink`` condition this caps
    accuracy at 50%, because the same frame pair appears in both orders with
    opposite labels. On the growth-only condition it does NOT -- frame 1 is a
    shape and frame 2 is a bar, so a model can tell them apart by content and
    order carries nothing extra.
    """
    idx = list(frame_indices)
    out = x.clone()
    batch = x.shape[1]
    for b in range(batch):
        perm = torch.randperm(len(idx), generator=generator)
        out[idx, b] = x[[idx[p] for p in perm], b]
    return out
