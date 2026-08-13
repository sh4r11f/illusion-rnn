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
