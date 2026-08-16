"""Supervised training and evaluation for TAM/motion tasks.

The pipeline is the neurogym supervised one: ``ngym.Dataset`` yields
``(inputs (T, B, H, W), labels (T, B))`` batches of concatenated trials;
the model is trained with cross-entropy over every timestep. An optional
``encoder`` callable maps raw image batches ``(T, B, H, W)`` to feature
batches ``(T, B, F)`` before they reach the model — pass
``cnn_encoder(ShapesCNN(...), ...)`` to reproduce the CNN-features
pipeline the ``rnn-cnnfeat64_*`` checkpoints were trained with.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

import neurogym as ngym

from illusion_rnn.envs import MotionTask, TAMCorrespondenceTask, TAMTask


def resolve_device(device=None) -> torch.device:
    """Explicit argument > cuda > mps > cpu."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(task: str, **kwargs):
    """Construct a task env: ``task`` is ``"tam"`` (TAMTask), ``"motion"``
    (MotionTask), or ``"correspondence"`` (TAMCorrespondenceTask); ``kwargs``
    pass through to the constructor."""
    if task == "tam":
        return TAMTask(**kwargs)
    if task == "motion":
        return MotionTask(**kwargs)
    if task == "correspondence":
        return TAMCorrespondenceTask(**kwargs)
    msg = f"Unknown task {task!r}; expected 'tam', 'motion' or 'correspondence'"
    raise ValueError(msg)


def make_dataset(env, batch_size: int = 16, seq_len: int = 100) -> ngym.Dataset:
    """Wrap an env in a neurogym supervised Dataset (env is deep-copied)."""
    return ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)


def _prepare_inputs(inputs: np.ndarray, encoder) -> np.ndarray:
    if encoder is not None:
        return encoder(inputs)
    return inputs.reshape(*inputs.shape[:2], -1)


def train(
    model,
    datasets,
    n_epochs: int = 1000,
    lr: float = 5e-4,
    device=None,
    encoder=None,
    log_every: int = 100,
) -> dict:
    """Train ``model`` on one or more ``ngym.Dataset``s; returns per-epoch
    ``{"loss": [...], "accuracy": [...]}``."""
    device = resolve_device(device)
    model = model.to(device)
    model.train()
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": []}

    for epoch in range(n_epochs):
        batches = [dataset() for dataset in datasets]
        inputs = np.concatenate([b[0] for b in batches], axis=1)
        labels = np.concatenate([b[1] for b in batches], axis=1).flatten()

        x = torch.from_numpy(_prepare_inputs(inputs, encoder)).float().to(device)
        y = torch.from_numpy(labels).long().to(device)

        optimizer.zero_grad()
        out, _ = model(x)
        # `.reshape` rather than `.view`: FrameOnlyNet and FFStack return
        # `out.permute(1, 0, 2)`, which is non-contiguous, and `.view` requires
        # contiguity. `.reshape` is a strict superset (falls back to a copy
        # only when a view isn't possible) so this is a no-op for RNNNet/
        # GRUNet/CTRNN, whose outputs are already contiguous.
        out = out.reshape(-1, out.shape[-1])
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        accuracy = (out.argmax(dim=1) == y).float().mean().item()
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)
        if log_every and (epoch + 1) % log_every == 0:
            print(
                f"epoch {epoch + 1}/{n_epochs}  "
                f"loss {loss.item():.4f}  acc {accuracy:.3f}",
            )
    return history


def summarize_trials(trials: list, n_actions: int, abstain_action: int = 0):
    """Reduce per-trial records to (accuracy, abstention_rate, committed_accuracy,
    confusion).

    `abstain_action` is the fixation class. Ground truth is never fixation at the
    decision step, so choosing it is a refusal to commit rather than a wrong
    answer -- the two are reported separately because collapsing them hides the
    difference between "guessed wrong" and "never left the null state".

    `committed_accuracy` is NaN (not 0.0) when nothing was committed, so an
    all-abstain run cannot be misread as an all-wrong run.
    """
    n = len(trials)
    correct = sum(bool(t["correct"]) for t in trials)
    abstained = sum(t["choice"] == abstain_action for t in trials)
    committed_n = n - abstained

    confusion = np.zeros((n_actions, n_actions), dtype=int)
    for t in trials:
        confusion[t["ground_truth"], t["choice"]] += 1

    accuracy = correct / n if n else float("nan")
    abstention_rate = abstained / n if n else float("nan")
    committed_accuracy = correct / committed_n if committed_n else float("nan")
    return accuracy, abstention_rate, committed_accuracy, confusion


@dataclass
class EvalResult:
    """Result of ``evaluate``.

    ``accuracy`` counts abstentions as errors (the historical definition, kept
    so existing numbers stay comparable). ``abstention_rate`` and
    ``committed_accuracy`` separate the two failure modes; ``confusion`` is
    indexed ``[ground_truth, choice]``.
    """

    accuracy: float
    abstention_rate: float = float("nan")
    committed_accuracy: float = float("nan")
    confusion: np.ndarray | None = None
    trials: list = field(default_factory=list)
    activity: list = field(default_factory=list)


def evaluate(model, env, n_trials: int = 100, device=None, encoder=None) -> EvalResult:
    """Run ``n_trials`` single trials through ``model``; choice is the argmax
    of the final timestep's output."""
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    env.reset()

    trials, activity = [], []
    with torch.no_grad():
        for _ in range(n_trials):
            env.new_trial()
            ob, gt = env.ob, env.gt
            inputs = _prepare_inputs(ob[:, np.newaxis], encoder)
            x = torch.from_numpy(inputs).float().to(device)
            pred, hidden = model(x)
            choice = int(pred[-1, 0].argmax().item())
            ground_truth = int(gt[-1])
            trials.append(
                {
                    "ground_truth": ground_truth,
                    "choice": choice,
                    "correct": choice == ground_truth,
                },
            )
            activity.append(hidden[:, 0].cpu().numpy())

    accuracy, abstention_rate, committed_accuracy, confusion = summarize_trials(
        trials, n_actions=env.action_space.n,
        abstain_action=env.choice_names["fixation"],
    )
    return EvalResult(
        accuracy=accuracy,
        abstention_rate=abstention_rate,
        committed_accuracy=committed_accuracy,
        confusion=confusion,
        trials=trials,
        activity=activity,
    )


def cnn_encoder(cnn, device=None):
    """Batched encoder: runs every frame of a ``(T, B, H, W)`` batch through
    ``cnn`` in one forward pass, returning ``(T, B, feature_dim)``."""
    device = resolve_device(device)
    cnn = cnn.to(device)
    cnn.eval()

    def encode(inputs: np.ndarray) -> np.ndarray:
        n_steps, n_batch, height, width = inputs.shape
        x = torch.from_numpy(
            inputs.reshape(n_steps * n_batch, 1, height, width),
        ).float().to(device)
        with torch.no_grad():
            _, features = cnn(x)
        return features.cpu().numpy().reshape(n_steps, n_batch, -1)

    return encode
