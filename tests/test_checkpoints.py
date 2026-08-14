from pathlib import Path

import pytest
import torch

from illusion_rnn.models import ShapesCNN, load_rnn
from illusion_rnn.training import evaluate, make_env

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

RNN_SPECS = [
    ("rnn-pixel_h2048_tam-horiz.pt", 4096, 2048, 6),
    ("rnn-cnnfeat64_h1024_tam-horiz.pt", 64, 1024, 6),
    ("rnn-cnnfeat64_h2048_tam-horiz.pt", 64, 2048, 6),
]


def _materialized(name: str) -> bool:
    """False for missing files and un-smudged LFS pointers (~130 bytes)."""
    path = CHECKPOINT_DIR / name
    return path.exists() and path.stat().st_size > 1024


def _require(name: str) -> Path:
    if not _materialized(name):
        pytest.skip(f"{name} not materialized (run: git lfs install --local && git lfs checkout)")
    return CHECKPOINT_DIR / name


@pytest.mark.parametrize("name,input_size,hidden_size,output_size", RNN_SPECS)
def test_rnn_checkpoints_load(name, input_size, hidden_size, output_size):
    model = load_rnn(_require(name))
    assert model.rnn.input2h.in_features == input_size
    assert model.rnn.h2h.in_features == hidden_size
    assert model.fc.out_features == output_size


def test_shapes_cnn_checkpoint_loads():
    path = _require("cnn-shapes_feat64_100px.pt")
    model = ShapesCNN()  # defaults must match the checkpoint
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)  # strict — raises on any mismatch


def test_flagship_above_chance():
    model = load_rnn(_require("rnn-pixel_h2048_tam-horiz.pt"))
    env = make_env("tam", box_shape="square", variant="standard",
                   stim_ori="horizontal", img_size=64)
    env.seed(7)
    result = evaluate(model, env, n_trials=30, device="cpu")
    assert result.accuracy > 0.5  # chance is ~1/3
