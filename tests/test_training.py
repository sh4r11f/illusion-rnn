import numpy as np
import torch

from illusion_rnn.models import RNNNet, ShapesCNN
from illusion_rnn.training import (
    EvalResult,
    cnn_encoder,
    evaluate,
    make_dataset,
    make_env,
    resolve_device,
    train,
)


def _tiny_env():
    env = make_env(
        "tam", box_shape="square", variant="standard",
        stim_ori="horizontal", img_size=16,
    )
    env.seed(0)
    return env


def test_make_env_dispatch():
    assert type(_tiny_env()).__name__ == "TAMTask"
    motion = make_env("motion", box_shape="circle", motion_type="tracking",
                      stim_ori="horizontal", img_size=16)
    assert type(motion).__name__ == "MotionTask"
    try:
        make_env("bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_make_dataset_shapes():
    ds = make_dataset(_tiny_env(), batch_size=4, seq_len=18)
    inputs, labels = ds()
    assert inputs.shape == (18, 4, 16, 16)
    assert labels.shape == (18, 4)


def test_train_decreases_loss():
    torch.manual_seed(0)
    ds = make_dataset(_tiny_env(), batch_size=4, seq_len=18)
    model = RNNNet(input_size=16 * 16, hidden_size=32, output_size=6, dt=50)
    history = train(model, ds, n_epochs=40, lr=1e-3, device="cpu", log_every=0)
    assert len(history["loss"]) == 40
    assert len(history["accuracy"]) == 40
    assert np.mean(history["loss"][-5:]) < np.mean(history["loss"][:5])


def test_train_multi_dataset():
    torch.manual_seed(0)
    datasets = [
        make_dataset(_tiny_env(), batch_size=2, seq_len=9),
        make_dataset(_tiny_env(), batch_size=3, seq_len=9),
    ]
    model = RNNNet(16 * 16, 16, 6, dt=50)
    history = train(model, datasets, n_epochs=2, lr=1e-3, device="cpu", log_every=0)
    assert len(history["loss"]) == 2


def test_evaluate_structure():
    model = RNNNet(16 * 16, 32, 6, dt=50)
    result = evaluate(model, _tiny_env(), n_trials=5, device="cpu")
    assert isinstance(result, EvalResult)
    assert len(result.trials) == 5
    assert len(result.activity) == 5
    assert result.activity[0].shape == (9, 32)
    assert set(result.trials[0]) == {"ground_truth", "choice", "correct"}
    assert 0.0 <= result.accuracy <= 1.0


def test_encoder_hook_in_train_and_evaluate():
    def mean_encoder(inputs):  # (T, B, H, W) -> (T, B, 1)
        return inputs.mean(axis=(2, 3))[..., np.newaxis]

    torch.manual_seed(0)
    ds = make_dataset(_tiny_env(), batch_size=2, seq_len=9)
    model = RNNNet(input_size=1, hidden_size=8, output_size=6, dt=50)
    history = train(model, ds, n_epochs=2, lr=1e-3, device="cpu",
                    encoder=mean_encoder, log_every=0)
    assert len(history["loss"]) == 2
    result = evaluate(model, _tiny_env(), n_trials=3, device="cpu",
                      encoder=mean_encoder)
    assert result.activity[0].shape == (9, 8)


def test_cnn_encoder_shapes():
    # NOTE: 32 is the smallest input ShapesCNN's conv stack supports.
    cnn = ShapesCNN(input_size=32, feature_dim=12)
    encoder = cnn_encoder(cnn, device="cpu")
    out = encoder(np.random.rand(3, 2, 32, 32).astype(np.float32))
    assert out.shape == (3, 2, 12)


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"
