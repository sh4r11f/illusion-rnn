import torch

from illusion_rnn.models import CTRNN, RNNNet, ShapesCNN, load_rnn


def test_ctrnn_alpha():
    assert CTRNN(4, 8, dt=50).alpha == 0.5
    assert CTRNN(4, 8).alpha == 1


def test_rnnnet_shapes():
    model = RNNNet(input_size=64, hidden_size=32, output_size=6, dt=50)
    x = torch.randn(9, 4, 64)
    out, activity = model(x)
    assert out.shape == (9, 4, 6)
    assert activity.shape == (9, 4, 32)


def test_rnnnet_state_dict_layout():
    # Frozen layout: shipped checkpoints use these exact key names.
    keys = set(RNNNet(64, 32, 6, dt=50).state_dict())
    assert keys == {
        "rnn.input2h.weight",
        "rnn.input2h.bias",
        "rnn.h2h.weight",
        "rnn.h2h.bias",
        "fc.weight",
        "fc.bias",
    }


def test_shapes_cnn_flat_dim():
    assert ShapesCNN._flat_dim(100) == 10368


def test_shapes_cnn_shapes():
    model = ShapesCNN()
    x = torch.randn(2, 1, 100, 100)
    logits, features = model(x)
    assert logits.shape == (2, 9)
    assert features.shape == (2, 64)


def test_shapes_cnn_small_input():
    model = ShapesCNN(input_size=32, feature_dim=16)
    logits, features = model(torch.randn(3, 1, 32, 32))
    assert logits.shape == (3, 9)
    assert features.shape == (3, 16)


def test_load_rnn_roundtrip(tmp_path):
    model = RNNNet(16, 8, 6, dt=50)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_rnn(path)
    assert not loaded.training  # eval mode
    assert loaded.rnn.input2h.in_features == 16
    assert loaded.rnn.h2h.in_features == 8
    assert loaded.fc.out_features == 6
