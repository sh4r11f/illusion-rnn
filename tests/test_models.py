import pytest
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


from illusion_rnn.models import FFStack, FrameOnlyNet, GRUNet, shuffle_frames

T, B, F, H, OUT = 9, 4, 64, 16, 6


@pytest.mark.parametrize("build", [
    lambda: GRUNet(F, H, OUT),
    lambda: FFStack(F, H, OUT, n_steps=T),
    lambda: FrameOnlyNet(F, H, OUT, n_steps=T, frame_index=3),
])
def test_models_honour_the_shared_contract(build):
    model = build()
    out, activity = model(torch.randn(T, B, F))
    assert out.shape == (T, B, OUT)
    assert activity.shape[:2] == (T, B)


def test_frame_only_net_reads_exactly_one_timestep():
    """If it can see any other timestep the baseline is not a baseline."""
    model = FrameOnlyNet(F, H, OUT, n_steps=T, frame_index=3)
    x = torch.zeros(T, B, F)
    base, _ = model(x)
    for t in range(T):
        probe = torch.zeros(T, B, F)
        probe[t] = 1.0
        out, _ = model(probe)
        changed = not torch.allclose(out, base)
        assert changed == (t == 3), f"timestep {t}: changed={changed}"


def test_ffstack_sees_every_timestep():
    model = FFStack(F, H, OUT, n_steps=T)
    x = torch.zeros(T, B, F)
    base, _ = model(x)
    for t in range(T):
        probe = torch.zeros(T, B, F)
        probe[t] = 1.0
        out, _ = model(probe)
        assert not torch.allclose(out, base), f"blind to timestep {t}"


def test_ffstack_output_varies_across_time():
    """It must be able to answer `fixation` early and a direction late, or the
    per-timestep loss punishes it for a handicap the recurrent models escape."""
    model = FFStack(F, H, OUT, n_steps=T)
    out, _ = model(torch.randn(T, B, F))
    assert not torch.allclose(out[0], out[-1])


def test_shuffle_frames_permutes_only_the_named_timesteps():
    x = torch.arange(T * 1 * 2, dtype=torch.float32).reshape(T, 1, 2)
    g = torch.Generator().manual_seed(0)
    y = shuffle_frames(x, frame_indices=(2, 3, 4, 5, 6), generator=g)
    for t in (0, 1, 7, 8):
        torch.testing.assert_close(y[t], x[t])
    assert sorted(y[i, 0, 0].item() for i in (2, 3, 4, 5, 6)) == \
           sorted(x[i, 0, 0].item() for i in (2, 3, 4, 5, 6))


def test_shuffle_frames_actually_changes_the_order_sometimes():
    x = torch.arange(T * 1 * 2, dtype=torch.float32).reshape(T, 1, 2)
    g = torch.Generator().manual_seed(0)
    permuted = [shuffle_frames(x, (2, 3, 4, 5, 6), g) for _ in range(20)]
    assert any(not torch.allclose(p, x) for p in permuted)
