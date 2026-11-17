import numpy as np
import pytest
import torch

from illusion_rnn import apparent as ap


def _batch(n=40, seed=0, **kwargs):
    return ap.sample_motion(np.random.default_rng(seed), n, **kwargs)


def test_sample_motion_shapes_and_ink():
    batch = _batch(n_frames=8, img_size=32)
    assert batch.frames.shape == (40, 8, 32, 32)
    assert batch.centres.shape == (40, 8, 2)
    assert set(np.unique(batch.frames)) <= {0.0, 1.0}
    # the shape is fully on the canvas in every frame: ink never changes
    ink = batch.frames.sum(axis=(2, 3))
    assert np.all(ink == ink[:, :1])


def test_centres_match_rendered_ink_and_move_linearly():
    batch = _batch(p_static=0.0)
    yy, xx = np.mgrid[0:32, 0:32]
    ink = batch.frames.sum(axis=(2, 3))
    cx = (batch.frames * xx).sum(axis=(2, 3)) / ink
    cy = (batch.frames * yy).sum(axis=(2, 3)) / ink
    np.testing.assert_allclose(np.stack([cx, cy], axis=-1), batch.centres, atol=1e-4)
    # constant velocity up to pixel rounding
    fractions = np.linspace(0, 1, 8)[None, :, None]
    ideal = batch.centres[:, :1] + fractions * (batch.centres[:, -1:] - batch.centres[:, :1])
    assert np.abs(batch.centres - ideal).max() <= 1.0


def test_disp_range_and_static():
    moving = _batch(p_static=0.0, disp_range=(0.25, 0.75), img_size=32)
    dist = np.linalg.norm(moving.centres[:, -1] - moving.centres[:, 0], axis=1)
    assert dist.min() >= 0.25 * 32 - 1.5 and dist.max() <= 0.75 * 32 + 1.5
    static = _batch(p_static=1.0)
    assert np.all(static.frames == static.frames[:, :1])


def test_infeasible_geometry_raises_instead_of_hanging():
    with pytest.raises(ValueError, match="cannot move"):
        _batch(img_size=10, size_range=(8, 8), p_static=0.0)


def test_perfect_triangle_has_zero_position_error():
    batch = _batch(n=30, shapes=("triangle",), p_static=0.0, disp_range=(0.6, 0.75))
    visible = ap.endpoint_mask(30, 8)
    assert ap.imputation_metrics(batch.frames, batch, visible)["pos_err"] == pytest.approx(
        0.0, abs=1e-4)
    assert ap.patch_position_metrics(batch.frames, batch, visible)["by_frame"][6][
        "patch_err"] == pytest.approx(0.0, abs=1e-4)


def test_unknown_shape_raises():
    with pytest.raises(ValueError, match="Unknown shape"):
        _batch(shapes=("blob",))


@pytest.mark.parametrize("max_gap", [1, 3, None])
def test_sample_masks_constraints(max_gap):
    visible = ap.sample_masks(np.random.default_rng(0), 2000, 8, max_gap=max_gap)
    assert visible.shape == (2000, 8) and visible.dtype == bool
    assert visible.sum(axis=1).min() >= 2
    runs = [ap._longest_run(~row)[1] for row in visible]
    if max_gap is not None:
        assert max(runs) <= max_gap
    else:
        assert max(runs) > 3
    assert (~visible).any()


def test_endpoint_and_gap_masks():
    visible = ap.endpoint_mask(3, 8)
    assert visible[:, [0, -1]].all() and not visible[:, 1:-1].any()
    np.testing.assert_array_equal(ap.gap_mask(3, 8, 6), visible)
    assert ap.gap_mask(1, 8, 2)[0].tolist() == [True] * 3 + [False] * 2 + [True] * 3
    with pytest.raises(ValueError):
        ap.gap_mask(1, 8, 7)


def test_apply_mask_blanks_only_hidden_frames():
    batch = _batch(n=4)
    visible = ap.endpoint_mask(4, 8)
    masked = ap.apply_mask(batch.frames, visible)
    assert not masked[:, 1:-1].any()
    np.testing.assert_array_equal(masked[:, [0, -1]], batch.frames[:, [0, -1]])


@pytest.mark.parametrize("cell", ap.CELLS)
@pytest.mark.parametrize("readout", ap.READOUTS)
def test_imputation_net_contract(cell, readout):
    model = ap.ImputationNet(16, 12, n_frames=5, cell=cell, readout=readout)
    out, activity = model(torch.randn(5, 3, 16))
    assert out.shape == (5, 3, 16)
    assert activity.shape == (5, 3, 12)
    with pytest.raises(ValueError, match="timesteps"):
        model(torch.randn(4, 3, 16))


def test_readout_causality():
    """Only the ``final`` readout lets a late frame change an early output."""
    torch.manual_seed(0)
    x = torch.randn(5, 2, 16)
    late = x.clone()
    late[-1] += 1.0
    for readout, depends in (("online", False), ("final", True)):
        model = ap.ImputationNet(16, 12, n_frames=5, readout=readout)
        changed = not torch.allclose(model(x)[0][:-1], model(late)[0][:-1])
        assert changed is depends


def test_metrics_separate_fill_in_from_jump_and_superposition():
    batch = _batch(n=60, shapes=("square",), p_static=0.0)
    visible = ap.endpoint_mask(60, 8)

    filled = ap.imputation_metrics(batch.frames, batch, visible)
    assert filled["n_blanked"] == 60 * 6
    assert filled["iou_true"] == 1.0
    assert filled["path_slope"] == pytest.approx(1.0, abs=1e-6)
    assert filled["pos_err"] == pytest.approx(0.0, abs=1e-4)
    assert filled["ink_ratio"] == pytest.approx(1.0)

    jump = batch.frames.copy()
    jump[:, 1:4] = batch.frames[:, :1]
    jump[:, 4:7] = batch.frames[:, -1:]
    jumped = ap.imputation_metrics(jump, batch, visible)
    assert jumped["iou_nearest"] == 1.0
    assert jumped["iou_true"] < 0.5

    both = np.maximum(batch.frames[:, :1], batch.frames[:, -1:])
    superposed = ap.imputation_metrics(np.repeat(both, 8, axis=1), batch, visible)
    assert superposed["iou_union"] == 1.0
    assert superposed["path_slope"] == pytest.approx(0.0, abs=1e-6)


def test_metrics_handle_blank_reconstruction():
    batch = _batch(n=5, p_static=0.0)
    result = ap.imputation_metrics(np.zeros_like(batch.frames), batch, ap.endpoint_mask(5, 8))
    assert result["iou_true"] == 0.0 and result["ink_ratio"] == 0.0
    assert result["path_slope"] is None and result["pos_err"] is None


def test_training_reduces_loss_and_evaluates():
    torch.manual_seed(0)
    model = ap.ImputationNet(16 * 16, 32, n_frames=5, cell="gru")
    kwargs = {"img_size": 16, "size_range": (3, 4)}
    history = ap.train_imputation(model, n_steps=60, batch_size=16, max_gap=2,
                                  device="cpu", log_every=0, **kwargs)
    assert np.mean(history["loss"][-10:]) < np.mean(history["loss"][:10])
    for condition in ("apparent", "flicker"):
        result = ap.evaluate_imputation(model, condition, n_trials=8, device="cpu", **kwargs)
        assert result["n_blanked"] == 8 * 3
    assert ap.evaluate_imputation(model, "gap", gap=1, n_trials=8, device="cpu",
                                  **kwargs)["n_blanked"] == 8
    with pytest.raises(ValueError, match="Unknown condition"):
        ap.evaluate_imputation(model, "nope", device="cpu")


def test_patch_position_perfect_reconstruction_is_at_the_floor():
    batch = _batch(n=60, shapes=("square",), p_static=0.0, disp_range=(0.6, 0.75))
    visible = ap.endpoint_mask(60, 8)
    result = ap.patch_position_metrics(batch.frames, batch, visible)
    assert result["patch_err"] == pytest.approx(result["truth_err"])
    assert result["patch_err"] < result["jump_err"] < result["stay_err"]
    # far from the start nothing is clipped, so the centre is exact
    assert result["by_frame"][6]["patch_err"] == pytest.approx(0.0, abs=1e-4)


def test_patch_position_ignores_ink_held_at_start():
    batch = _batch(n=30, shapes=("square",), p_static=0.0, disp_range=(0.6, 0.75))
    visible = ap.endpoint_mask(30, 8)
    held = np.repeat(batch.frames[:, :1], 8, axis=1)
    assert ap.patch_position_metrics(held, batch, visible)["patch_found"] == 0.0
    # faint, misshapen ink at the right place still scores as the right place
    faint = np.maximum(0.3 * batch.frames, held)
    result = ap.patch_position_metrics(faint, batch, visible)
    assert result["by_frame"][6]["patch_found"] == 1.0
    assert result["by_frame"][6]["patch_err"] == pytest.approx(0.0, abs=1e-4)


def test_patch_position_jump_scores_as_jump_and_static_is_skipped():
    batch = _batch(n=30, shapes=("square",), p_static=0.0, disp_range=(0.6, 0.75))
    visible = ap.endpoint_mask(30, 8)
    jump = np.repeat(batch.frames[:, -1:], 8, axis=1)
    result = ap.patch_position_metrics(jump, batch, visible)
    assert result["by_frame"][6]["patch_err"] == pytest.approx(
        result["by_frame"][6]["jump_err"], abs=1e-4)
    assert result["by_frame"][2]["patch_err"] > result["by_frame"][2]["jump_err"]
    static = _batch(n=5, p_static=1.0)
    assert ap.patch_position_metrics(static.frames, static, ap.endpoint_mask(5, 8))[
        "patch_found"] is None


def test_largest_patch_picks_heaviest_region():
    weights = np.zeros((8, 8))
    weights[0, 0] = 0.9
    weights[4:6, 4:6] = 0.5
    np.testing.assert_allclose(ap._largest_patch(weights, 0.25), [4.5, 4.5])
    assert ap._largest_patch(weights, 0.95) is None
