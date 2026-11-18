import json

import numpy as np
import pytest
import torch

from illusion_rnn.jepa import MotionJEPA, PixelControl
from scripts import evaluate_motion_jepa as exp
from scripts import run_stimulus_robustness as suite


@pytest.mark.parametrize("kind", [MotionJEPA, PixelControl])
def test_masked_images_never_enter_context_or_predictions(kind):
    model = kind(12, 24)
    x = torch.randn(2, 8, 32, 32)
    visible = torch.tensor([[True, False]*4, [False, True]*4])
    h, z, enc = model(x, visible)
    poison = x.masked_fill(~visible[..., None, None], torch.nan)
    for a, b in zip((h, z, enc), model(poison, visible)):
        torch.testing.assert_close(a, b)
    assert h.shape == (2, 24) and z.shape == (2, 8, 12)
    x2 = x.masked_fill(visible[..., None, None], 0)
    assert not torch.equal(h, model(x2, visible)[0])
    with pytest.raises(ValueError, match="both visible and masked"):
        model.objective(x, torch.ones_like(visible))


def test_target_receives_only_ema_not_gradients_and_encodes_one_frame_at_a_time():
    model = MotionJEPA(12, 24)
    x = torch.randn(4, 8, 32, 32)
    visible = torch.ones(4, 8, dtype=torch.bool)
    visible[:, 2:5] = False
    with torch.no_grad():
        before = [p.clone() for p in model.target.parameters()]
        one_frame = model.target(x[:, 3:4])
        torch.testing.assert_close(one_frame, model.target(x)[:, 3:4])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.objective(x, visible).backward()
    assert all(p.grad is None for p in model.target.parameters())
    assert all(p.grad is not None for p in model.encoder.parameters())
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, model.target.parameters()))
    model.update_target(.9)
    for old, ema, live in zip(before, model.target.parameters(), model.encoder.parameters()):
        torch.testing.assert_close(ema, .9*old + .1*live)


def test_future_not_required_even_to_construct_features():
    model = MotionJEPA(6, 8)
    frames = np.random.default_rng(6).random((3, 8, 32, 32), dtype=np.float32)
    expected = exp.features(model, frames[:, :4], exp.PREFIX)
    frames[:, 4:] = np.nan
    for before, after in zip(expected, exp.features(model, frames, exp.PREFIX)):
        np.testing.assert_array_equal(before, after)


def test_pixel_readout_reports_xy_not_yx():
    model = PixelControl(6, 8)
    with torch.no_grad():
        model.pixels.weight.zero_()
        model.pixels.bias.fill_(-100.)
        model.pixels.bias[5*32 + 17] = 100.
    key, xy = exp.native_position(model, np.zeros((4, 3, 6), dtype=np.float32), None)
    assert key == "ink_centre"
    np.testing.assert_allclose(xy, np.broadcast_to([17, 5], xy.shape), atol=1e-6)


@pytest.mark.parametrize("kind", [MotionJEPA, PixelControl])
def test_frozen_probes_score_all_displays_and_chance_controls(kind):
    ep, forecasts = suite.geometry(4)
    excluded, last = suite.excluded_pairs(ep, forecasts)
    fit = exp.fore.dataset(4, exp.fore.FIT_SEED, last)
    model = kind(6, 8)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    result = exp.score(model, ep, forecasts, 12, excluded, fit)
    json.dumps(result, allow_nan=False)
    assert len(result["conditions"]) == 9
    for row in result["conditions"]:
        modes = row["interpolation"]["endpoints"]
        assert ("predicted_embedding" in modes) == (kind == MotionJEPA)
        for method in modes.values():
            assert method["accel_choice"] == pytest.approx(1/3)
            assert method["path_choice"] == pytest.approx(1/3)
        control = row["forecast"]["last_velocity"]["straight"]
        assert len(control) == 4 and control[-1]["separation_gain"] == 0
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
