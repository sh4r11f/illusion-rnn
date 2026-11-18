import numpy as np
import pytest
import torch

from illusion_rnn.apparent import ImputationNet
from illusion_rnn.jepa import MotionJEPA
from scripts import profile_motion_units as p


def test_split_geometry_and_paired_targets():
    seen = set()
    fit = p.movies(24, 4, ("square", "circle"), seen)
    test = p.movies(24, 4, ("cross",), seen)
    assert len(seen) == 48
    for d in (fit, test):
        assert torch.equal(d.flashes[::2, (0, -1)], d.flashes[1::2, (0, -1)])
        np.testing.assert_allclose(d.targets["bend"][::2], -d.targets["bend"][1::2], atol=1e-6)
        np.testing.assert_array_equal(d.targets["start"][::2], d.targets["start"][1::2])
        assert ((d.normals.numpy()*d.targets["bend"][1::2]).sum(1) > 0).all()


@torch.no_grad()
@pytest.mark.parametrize("m", ["rnn", "jepa"])
def test_progressive_reveal_and_final_sites(m):
    model = MotionJEPA(10, 20) if m == "jepa" else ImputationNet(32**2, 20, 8)
    data = p.movies(8, 912, ("cross",), set())
    h, site, no = p.activities(model, data.flashes)
    np.testing.assert_allclose(h[::2, :3], h[1::2, :3], atol=3e-6)
    np.testing.assert_allclose(no[::2], no[1::2], atol=3e-6)
    altered = data.flashes.clone()
    altered[:, -1] = torch.roll(altered[:, -1], 3, dims=-1)
    np.testing.assert_array_equal(p.activities(model, altered)[0][:, :-1], h[:, :-1])
    if m == "jepa":
        np.testing.assert_array_equal(h[:, 3], h[:, 6])
        np.testing.assert_allclose(model.context[3:](torch.from_numpy(site)).numpy(),
                                   h[:, -1], atol=1e-6)
    else:
        np.testing.assert_array_equal(h[:, -1], site)


def test_unit_rank_uses_only_fitting_examples_and_handles_silent_units():
    rng = np.random.default_rng(222)
    x = rng.normal(size=(100, 8))
    x[:, -1] = 0
    y = x[:, :1]*[[2., -1.]]
    fit, score, order = p.unit_fits(x, y, x, y)
    np.testing.assert_allclose([fit[0], score[0], score[-1]], [1, 1, 0], atol=1e-10)
    assert order[0] == 0
    np.testing.assert_array_equal(order, p.unit_fits(x, y, x, y[::-1])[2])


def test_time_blind_readout_cannot_move_with_frozen_state():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(80, 1, 2))
    h = np.repeat(v, 8, axis=1)
    target = v * np.arange(1, 9)[None, :, None]
    m = p.blank_motion(h, h, target, target)
    assert m["time_blind"]["projected_travel"] == [0.]*3
    assert m["frozen_cue_plus_clock"]["projected_travel"][-1] > .95
    np.testing.assert_allclose(m["true"]["projected_travel"], [0, .5, 1])


@torch.no_grad()
def test_whole_population_swap_and_block_are_intact_effect():
    model = ImputationNet(32**2, 16, 8)
    data = p.movies(16, 812, ("square",), set())
    _, x, _ = p.activities(model, data.flashes)
    rng = np.random.default_rng(11)
    out = p.swaps(model, x, np.arange(16), data.normals, None, rng)
    all_units = out["sizes"][-1]
    assert all_units["copied_px"] == pytest.approx(out["intact_px"])
    assert all_units["removed_px"] == pytest.approx(out["intact_px"])
