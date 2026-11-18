import json

import numpy as np
import pytest
import torch

from illusion_rnn.apparent import ImputationNet, endpoint_mask
from illusion_rnn.jepa import MotionJEPA
from scripts import trace_position as tr


def test_only_one_flash_moves_exactly_four_pixels_without_wrap():
    a, m, d, c = tr.pairs(16, 81, ("triangle",))
    assert np.unique(d.numpy(), axis=0, return_counts=True)[1].tolist() == [4]*4
    for end in range(2):
        b = tr.donor(a, m, end)
        assert torch.equal(a[:, 1-end], b[:, 1-end])
        for i, mask in enumerate(b[:, end]):
            yx = torch.nonzero(mask).float().flip((1,))
            assert len(yx) == int(a[i, end].sum())
            torch.testing.assert_close(yx.mean(0), c[i, end]+d[i])
        assert tr.input_movie(b)[:, 1:-1].count_nonzero() == 0


@torch.no_grad()
@pytest.mark.parametrize("name", ["rnn", "jepa"])
def test_whole_site_patches_reproduce_real_counterfactual(name):
    x, m, _, _ = tr.pairs(4, 1, ("circle",))
    model = MotionJEPA(8, 16) if name == "jepa" else ImputationNet(32**2, 16, 8)
    for e in range(2):
        b = tr.donor(x, m, e)
        u, v = (tr.states(model, xx, e) for xx in (x, b))
        mov = tr.input_movie(b)
        if name == "jepa":
            target, _, _ = model(mov, torch.from_numpy(endpoint_mask(len(b), 8)))
            keys = ["slot", "mix", "final"]
            src, dst = torch.arange(8), torch.arange(16)
            actual = tr.edge_state(model, u, v, e, src, dst)
            torch.testing.assert_close(actual, target)
        else:
            _, activity = model(mov.flatten(2).transpose(0, 1))
            target = activity[-1]
            keys = ["flash", "gap", "final"] if e == 0 else ["final"]
            if e == 0:
                actual = tr.edge_state(model, u, v, e, torch.arange(16), torch.arange(16))
                leak = (1-model.rnn.alpha)*(v["gap"]-u["gap"])
                torch.testing.assert_close(actual+leak, target)
            else:
                assert torch.equal(u["gap"], v["gap"])
        torch.testing.assert_close(v["final"], target)
        for key in keys:
            torch.testing.assert_close(tr.resume(model, u, key, v[key], e), target)
            torch.testing.assert_close(tr.resume(model, u, key, u[key], e), u["final"])


def test_translation_basis_preserves_planted_signal():
    step = torch.tensor([[4., 0.], [-4., 0.], [0., 4.], [0., -4.]])
    diff = torch.zeros(4, 12)
    diff[:, (5, 9)] = step
    q, rank = tr.basis(diff, step)
    torch.testing.assert_close((diff @ q) @ q.T, diff)
    assert set(rank[:2].tolist()) == {5, 9}


@torch.no_grad()
@pytest.mark.parametrize("name", ["rnn", "jepa"])
def test_single_edge_edit_changes_only_its_contribution_before_nonlinearity(name):
    a, moved, _, _ = tr.pairs(8, 72, ("square",))
    model = MotionJEPA(10, 12) if name == "jepa" else ImputationNet(32**2, 12, 8)
    u, v = (tr.states(model, x, 0) for x in (a, tr.donor(a, moved, 0)))
    pre, dx, w = tr.route_inputs(model, u, v, 0)
    src, dst = 3, 4
    masked = tr.mask_weights(w, torch.tensor([dst*w.size(1)+src]), 1)
    assert masked.count_nonzero() == 1
    manually_edited = pre.clone()
    manually_edited[:, dst] += w[dst, src]*dx[:, src]
    actual = tr.route_finish(model, u, pre+torch.nn.functional.linear(dx, masked))
    torch.testing.assert_close(actual, tr.route_finish(model, u, manually_edited))
    torch.testing.assert_close(actual, tr.edge_state(
        model, u, v, 0, torch.tensor([src]), torch.tensor([dst])))


@torch.no_grad()
def test_trace_reports_native_output_and_directional_controls():
    fit = tr.pairs(8, 32, ("square",))
    test = tr.pairs(4, 33, ("triangle",))
    result = tr.evaluate(MotionJEPA(10, 12), fit, test, end=1, seed=0)
    assert result["rows"]["full"]["native_recovery"] == 1
    assert len(result["edge_units"]["slot"]) == tr.K
    assert len(set(map(tuple, result["ranked_edges_source_target"]))) == 64
    assert len(result["rows"]["final/random3"]["by_hidden_frame_px"]) == 6
    # The JEPA head and the evaluation readout are affine after the final state.
    r = result["rows"]
    assert r["route/all"]["xy_recovery"] == pytest.approx(1, abs=1e-5)
    assert abs(r["block/all"]["shift_px"]) < 1e-5
    assert r["full"]["shift_px"] == pytest.approx(
        r["final/position"]["shift_px"]+r["final/blocked"]["shift_px"], abs=1e-5)
    json.dumps(result, allow_nan=False)
