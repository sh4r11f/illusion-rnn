import numpy as np
import pytest
import torch

from illusion_rnn.apparent import ImputationNet
from illusion_rnn.jepa import MotionJEPA
from scripts import trace_cue as cue


def test_only_frame_four_changes_and_bend_is_normal_to_the_chord():
    p = cue.cue_pairs(40, 99, ("triangle",))
    assert torch.equal(p.original[:, (0, -1)], p.moved[:, (0, -1)])
    assert ((p.original[:, 1] != p.moved[:, 1]).any(-1).any(-1)).all()
    assert p.truth[:, (0, -1)].count_nonzero() == 0
    assert len({np.packbits(x).tobytes() for x in p.original[:, (0, -1)].numpy().astype(bool)}) == 40
    grid = torch.arange(cue.tr.SIDE)
    def centroid(x):
        return torch.stack([(x.sum(dim)*grid).sum(-1) for dim in (-2, -1)], -1)/x.sum((-2, -1))[..., None]
    torch.testing.assert_close(centroid(p.moved) - centroid(p.original), p.truth[:, cue.SHOWN])
    a, _, b = centroid(p.original).unbind(1)
    torch.testing.assert_close(((b-a)*p.axis).sum(-1), torch.zeros(len(a)), atol=1e-5, rtol=0)
    assert (cue.per_pair(p.truth, p.axis)["bend_px"] > 0).all()


def test_bend_rejects_global_shift_and_endpoint_tilt_and_excludes_cue():
    t = torch.arange(8).float()[None, :, None]
    delta = (2+t)*torch.tensor([[[0., 1.]]])
    axis = torch.tensor([[0., 1.]])
    delta[:, cue.CUE] += 100
    def get(x):
        return cue.per_pair(x, axis)
    for key in ("bend_px", "hidden_only_bend_px"):
        torch.testing.assert_close(get(delta)[key], torch.zeros(1), atol=1e-6, rtol=0)
    delta[:, cue.SCORE[1]] += 3
    signed = (delta*axis[:, None]).sum(-1)
    torch.testing.assert_close(signed @ cue.bend_weights(), get(delta)["bend_px"])
    assert get(delta)["bend_px"].item() == pytest.approx(.6)


@torch.no_grad()
@pytest.mark.parametrize("name", ["JEPA", "RNN"])
def test_three_flash_route_and_no_cue_negative_control(name):
    p = cue.cue_pairs(8, 123, ("square",))
    model = (MotionJEPA(12, 24) if name == "JEPA" else ImputationNet(32**2, 24, 8)).eval()
    u, v = (cue.cue_states(model, x) for x in (p.original, p.moved))
    masked = [cue.ep.states(model, x[:, (0, -1)], 0)["final"] for x in (p.original, p.moved)]
    assert torch.equal(*masked)
    pre, dx, w = cue.ep.route_inputs(model, u, v, 0, slot_time=cue.CUE)
    transfer = cue.ep.route_finish(model, u, pre+torch.nn.functional.linear(dx, w))
    if name == "JEPA":
        for t in (0, 7):
            sl = cue.ep.time_slice(model, t)
            assert torch.equal(u["_slots"][:, sl], v["_slots"][:, sl])
        x = p.original.new_zeros(len(dx), 8, 32, 32)
        vis = torch.zeros(len(dx), 8, dtype=torch.bool)
        x[:, cue.SHOWN] = p.moved
        vis[:, cue.SHOWN] = True
        torch.testing.assert_close(v["final"], model(x, vis)[0])
    else:
        transfer += (1-model.rnn.alpha)*(v["gap"]-u["gap"])
    torch.testing.assert_close(transfer, v["final"])
