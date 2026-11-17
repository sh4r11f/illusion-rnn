import json

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn import trajectories as tr
from scripts import run_prefix_forecast as exp


def test_matched_final_step_and_distinct_futures():
    tracks, sizes, _ = tr.forecast_tracks(np.random.default_rng(9), 200)
    assert np.rint(tracks).min() >= 0
    assert (np.rint(tracks) <= (tr.SIDE - sizes)[:, None, None, None, None]).all()
    for i in (1, 2):
        np.testing.assert_array_equal(tracks[:, i, :, exp.P-2:exp.P],
                                      tracks[:, 0, :, exp.P-2:exp.P])
    speed = np.linalg.norm(np.diff(tracks, axis=-2), axis=-1)
    assert (np.diff(speed[:, 1], axis=-1) > 0).all()
    assert (np.diff(speed[:, 2], axis=-1) < 0).all()
    np.testing.assert_allclose(np.diff(speed[:, 0], axis=-1), 0., atol=1e-12)
    assert np.linalg.norm(tracks[:, 1, :, -1]-tracks[:, 2, :, -1], axis=-1).min() > 3


def test_no_future_inputs_and_no_repeated_last_pair_between_splits():
    test = exp.dataset(12, 4)
    excluded = {k for row in exp.keys(test) for k in row}
    fit = exp.dataset(7, 4, excluded)
    assert excluded.isdisjoint(k for row in exp.keys(fit) for k in row)
    prefix = test.prefix.reshape(-1, exp.P, tr.SIDE, tr.SIDE)
    x = exp.masked_input(prefix)
    assert x.shape == (tr.FRAMES, len(prefix), tr.SIDE**2)
    assert x[exp.P:].count_nonzero() == 0
    np.testing.assert_array_equal(x[:exp.P].transpose(0, 1).numpy().reshape(prefix.shape), prefix)


def test_baselines_continue_velocity_or_acceleration_without_future():
    t = np.arange(tr.FRAMES, dtype=float)
    xy = np.stack((2*t + .4*t**2, 5-t), axis=-1)[None]
    pred = exp.kinematics(xy[:, :exp.P])
    np.testing.assert_allclose(pred["prefix_quadratic"], xy[:, exp.P:], atol=1e-10)
    step = xy[:, exp.P-1] - xy[:, exp.P-2]
    for h in range(1, exp.H+1):
        np.testing.assert_allclose(pred["last_velocity"][:, h-1], xy[:, exp.P-1] + h*step)


def test_separation_gain_controls_and_missing_patch_denominator():
    m = exp.sample(np.random.default_rng(40), 8)
    y = m.xy[..., exp.P:, :]
    cv = exp.kinematics(m.xy[..., :exp.P, :])["last_velocity"]
    partial = y.copy()
    partial[:4] = np.nan
    for p, gain, coverage in ((y, 1., 1.), (cv, 0., 1.), (partial, 1., .5)):
        rows = exp.scores(p, m.xy)
        for path in rows.values():
            assert all(r["separation_gain"] == gain and r["coverage"] == coverage for r in path)
            if coverage < 1:
                assert all(r["within_2px_all"] == coverage for r in path)
    empty = exp.scores(np.full_like(y, np.nan), m.xy)
    assert empty["straight"][-1]["separation_gain"] is None
    json.dumps(empty, allow_nan=False)


def test_frozen_rnn_readouts_smoke():
    m = ap.ImputationNet(tr.SIDE**2, 8, tr.FRAMES)
    initial = {k: p.clone() for k, p in m.state_dict().items()}
    scores = exp.network_scores(m, exp.dataset(4, 123), exp.dataset(3, 456))
    assert scores.keys() == {"probe", "ink_centre", "patch"}
    json.dumps(scores, allow_nan=False)
    assert all(torch.equal(initial[k], p) for k, p in m.state_dict().items())
