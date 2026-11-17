import json

import numpy as np
import pytest
import torch

from illusion_rnn import apparent as ap
from illusion_rnn import trajectories as tr
from scripts import run_trajectory_experiment as exp


def test_tangential_speed_and_path_geometry():
    c = tr.Chords(np.zeros((1, 2)), np.array([[1., 0.]]), np.array([[16.]]),
                  np.array([4]), np.array(["square"]))
    for path in tr.PATH:
        tracks = [tr.trajectory(c, accel, path)[0] for accel in tr.ACCEL]
        steady, faster, slower = [np.linalg.norm(np.diff(xy, axis=0), axis=1) for xy in tracks]
        np.testing.assert_allclose(steady, steady[0])
        assert np.all(np.diff(faster) > 0)
        np.testing.assert_allclose(slower, faster[::-1])
        for xy in tracks:
            np.testing.assert_allclose(xy[[0, -1]], [[-8, 0], [8, 0]])
            if path == "straight":
                assert np.all(xy[:, 1] == 0)
            else:
                sign = 1 if path == "arc_left" else -1
                assert np.all(xy[1:-1, 1] * sign < 0)
                r = 8 / np.sin(tr.ANGLE / 2)
                centre = [0, sign * r * np.cos(tr.ANGLE / 2)]
                np.testing.assert_allclose(np.linalg.norm(xy - centre, axis=1), r)


def test_variants_share_endpoint_pixels_and_remain_in_bounds():
    c = tr.sample_chords(np.random.default_rng(344), 48)
    ref = tr.render(c, "steady", "straight")
    for accel in tr.ACCEL:
        for path in tr.PATH:
            batch = tr.render(c, accel, path)
            assert torch.equal(exp.inputs(ref, False), exp.inputs(batch, False))
            visible = exp.inputs(batch, True)
            assert visible[exp.SCORE].count_nonzero() == 0
            assert visible[exp.CUE].count_nonzero() > 0
            mass = batch.frames.sum((-2, -1))
            assert (mass == mass[:, :1]).all()
            yy, xx = np.mgrid[:tr.SIDE, :tr.SIDE]
            truth = np.stack([(batch.frames * grid).sum((-2, -1))/mass
                              for grid in (xx, yy)], axis=-1)
            np.testing.assert_allclose(batch.centres, truth, atol=2e-6)


def test_matched_choice_chance_and_perfect_controls():
    c = tr.sample_chords(np.random.default_rng(118), 24)
    targets = np.array([[tr.render(c, a, p).centres for p in tr.PATH] for a in tr.ACCEL])
    truth = targets[:, :, :, exp.SCORE, :]
    perfect = exp.factor_scores(truth, truth)
    assert perfect["accel_choice"] == perfect["path_choice"] == 1
    same = np.broadcast_to(truth[:1, :1], truth.shape)
    chance = exp.factor_scores(same, truth)
    assert chance["accel_choice"] == pytest.approx(1/3)
    assert chance["path_choice"] == pytest.approx(1/3)
    line = exp.visible_line(targets[0, 0], True)
    targets[0, 0, :, exp.SCORE, :] = 100
    np.testing.assert_array_equal(line, exp.visible_line(targets[0, 0], True))


def test_probe_evaluation_smoke():
    model = ap.ImputationNet(tr.SIDE**2, 8, tr.FRAMES)
    result = exp.test_seed(model, 12, 5)
    json.dumps(result, allow_nan=False)
    assert result["endpoints"]["probe"]["accel_choice"] == pytest.approx(1/3)
    for row in result.values():
        assert 0 <= row["patch_within_2px"] <= row["patch_found"] <= 1
