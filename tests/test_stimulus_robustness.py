import json
from itertools import product

import numpy as np
import pytest
import torch

from illusion_rnn import apparent as ap, trajectories as tr
from scripts import run_stimulus_robustness as exp


def test_visible_inputs_vary_but_ink_centres_and_trajectories_do_not():
    ep, future = exp.geometry(7)
    masks = set()
    for shape, size in product(exp.STIMULI, exp.SIZES):
        batch = exp.render(ep, shape, size)
        np.testing.assert_array_equal(batch.centres, ep.reshape(-1, tr.FRAMES, 2))
        yy, xx = np.mgrid[:tr.SIDE, :tr.SIDE]
        mass = batch.frames.sum((-2, -1))
        assert (mass == mass[0, 0]).all() and mass[0, 0] > 0
        centroid = np.stack([(batch.frames*grid).sum((-2, -1))/mass for grid in (xx, yy)], -1)
        np.testing.assert_array_equal(centroid, batch.centres)
        frames = batch.frames.reshape(*ep.shape[:3], tr.FRAMES, tr.SIDE, tr.SIDE)
        for a, p in product(range(3), repeat=2):
            np.testing.assert_array_equal(frames[:, a, p][:, [0, -1]], frames[:, 0, 0][:, [0, -1]])
        masks.add(frames[0, 0, 0, 0].tobytes())
        prefix = exp.prefixes(future, shape, size)
        for a in (1, 2):
            np.testing.assert_array_equal(prefix.prefix[:, a, :, -2:], prefix.prefix[:, 0, :, -2:])
    assert len(masks) == len(exp.STIMULI)*len(exp.SIZES)


def test_pixel_pairs_excluded_from_probe_fits():
    ep, future = exp.geometry(9)
    excluded_ep, excluded_last = exp.excluded_pairs(ep, future)
    for shape, size in product(exp.STIMULI, exp.SIZES):
        batch = exp.render(ep[:, 2, 2], shape, size)
        assert set(exp.interp.endpoint_keys(batch)) <= excluded_ep
        keys = exp.fore.keys(exp.prefixes(future[:, 1:2], shape, size))
        assert all(key in excluded_last for row in keys for key in row)
    fit = exp.fore.dataset(8, exp.fore.FIT_SEED, excluded_last)
    assert all(k not in excluded_last for row in exp.fore.keys(fit) for k in row)


def test_all_conditions_with_fixed_weights_and_matched_controls():
    ep, future = exp.geometry(5)
    excluded_ep, last = exp.excluded_pairs(ep, future)
    fit = exp.fore.dataset(5, exp.fore.FIT_SEED, last)
    model = ap.ImputationNet(tr.SIDE**2, 4, tr.FRAMES)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    rows, differences = exp.evaluate_seed(model, model, ep, future, excluded_ep, fit, 9)
    intervals = exp.paired_intervals([differences], draws=100)
    json.dumps(rows, allow_nan=False)
    assert len(rows) == len(exp.STIMULI)*len(exp.SIZES)
    for row, ci in zip(rows, intervals):
        final = row["last_hidden_frame"]
        expected = (final["ink_centre"]["straight"]["mean_error_px"]
                    - final["last_velocity"]["straight"]["mean_error_px"])
        assert ci["mean_px"] == pytest.approx(expected)
        assert ci["interval_95_px"][0] <= expected <= ci["interval_95_px"][1]
        classic = row["straight_fill_in"]
        assert 0 <= classic["patch_within_2px_all"] <= classic["patch_found"] <= 1
        choices = row["trajectory_choice"]["endpoints"]["probe"]
        assert choices["accel_choice"] == pytest.approx(1/3)
        assert choices["path_choice"] == pytest.approx(1/3)
        for cue in ("endpoints", "plus_cue"):
            control = row["trajectory_choice"][cue]["visible_line"]
            assert control == rows[0]["trajectory_choice"][cue]["visible_line"]
        for path in tr.PATH:
            baseline = row["last_hidden_frame"]["last_velocity"][path]
            assert baseline == rows[0]["last_hidden_frame"]["last_velocity"][path]
            assert baseline["separation_gain"] == 0.
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
