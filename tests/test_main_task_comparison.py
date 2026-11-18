import json

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn.jepa import MotionJEPA
from scripts import compare_main_task as exp


def test_original_task_split_has_no_repeated_endpoint_pairs():
    seen = set()
    fit = exp.rp.sample_trials(64, 22, fit=True, seen=seen, holdouts=False)
    test = exp.rp.sample_trials(64, 22, fit=False, seen=seen, holdouts=False)
    assert len(seen) == 128
    for trials in (fit, test):
        assert set(trials.shapes) == set(exp.rp.FAMILIAR)
        assert exp.rp.inside_tile(trials.centres).any()
        assert np.all(np.any(trials.centres[:, 0] != trials.centres[:, -1], axis=1))


def test_controls_and_missing_patch_denominator():
    truth = np.tile(np.arange(8)[None, :, None], (2, 1, 2)).astype(float)
    ref = exp.controls(truth)
    np.testing.assert_array_equal(ref["line"], truth[:, 1:-1])
    assert exp.metrics(ref["line"], truth[:, 1:-1])["mean_error_px"] == 0
    assert (ref["jump"][:, :3] == 0).all() and (ref["jump"][:, 3:] == 7).all()
    xy = truth[:, 1:-1].copy()
    xy[0] = np.nan
    score = exp.metrics(xy, truth[:, 1:-1])
    assert score["mean_error_px"] == 0
    assert score["coverage"] == score["within_2px_all"] == .5
    assert exp.paired_interval(np.full((3, 20), -1.5), 30)["paired_95_px"] == [-1.5, -1.5]


def test_both_models_use_only_test_endpoints_and_score_six_positions():
    seen = set()
    fit = exp.rp.sample_trials(16, 123, fit=True, seen=seen, holdouts=False)
    test = exp.rp.sample_trials(7, 456, fit=False, seen=seen, holdouts=False)
    rnn, jepa = ap.ImputationNet(32**2, 8, 8), MotionJEPA(8, 16)
    weights = [{k: v.clone() for k, v in model.state_dict().items()} for model in (rnn, jepa)]
    preds = exp.predictions(rnn.eval(), jepa.eval(), fit, test)
    assert set(preds) == {"rnn_state", "jepa_state", "rnn_output", "jepa_output", "rnn_patch"}
    assert all(x.shape == (7, 6, 2) for x in preds.values())
    rows = exp.examples(test, preds)
    json.dumps(rows, allow_nan=False)
    for row in rows:
        assert row["trial"] == np.flatnonzero(test.shapes == row["shape"])[0]
    test.centres[:] = np.nan
    again = exp.predictions(rnn, jepa, fit, test)
    for k, actual in again.items():
        np.testing.assert_allclose(actual, preds[k], equal_nan=True)
    for model, before in zip((rnn, jepa), weights):
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
