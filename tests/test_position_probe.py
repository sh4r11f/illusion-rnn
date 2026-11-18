import json

import numpy as np
import pytest
import torch

from illusion_rnn import apparent as ap
from scripts import run_position_probe as probe


def test_probe_splits_disallow_shape_location_and_duplicate_movies():
    seen = set()
    fit = probe.sample_trials(64, 1, fit=True, seen=seen)
    # Same RNG seed to stress the exact-pair deduplication.
    test = probe.sample_trials(256, 1, fit=False, seen=seen)
    assert set(fit.shapes) == set(probe.FAMILIAR)
    assert not probe.inside_tile(fit.centres).any()
    novel = (test.shapes == probe.NOVEL)[:, None]
    tile = probe.inside_tile(test.centres[:, 1:-1])
    assert all(m.any() for m in (novel & tile, novel & ~tile, ~novel & tile, ~novel & ~tile))
    movies = np.concatenate([fit.endpoints, test.endpoints])
    assert len({m.tobytes() for m in movies}) == len(movies)
    np.testing.assert_array_equal(probe.inside_tile(np.array([[11, 11], [11, 21]])),
                                  [True, False])


def test_probe_extract_uses_masked_final_state_without_updating_model():
    model = ap.ImputationNet(probe.SIDE**2, 8, probe.FRAMES)
    batch = ap.sample_motion(np.random.default_rng(12), 4, p_static=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ends = batch.frames[:, [0, -1]].astype(bool)
    state, xy = probe.extract(model, ends, patches=True)
    visible = ap.endpoint_mask(len(ends), probe.FRAMES)
    x = ap._to_model_input(ap.apply_mask(batch.frames, visible), "cpu")
    with torch.no_grad():
        logits, activity = model(x)
    np.testing.assert_allclose(state, activity[-1].numpy())
    state_only, _ = probe.extract(model, ends)
    np.testing.assert_array_equal(state_only, state)
    probs = logits.sigmoid().transpose(0, 1).reshape(batch.frames.shape).numpy()
    ref = ap.patch_position_metrics(probs, batch, visible)
    err = np.linalg.norm(xy - batch.centres[:, 1:-1], axis=-1)
    assert np.nanmean(err) == pytest.approx(ref["patch_err"])
    assert np.isfinite(err).mean() == pytest.approx(ref["patch_found"])
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v)


def test_affine_probe_recovers_coordinates_on_fresh_trials():
    rng = np.random.default_rng(4)
    x, test, w = rng.normal(size=(1024, 5)), rng.normal(size=(7, 5)), rng.normal(size=(5, 12))
    xy = (x @ w + 12).reshape(-1, 6, 2)
    result = probe.ridge_predict(x, xy, test)
    np.testing.assert_allclose(result.reshape(-1, 12), test @ w + 12, atol=.02)


def test_no_patch_counts_as_miss_not_zero_error():
    test = probe.sample_trials(5, 88, fit=False, seen=set())
    truth = test.centres[:, 1:-1]
    patches = truth.copy()
    patches[:, 1::2] = np.nan
    scores = probe.score(test, {"probe": truth}, patches)
    overall = scores["overall"]
    assert overall["all"]["patch"] == {"mean_px": 0, "within_2px": .5}
    assert overall["missing"]["patch"] == {"mean_px": None, "within_2px": 0}
    assert overall["missing"]["probe"] == {"mean_px": 0, "within_2px": 1}
    assert sum(scores[s]["all"]["n_frames"]
               for s in ("familiar", "shape", "location", "both")) == truth.shape[0] * 6
    json.dumps(scores, allow_nan=False)
