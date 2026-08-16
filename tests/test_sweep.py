"""Integration checks that the task is learnable and the baselines are floored.

These train real (tiny) models, so they are slower than the unit tests. They are
the cheap version of verification checkpoint 2; scripts/verify_baselines.py is
the full version.
"""
import numpy as np
import pytest

from illusion_rnn import evaluate, make_dataset, make_env, train
from illusion_rnn.models import FrameOnlyNet, RNNNet


@pytest.mark.slow
def test_frame2_only_baseline_cannot_beat_chance():
    """The gate. Frame 2 is independent of the label by construction, so a
    model that sees only frame 2 must sit at 50% no matter how long it trains.
    """
    env = make_env("correspondence", split="train", img_size=32)
    env.seed(0)
    # seq_len must equal a single trial's length (9 steps for the default
    # correspondence timing), not a multiple of it: ngym.Dataset concatenates
    # trials to fill seq_len, but FrameOnlyNet enforces an exact n_steps
    # match by construction (it fails loudly on any other T), so a seq_len
    # of 90 (10 concatenated trials) would raise a shape error immediately,
    # before the model ever gets a gradient step. See tests/test_training.py
    # for the same seq_len == n_steps convention used elsewhere.
    dataset = make_dataset(env, batch_size=32, seq_len=9)
    model = FrameOnlyNet(
        32 * 32, 128, 6, n_steps=9, frame_index=env.frame2_indices[0],
    )
    train(model, dataset, n_epochs=300, device="cpu", log_every=0)

    eval_env = make_env("correspondence", split="train", img_size=32)
    eval_env.seed(1)
    result = evaluate(model, eval_env, n_trials=400, device="cpu")
    committed = result.committed_accuracy
    assert np.isnan(committed) or committed < 0.60, (
        f"frame-2-only reached {committed:.3f}; the generator leaks and every "
        f"downstream result is void"
    )


@pytest.mark.slow
def test_the_task_is_learnable_by_a_recurrent_model():
    """The other half of the gate: if the CTRNN cannot learn it either, the
    task is impossible rather than merely well-controlled."""
    env = make_env("correspondence", split="train", img_size=32)
    env.seed(0)
    dataset = make_dataset(env, batch_size=32, seq_len=90)
    model = RNNNet(32 * 32, 256, 6, dt=50)
    train(model, dataset, n_epochs=600, device="cpu", log_every=0)

    eval_env = make_env("correspondence", split="train", img_size=32)
    eval_env.seed(1)
    result = evaluate(model, eval_env, n_trials=400, device="cpu")
    assert result.accuracy > 0.70, (
        f"CTRNN reached only {result.accuracy:.3f}; the task may be unlearnable "
        f"at this size rather than well-controlled"
    )
