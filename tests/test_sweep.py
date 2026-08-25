"""Integration checks that the task is learnable and the baselines are floored.

These train real (tiny) models, so they are slower than the unit tests. They are
the cheap version of verification checkpoint 2; scripts/verify_baselines.py is
the full version.
"""
import numpy as np
import pytest

from illusion_rnn import evaluate, make_dataset, make_env, train
from illusion_rnn.models import FrameOnlyNet, RNNNet
from illusion_rnn.sweep import ARCHITECTURES, aggregate, build_model, run_grid


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


def test_every_named_architecture_builds():
    # img_size=32, not the brief's literal 16: generate.py's default
    # bar_length_range=(16, 28) requires img_size >= 29 (usable =
    # img_size - bar_length_range[1] must be >= 1), so a 16px correspondence
    # canvas raises ValueError on the very first new_trial() regardless of
    # sweep.py -- confirmed with a vanilla make_env() call. generate.py is
    # frozen (Tasks 3-6), so the fix is here; 32 matches the img_size already
    # used by this file's other correspondence tests.
    env = make_env("correspondence", split="train", img_size=32)
    env.seed(0); env.new_trial()
    for name in ARCHITECTURES:
        model = build_model(name, input_size=32 * 32, hidden_size=8,
                            output_size=6, n_steps=env.ob.shape[0], env=env)
        assert model is not None, name


def test_build_model_rejects_an_unknown_architecture():
    env = make_env("correspondence", split="train", img_size=32)
    env.seed(0); env.new_trial()
    with pytest.raises(ValueError, match="Unknown architecture 'transformer'"):
        build_model("transformer", input_size=1024, hidden_size=8,
                    output_size=6, n_steps=9, env=env)


@pytest.mark.slow
def test_run_grid_produces_one_record_per_cell_and_split():
    records = run_grid(
        architectures=("FF2Only",), families=("balanced",),
        hidden_sizes=(8,), seeds=(0, 1),
        img_size=32, n_epochs=5, n_eval_trials=20,
    )
    assert len(records) == 2 * 4      # 2 seeds x 4 splits
    for r in records:
        assert set(r) >= {"architecture", "family", "hidden_size", "seed",
                          "split", "accuracy", "abstention_rate",
                          "committed_accuracy"}


def test_aggregate_computes_mean_and_ci_per_cell():
    records = [
        {"architecture": "A", "family": "balanced", "hidden_size": 8,
         "split": "train", "seed": s, "accuracy": a}
        for s, a in enumerate([0.5, 0.6, 0.7, 0.8, 0.9])
    ]
    out = aggregate(records)
    assert len(out) == 1
    row = out[0]
    assert row["accuracy_mean"] == pytest.approx(0.7)
    assert row["n_seeds"] == 5
    assert row["accuracy_ci_low"] < 0.7 < row["accuracy_ci_high"]


def test_aggregate_reports_a_degenerate_ci_for_a_single_seed():
    """One seed gives no interval; it must say so rather than invent one."""
    records = [{"architecture": "A", "family": "balanced", "hidden_size": 8,
                "split": "train", "seed": 0, "accuracy": 0.7}]
    row = aggregate(records)[0]
    assert row["n_seeds"] == 1
    assert np.isnan(row["accuracy_ci_low"])
