import numpy as np
import torch

from illusion_rnn.models import RNNNet, ShapesCNN
from illusion_rnn.training import (
    EvalResult,
    cnn_encoder,
    evaluate,
    make_dataset,
    make_env,
    resolve_device,
    train,
)


def _tiny_env():
    env = make_env(
        "tam", box_shape="square", variant="standard",
        stim_ori="horizontal", img_size=16,
    )
    env.seed(0)
    return env


def test_make_env_dispatch():
    assert type(_tiny_env()).__name__ == "TAMTask"
    motion = make_env("motion", box_shape="circle", motion_type="tracking",
                      stim_ori="horizontal", img_size=16)
    assert type(motion).__name__ == "MotionTask"
    try:
        make_env("bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_make_dataset_shapes():
    ds = make_dataset(_tiny_env(), batch_size=4, seq_len=18)
    inputs, labels = ds()
    assert inputs.shape == (18, 4, 16, 16)
    assert labels.shape == (18, 4)


def test_train_decreases_loss():
    torch.manual_seed(0)
    ds = make_dataset(_tiny_env(), batch_size=4, seq_len=18)
    model = RNNNet(input_size=16 * 16, hidden_size=32, output_size=6, dt=50)
    history = train(model, ds, n_epochs=40, lr=1e-3, device="cpu", log_every=0)
    assert len(history["loss"]) == 40
    assert len(history["accuracy"]) == 40
    assert np.mean(history["loss"][-5:]) < np.mean(history["loss"][:5])


def test_train_multi_dataset():
    torch.manual_seed(0)
    datasets = [
        make_dataset(_tiny_env(), batch_size=2, seq_len=9),
        make_dataset(_tiny_env(), batch_size=3, seq_len=9),
    ]
    model = RNNNet(16 * 16, 16, 6, dt=50)
    history = train(model, datasets, n_epochs=2, lr=1e-3, device="cpu", log_every=0)
    assert len(history["loss"]) == 2


def test_evaluate_structure():
    model = RNNNet(16 * 16, 32, 6, dt=50)
    result = evaluate(model, _tiny_env(), n_trials=5, device="cpu")
    assert isinstance(result, EvalResult)
    assert len(result.trials) == 5
    assert len(result.activity) == 5
    assert result.activity[0].shape == (9, 32)
    assert set(result.trials[0]) == {"ground_truth", "choice", "correct"}
    assert 0.0 <= result.accuracy <= 1.0


def test_encoder_hook_in_train_and_evaluate():
    def mean_encoder(inputs):  # (T, B, H, W) -> (T, B, 1)
        return inputs.mean(axis=(2, 3))[..., np.newaxis]

    torch.manual_seed(0)
    ds = make_dataset(_tiny_env(), batch_size=2, seq_len=9)
    model = RNNNet(input_size=1, hidden_size=8, output_size=6, dt=50)
    history = train(model, ds, n_epochs=2, lr=1e-3, device="cpu",
                    encoder=mean_encoder, log_every=0)
    assert len(history["loss"]) == 2
    result = evaluate(model, _tiny_env(), n_trials=3, device="cpu",
                      encoder=mean_encoder)
    assert result.activity[0].shape == (9, 8)


def test_cnn_encoder_shapes():
    # NOTE: 32 is the smallest input ShapesCNN's conv stack supports.
    cnn = ShapesCNN(input_size=32, feature_dim=12)
    encoder = cnn_encoder(cnn, device="cpu")
    out = encoder(np.random.rand(3, 2, 32, 32).astype(np.float32))
    assert out.shape == (3, 2, 12)


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"


def test_summarize_trials_counts_abstention_separately():
    """A model answering the fixation class is abstaining, not answering wrong.

    Ground truth is never `fixation` at the decision step, so any fixation
    choice is a refusal to commit. Folding those into `accuracy` is what made
    the 2023 outline number (0.100) read as below-chance confusion.
    """
    from illusion_rnn.training import summarize_trials

    trials = [
        {"ground_truth": 1, "choice": 1, "correct": True},   # correct
        {"ground_truth": 1, "choice": 3, "correct": False},  # wrong direction
        {"ground_truth": 3, "choice": 0, "correct": False},  # abstained
        {"ground_truth": 3, "choice": 0, "correct": False},  # abstained
    ]
    acc, abstention, committed, confusion = summarize_trials(trials, n_actions=6)
    assert acc == 0.25                      # 1 of 4
    assert abstention == 0.5                # 2 of 4 answered fixation
    assert committed == 0.5                 # 1 of the 2 that committed
    assert confusion[1, 1] == 1
    assert confusion[1, 3] == 1
    assert confusion[3, 0] == 2
    assert confusion.sum() == 4


def test_summarize_trials_all_abstained_gives_nan_committed_accuracy():
    """Committed accuracy over zero committed trials must be NaN, not 0.0.

    Returning 0.0 would silently claim the model got everything wrong when in
    fact it answered nothing.
    """
    from illusion_rnn.training import summarize_trials

    trials = [{"ground_truth": 1, "choice": 0, "correct": False}]
    acc, abstention, committed, _ = summarize_trials(trials, n_actions=6)
    assert acc == 0.0
    assert abstention == 1.0
    assert np.isnan(committed)


def test_eval_result_exposes_new_fields():
    result = EvalResult(
        accuracy=0.5, abstention_rate=0.25,
        committed_accuracy=0.667, confusion=np.zeros((6, 6)),
    )
    assert result.abstention_rate == 0.25
    assert result.committed_accuracy == 0.667
    assert result.confusion.shape == (6, 6)
