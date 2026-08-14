import matplotlib.pyplot as plt

from illusion_rnn.plotting import plot_training_curves, plot_trials
from illusion_rnn.training import make_env


def test_plot_trials_grid():
    env = make_env("tam", box_shape="square", variant="standard",
                   stim_ori="horizontal", img_size=16)
    env.seed(0)
    fig = plot_trials(env, n_trials=2)
    assert len(fig.axes) == 2 * 9  # trials x timesteps
    plt.close(fig)


def test_plot_trials_save(tmp_path):
    env = make_env("motion", box_shape="circle", motion_type="continuous",
                   stim_ori="horizontal", img_size=16)
    env.seed(0)
    out = tmp_path / "trials.png"
    fig = plot_trials(env, n_trials=1, save_path=out)
    assert out.exists()
    plt.close(fig)


def test_plot_training_curves():
    history = {"loss": [1.5, 0.8, 0.4], "accuracy": [0.2, 0.5, 0.9]}
    fig = plot_training_curves(history)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_package_exports():
    import illusion_rnn

    for name in (
        "TAMTask", "MotionTask", "TAM_CHOICES", "MOTION_CHOICES",
        "rotate_stimuli", "load_tam", "load_motion", "SHAPES",
        "TAM_VARIANTS", "MOTION_TYPES", "CTRNN", "RNNNet", "ShapesCNN",
        "load_rnn", "make_env", "make_dataset", "train", "evaluate",
        "EvalResult", "cnn_encoder", "resolve_device", "plot_trials",
        "plot_training_curves",
    ):
        assert hasattr(illusion_rnn, name), name
