"""illusion-rnn: a neurogym testbed for Transformational Apparent Motion (TAM)."""

from illusion_rnn.apparent import (
    ImputationNet,
    evaluate_imputation,
    sample_masks,
    sample_motion,
    train_imputation,
)
from illusion_rnn.envs import (
    MOTION_CHOICES,
    TAM_CHOICES,
    MotionTask,
    TAMCorrespondenceTask,
    TAMTask,
    rotate_stimuli,
)
from illusion_rnn.models import (
    CTRNN,
    FFStack,
    FrameOnlyNet,
    GRUNet,
    RNNNet,
    ShapesCNN,
    load_rnn,
    shuffle_frames,
)
from illusion_rnn.plotting import plot_training_curves, plot_trials
from illusion_rnn.stimuli import (
    MOTION_TYPES,
    SHAPES,
    TAM_VARIANTS,
    load_motion,
    load_tam,
)
from illusion_rnn.training import (
    EvalResult,
    cnn_encoder,
    evaluate,
    make_dataset,
    make_env,
    resolve_device,
    train,
)

__version__ = "1.0.0"

__all__ = [
    "CTRNN",
    "FFStack",
    "FrameOnlyNet",
    "GRUNet",
    "ImputationNet",
    "MOTION_CHOICES",
    "MOTION_TYPES",
    "EvalResult",
    "MotionTask",
    "RNNNet",
    "SHAPES",
    "ShapesCNN",
    "TAM_CHOICES",
    "TAM_VARIANTS",
    "TAMCorrespondenceTask",
    "TAMTask",
    "cnn_encoder",
    "evaluate",
    "evaluate_imputation",
    "load_motion",
    "load_rnn",
    "load_tam",
    "make_dataset",
    "make_env",
    "plot_training_curves",
    "plot_trials",
    "resolve_device",
    "rotate_stimuli",
    "sample_masks",
    "sample_motion",
    "shuffle_frames",
    "train",
    "train_imputation",
    "__version__",
]
