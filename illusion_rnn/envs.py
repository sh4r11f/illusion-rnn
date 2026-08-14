"""Neurogym trial environments for TAM and real-motion tasks.

Label maps are frozen: the shipped checkpoints' output heads were trained
against these exact indices.
"""

import numpy as np

import neurogym as ngym
from neurogym.core import TrialEnv

from illusion_rnn.stimuli import (
    MOTION_TYPES,
    SHAPES,
    TAM_VARIANTS,
    load_motion,
    load_tam,
)

TAM_CHOICES = {"fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5}
MOTION_CHOICES = {
    "no_motion": 0, "left": 1, "middle": 2, "right": 3,
    "down": 4, "up": 5, "fixation": 6,
}

# direction index -> stimulus-file motion id (vertical stimuli are the
# horizontal files rotated 90° clockwise, so up/down reuse left/right files)
_MOTION_ID = {0: "no_motion", 1: "left", 2: "middle", 3: "right", 4: "right", 5: "left"}


def _rotate(obj):
    if isinstance(obj, np.ndarray):
        return np.rot90(obj, k=-1)
    return [_rotate(item) for item in obj]


def rotate_stimuli(stimuli: dict) -> dict:
    """Return a copy of a stimulus dict with every frame rotated 90° clockwise."""
    return {key: _rotate(value) for key, value in stimuli.items()}


def _validate(name, value, allowed):
    if value not in allowed:
        msg = f"Unknown {name} {value!r}; expected one of {tuple(allowed)}"
        raise ValueError(msg)


class TAMTask(TrialEnv):
    """3AFC Transformational Apparent Motion task.

    Trial structure: fixation (100 ms) -> init frame (50 ms) -> a repeated
    second frame (4 x 50 ms) -> decision (100 ms), dt = 50 ms. Ground truth
    is ``fixation`` (0) through frame1 and the motion direction from frame2
    onward. Horizontal trials sample directions {left, middle, right};
    vertical trials {middle, down, up} on stimuli rotated 90° clockwise
    once at construction.

    Parameters
    ----------
    dt : int
        Time step in ms.
    box_shape : str
        One of ``illusion_rnn.stimuli.SHAPES``.
    variant : str
        Stimulus set: ``standard`` | ``basic`` | ``outline``.
    stim_ori : str
        ``horizontal`` | ``vertical``.
    stimuli : dict, optional
        Pre-loaded stimulus dict (see ``stimuli.load_tam``). Loaded from the
        packaged set for ``variant`` when None. Keys must use the variant's
        filename prefix (``tam`` or ``outline``).
    sigma : float
        Std of Gaussian noise added to the frame periods.
    img_size : int
        Height/width of the (square) observation.
    rewards, timing : dict, optional
        Overrides merged into the defaults.
    """

    def __init__(
        self,
        dt: int = 50,
        box_shape: str = "square",
        variant: str = "standard",
        stim_ori: str = "horizontal",
        stimuli: dict | None = None,
        sigma: float = 0.0,
        img_size: int = 64,
        rewards: dict | None = None,
        timing: dict | None = None,
    ):
        super().__init__(dt=dt)
        _validate("box_shape", box_shape, SHAPES)
        _validate("variant", variant, TAM_VARIANTS)
        _validate("stim_ori", stim_ori, ("horizontal", "vertical"))

        self.box_shape = box_shape
        self.variant = variant
        self.stim_ori = stim_ori
        self.sigma = sigma
        self.img_size = img_size
        self._prefix = "outline" if variant == "outline" else "tam"

        if stimuli is None:
            stimuli = load_tam(img_size, variant=variant)
        if stim_ori == "vertical":
            stimuli = rotate_stimuli(stimuli)
        self._stimuli = stimuli

        self.abort = False
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 100,
            "frame1": 50, "frame2": 50, "frame3": 50, "frame4": 50, "frame5": 50,
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.ob_shape = (img_size, img_size)
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=self.ob_shape, dtype=np.float32,
        )
        self.choice_names = dict(TAM_CHOICES)
        self.action_space = ngym.spaces.Discrete(6, name=self.choice_names)

    def _sample_direction(self) -> int:
        if self.stim_ori == "horizontal":
            options = [TAM_CHOICES["left"], TAM_CHOICES["middle"], TAM_CHOICES["right"]]
        else:
            options = [TAM_CHOICES["middle"], TAM_CHOICES["down"], TAM_CHOICES["up"]]
        return int(self.rng.choice(options))

    def _fixation_ob(self) -> np.ndarray:
        fix = np.zeros(self.ob_shape)
        center, half = self.img_size // 2, 1
        fix[center - half:center + half, center - half:center + half] = 1.0
        return fix

    def _new_trial(self, **kwargs):
        direction = self._sample_direction()
        trial = {
            "ground_truth": direction,
            "box_shape": self.box_shape,
            "variant": self.variant,
            "stim_ori": self.stim_ori,
            "noise": self.sigma,
        }
        trial.update(kwargs)

        motion_id = _MOTION_ID[trial["ground_truth"]]
        key = f"{self.box_shape}-{self._prefix}-{motion_id}"
        second_frames = self._stimuli[key]
        init_frame = self._stimuli[f"{self.box_shape}-{self._prefix}-init"][0]
        second = second_frames[int(self.rng.randint(len(second_frames)))]
        frames = [init_frame] + [second] * 4

        self.add_period(
            ["fixation", "frame1", "frame2", "frame3", "frame4", "frame5", "decision"],
        )
        self.add_ob(self._fixation_ob(), period=["fixation"])
        for i, frame in enumerate(frames):
            self.add_ob(frame, period=[f"frame{i + 1}"])
        self.add_ob(np.zeros(self.ob_shape), period=["decision"])
        self.add_randn(
            0, self.sigma,
            period=["frame1", "frame2", "frame3", "frame4", "frame5"],
        )

        self.set_groundtruth(self.choice_names["fixation"], period=["fixation", "frame1"])
        self.set_groundtruth(
            trial["ground_truth"],
            period=["frame2", "frame3", "frame4", "frame5", "decision"],
        )
        return trial

    def _step(self, action):
        new_trial = False
        reward = 0.0
        gt = self.gt_now
        fixation_action = self.choice_names["fixation"]

        if self.in_period("fixation"):
            if action != fixation_action:
                new_trial = self.abort
                reward += self.rewards["abort"]
        elif self.in_period("decision") and action != fixation_action:
            new_trial = True
            if action == gt:
                reward += self.rewards["correct"]
                self.performance = 1
            else:
                reward += self.rewards["fail"]

        return self.ob_now, reward, False, False, {"new_trial": new_trial, "gt": gt}


class MotionTask(TrialEnv):
    """Real-motion 4AFC task (control condition for TAM).

    Trial structure: fixation (100 ms) -> 5 motion frames (50 ms each) ->
    decision (150 ms), dt = 50 ms. Ground truth is ``fixation`` (6) outside
    the decision period and the motion direction during it. ``no_motion``
    trials repeat a single randomly drawn frame. Horizontal trials sample
    {no_motion, left, middle, right}; vertical {no_motion, middle, down, up}
    on stimuli rotated 90° clockwise once at construction.

    Parameters are as in ``TAMTask`` except ``motion_type``
    (``continuous`` | ``tracking``) replacing ``variant``.
    """

    def __init__(
        self,
        dt: int = 50,
        box_shape: str = "square",
        motion_type: str = "continuous",
        stim_ori: str = "horizontal",
        stimuli: dict | None = None,
        sigma: float = 0.0,
        img_size: int = 64,
        rewards: dict | None = None,
        timing: dict | None = None,
    ):
        super().__init__(dt=dt)
        _validate("box_shape", box_shape, SHAPES)
        _validate("motion_type", motion_type, MOTION_TYPES)
        _validate("stim_ori", stim_ori, ("horizontal", "vertical"))

        self.box_shape = box_shape
        self.motion_type = motion_type
        self.stim_ori = stim_ori
        self.sigma = sigma
        self.img_size = img_size
        self._prefix = {"continuous": "cnt", "tracking": "track"}[motion_type]

        if stimuli is None:
            stimuli = load_motion(img_size, motion_type=motion_type)
        if stim_ori == "vertical":
            stimuli = rotate_stimuli(stimuli)
        self._stimuli = stimuli

        self.abort = False
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 100,
            "frame1": 50, "frame2": 50, "frame3": 50, "frame4": 50, "frame5": 50,
            "decision": 150,
        }
        if timing:
            self.timing.update(timing)

        self.ob_shape = (img_size, img_size)
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=self.ob_shape, dtype=np.float32,
        )
        self.choice_names = dict(MOTION_CHOICES)
        self.action_space = ngym.spaces.Discrete(7, name=self.choice_names)

    def _sample_direction(self) -> int:
        if self.stim_ori == "horizontal":
            options = [0, 1, 2, 3]  # no_motion, left, middle, right
        else:
            options = [0, 2, 4, 5]  # no_motion, middle, down, up
        return int(self.rng.choice(options))

    def _fixation_ob(self) -> np.ndarray:
        fix = np.zeros(self.ob_shape)
        center, half = self.img_size // 2, 1
        fix[center - half:center + half, center - half:center + half] = 1.0
        return fix

    def _sample_frames(self, direction: int) -> list:
        if direction == MOTION_CHOICES["no_motion"]:
            file_dir = ("left", "right")[int(self.rng.randint(2))]
            exemplars = self._stimuli[f"{self.box_shape}-{self._prefix}-{file_dir}"]
            exemplar = exemplars[int(self.rng.randint(len(exemplars)))]
            frame = exemplar[int(self.rng.randint(len(exemplar)))]
            return [frame] * 5
        motion_id = _MOTION_ID[direction]
        exemplars = self._stimuli[f"{self.box_shape}-{self._prefix}-{motion_id}"]
        return exemplars[int(self.rng.randint(len(exemplars)))]

    def _new_trial(self, **kwargs):
        direction = self._sample_direction()
        trial = {
            "ground_truth": direction,
            "box_shape": self.box_shape,
            "motion_type": self.motion_type,
            "stim_ori": self.stim_ori,
            "noise": self.sigma,
        }
        trial.update(kwargs)
        frames = self._sample_frames(trial["ground_truth"])

        self.add_period(
            ["fixation", "frame1", "frame2", "frame3", "frame4", "frame5", "decision"],
        )
        self.add_ob(self._fixation_ob(), period=["fixation"])
        for i, frame in enumerate(frames):
            self.add_ob(frame, period=[f"frame{i + 1}"])
        self.add_ob(np.zeros(self.ob_shape), period=["decision"])
        self.add_randn(
            0, self.sigma,
            period=["frame1", "frame2", "frame3", "frame4", "frame5"],
        )

        self.set_groundtruth(
            self.choice_names["fixation"],
            period=["fixation", "frame1", "frame2", "frame3", "frame4", "frame5"],
        )
        self.set_groundtruth(trial["ground_truth"], period=["decision"])
        return trial

    def _step(self, action):
        new_trial = False
        reward = 0.0
        gt = self.gt_now
        fixation_action = self.choice_names["fixation"]

        if self.in_period("fixation"):
            if action != fixation_action:
                new_trial = self.abort
                reward += self.rewards["abort"]
        elif self.in_period("decision") and action != fixation_action:
            new_trial = True
            if action == gt:
                reward += self.rewards["correct"]
                self.performance = 1
            else:
                reward += self.rewards["fail"]

        return self.ob_now, reward, False, False, {"new_trial": new_trial, "gt": gt}
