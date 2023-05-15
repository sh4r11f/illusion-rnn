#!usr/bin/env python
"""
Created Nov 28 2022

By Sharif Saleki

Psychophysics task for TAM stimuli and paradigm made for neural networks in neurogym

"""
from typing import Union

import neurogym as ngym
from PIL import Image
from collections import defaultdict
from pathlib import Path

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class TAMTask(ngym.TrialEnv):
    """
    Uses TAM stimuli in a 3AFC choice task where the subject finds a shape correspondence and infers direction
    of motion based on contour matching.

    Each trial has the following structure: fixation -> first frame -> second frame -> decision period

    The class inherits dt, rewards, and timing from neurogym TrialEnv.

    Parameters
    ----------
    box_shape : str
        The shape of the boxes at the opposite ends of the TAM stimulus. Can be 'circle', 'square', 'rectangle',
        or 'triangle'.

    bar_shape : str
        The shape of the bars in the TAM stimulus. Can be 'curved', 'rectangle', or 'parallelogram'.

    stim_ori : str
        The orientation of the bars in the TAM stimulus. Can be 'horizontal' or 'vertical'.

    stim_connected : bool
        Whether the second frame stimuli in the TAM stimulus are connected or not.

    bar_connected: bool
        Whether the bar and the stimuli in the TAM stimulus are connected or not.

    sigma: float
        Input image's noise level

    cnn: bool
        Are the images going to pass through a CNN first?
    """
    metadata = {
        'description': "TAM dataset to train and test RNNs",
        'paper_link': '',
        'paper_name': "",
        'tags': ['perceptual', 'supervised', 'Transformational Apparent Motion']
    }

    def __init__(
            self,
            dt=50,
            box_shape: str = None,
            stimuli: dict = None,
            stim_type: str = None,
            stim_ori: str = None,
            sigma: float = 0.0,
            img_size: int = 32,
            cnn: bool = False,
            rewards=None,
            timing=None,
    ):

        # Initialize
        super().__init__(dt=dt)

        # Stimulus parameters
        self._box_shape = box_shape
        self._stimuli = stimuli
        self._stim_type = stim_type
        self._stim_ori = stim_ori
        self.sigma = sigma
        self.cnn = cnn

        # Reward parameters
        self.abort = False
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Timing parameters: total trial is 500ms
        self.dt = dt
        self.trial_dur = 500
        self.task_len = self.trial_dur // self.dt
        # self.timing = {
        #     'fixation': 100,
        #     'frame1': 50,
        #     'frame2': 50,
        #     'frame3': 50,
        #     'frame4': 50,
        #     'frame5': 50,
        #     'decision': 150
        # }
        self.timing = {
            'fixation': 50,
            "frame1": 200,
            "frame2": 200,
            "decision": 50
        }
        if timing:
            self.timing.update(timing)

        # Observation space
        self.ob_size = img_size  # H and W of each image
        self.tam_shape = (self.ob_size, self.ob_size)  # (H, W)
        self.observation_space = ngym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.tam_shape,
            dtype=np.float32,
        )

        # Action space
        # self.choice_names = {'no_motion': 0, 'left': 1, 'middle': 2, 'right': 3, 'down': 4, 'up': 5, 'fixation': 6}
        self.choice_names = {'fixation': 0, 'left': 1, 'middle': 2, 'right': 3, 'down': 4, 'up': 5}
        self.choices = list(self.choice_names.values())
        self.action_space = ngym.spaces.Discrete(6, name=self.choice_names)

    # Getter functions
    @property
    def box_shape(self):
        return self._box_shape

    @property
    def stimuli(self):
        return self._stimuli

    @property
    def stim_type(self):
        return self._stim_type

    @property
    def stim_ori(self):
        return self._stim_ori

    # Setter functions
    @box_shape.setter
    def box_shape(self, box_shape):
        self._box_shape = box_shape

    @stimuli.setter
    def stimuli(self, stimuli):
        self._stimuli = stimuli

    @stim_type.setter
    def stim_type(self, stim_type):
        self._stim_type = stim_type

    @stim_ori.setter
    def stim_ori(self, stim_ori):
        self._stim_ori = stim_ori

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            similarity: stimuli's similarity (degree of contours matching)
            obs:

        Returns
        -------
        dict : containing trial parameters
        """
        # Frames
        # Select stimuli
        if self._stim_ori == 'horizontal':
            trial_direction = np.random.choice([1, 2, 3])
            # trial_direction = np.random.choice([0, 1, 2, 3])
        else:
            trial_direction = np.random.choice([2, 4, 5])
            # trial_direction = np.random.choice([0, 2, 4, 5])
            for key, val in self._stimuli.items():
                for i, img in enumerate(val):
                    val[i] = np.rot90(img, k=-1)

        if trial_direction == 0:  # no motion
            motion_id = 'no_motion'
        elif trial_direction == 2:  # middle
            motion_id = 'middle'
        elif trial_direction in [3, 4]:  # right or down
            motion_id = 'right'
        elif trial_direction in [1, 5]:  # left or up
            motion_id = 'left'

        selected_stimuli = self._stimuli[f"{self._box_shape}-{self._stim_type}-{motion_id}"]

        # Set stimuli
        first_stim = self._stimuli[f"{self._box_shape}-{self._stim_type}-init"][0]
        frames = []
        # print(stimuli.keys())
        # print(f"{self._box_shape}-{self._stim_type}")
        # print(self._stimuli[f"{self._box_shape}-{self._stim_type}-init"])
        # for _ in range(1):
        #     frames.append(self._stimuli[f"{self._box_shape}-{self._stim_type}-init"][0])

        random_frame = random.choice(selected_stimuli)

        # for _ in range(4):
        #     frames.append(random_frame)

        # print(frames[0].shape)
        # Trial info
        trial = {
            'ground_truth': trial_direction,  # random condition
            'box_type': self._box_shape,
            'stim_ori': self._stim_ori,
            'stim_type': self._stim_type,
            "noise": self.sigma
        }
        trial.update(kwargs)  # allows wrapper to modify trial
        ground_truth = trial['ground_truth']

        # Add sequential periods
        # self.add_period(
        #     [
        #         'fixation',
        #         'frame1',
        #         'frame2',
        #         'frame3',
        #         'frame4',
        #         'frame5',
        #         'decision'
        #     ]
        # )
        self.add_period(
            [
                'fixation',
                'frame1',
                'frame2',
                'decision'
            ]
        )

        # Observations
        # fixation screen
        fix_size = 1
        fix = np.zeros(self.tam_shape)  # fixation

        # Fixation
        fix[
            (self.ob_size // 2 - fix_size):(self.ob_size // 2 + fix_size),
            (self.ob_size // 2 - fix_size):(self.ob_size // 2 + fix_size)
        ] = 1

        # Decision period
        decision_stim = np.zeros(self.tam_shape)

        # Add them to the observational environment
        self.add_ob(fix, period=['fixation'])
        # for i in range(len(frames)):
        #     self.add_ob(frames[i], period=['frame' + str(i + 1)])
        self.add_ob(first_stim, period=['frame1'])
        self.add_ob(random_frame, period=['frame2'])
        self.add_ob(decision_stim, period=['decision'])

        # Make some noise!
        # self.add_randn(0, self.sigma)
        # self.add_randn(0, self.sigma, period=['frame1', 'frame2', 'frame3', 'frame4', 'frame5'])
        self.add_randn(0, self.sigma, period=['frame1', 'frame2'])

        # Ground truth
        # self.set_groundtruth(0, period=['fixation', 'frame1'])
        self.set_groundtruth(0, period=['fixation', 'frame1'])
        # self.set_groundtruth(ground_truth, period=['frame2', 'frame3', 'frame4', 'frame5', 'decision'])
        self.set_groundtruth(ground_truth, period=['frame2', 'decision'])

        return trial

    def _make_second_order_horizontal_stim(
            self,
            tam_size: int,
            margin_size: int,
            direction: int,
            orientation: str
    ):
        """
        Make a second order stimulus with horizontal orientation

        Parameters
        ----------
        tam_size: int
            Size of the TAM

        margin_size: int
            Size of the margin between the TAM and the edge of the frame

        direction: int
            Ground truth of the trial. 0: no motion, -1: motion to the left, 1: motion to the right


        Returns
        -------
        stimuli: tuple
        """
        frame1_stim = (np.random.random(self.tam_shape[1:]) > 0.5).astype(np.int32)
        frame1_prime_stim = (np.random.random(self.tam_shape[1:]) > 0.5).astype(np.int32)
        frame2_stim = (np.random.random(self.tam_shape[1:]) > 0.5).astype(np.int32)
        frame2_prime_stim = (np.random.random(self.tam_shape[1:]) > 0.5).astype(np.int32)

        # left/top-side square
        frame1_stim[
        ((self.ob_size - tam_size) // 2): ((self.ob_size + tam_size) // 2),
        margin_size: (margin_size + tam_size)
        ] = 1

        # right/bottom-side square
        frame1_stim[
        ((self.ob_size - tam_size) // 2): ((self.ob_size + tam_size) // 2),
        (-margin_size - tam_size): (-margin_size)
        ] = 1

        # Connecting bar
        frame2_stim[
        ((self.ob_size - tam_size) // 2): ((self.ob_size + tam_size) // 2),
        margin_size: -margin_size
        ] = 1

        # Different conditions on the second frame to indicate different directions of motion
        # motion to the right/down: additional square at the right/bottom end
        if direction == 3:
            # Add extra square to the right/bottom-side stimulus
            frame2_stim[
            (((self.ob_size - tam_size) // 2) - tam_size): ((self.ob_size - tam_size) // 2),
            (-margin_size - tam_size): (-margin_size)
            ] = 1

        # motion to the left/up
        elif direction == 1:

            # Add extra square to the left/top-side stimulus
            frame2_stim[
            (((self.ob_size - tam_size) // 2) - tam_size): (self.ob_size - tam_size) // 2,
            margin_size: (margin_size + tam_size)
            ] = 1

        # Rotate the stimuli if needed
        if orientation == "vertical":
            frame1_stim = np.rot90(frame1_stim)
            frame2_stim = np.rot90(frame2_stim)

        # Add the extra dimension
        stimuli = (np.tile(frame1_stim, (3, 1, 1)), np.tile(frame2_stim, (3, 1, 1)))

        return stimuli

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has ended, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']

        Parameters
        ----------
        action : int
            The decision that the network made. Should be one of
            2 (fixating), -1 (left motion), 0 (middle motion), 1 (right motion)

        Returns
        -------
        tuple : (observation, reward, ?, dict {new_trial, ground_truth})
        """
        # Initiate new trial var
        new_trial = False

        # Rewards
        reward = 0
        gt = self.gt_now  # ground truth

        # Observations
        if self.in_period('fixation'):
            if action != 6:  # action = 6 means fixating

                # not fixating, so abort and gets no reward
                new_trial = self.abort
                reward += self.rewards['abort']

        elif self.in_period('decision'):
            if action != 6:  # took some action, not just fixating
                new_trial = True

                # correct response
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1

                # incorrect response
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class MotionTask(ngym.TrialEnv):
    """
    Creates an environment to train a network to see motion to the left, right, up, down, middle, or no motion at all.

    Each trial has the following structure: fixation -> first frame -> ... -> decision period

    The class inherits rewards and timing from neurogym TrialEnv.

    Parameters
    ----------
    stim_type: str
        Type of stimulus to be used. Can be 'tracking' or 'tam'

    sigma: float
        Input image's noise level

    img_size: int
        Size of the input image

    dt : int
        Number of steps in each trial that the stimulus is shown.

    cnn: bool
        Are the images going to pass through a CNN first?
    """
    metadata = {
        'description': "Generate motion task stimuli to teach an RNN to see motion",
        'paper_link': '',
        'paper_name': "",
        'tags': ['perceptual', 'supervised', 'Transformational Apparent Motion']
    }

    def __init__(
            self,
            dt=50,
            box_shape: str = None,
            motion_type: str = None,
            stim_ori: str = None,
            stimuli=None,
            sigma: float = 0.0,
            cnn: bool = False,
            img_size: int = 32,
            rewards=None,
            timing=None,
    ):

        # Initialize
        super().__init__(dt=dt)

        # Stimulus parameters
        self._box_shape = box_shape
        self._motion_type = motion_type
        self._stim_ori = stim_ori
        self._stimuli = stimuli
        self.sigma = sigma
        self.cnn = cnn

        # Reward parameters
        self.abort = False
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Timing parameters: total trial is 500ms
        self.dt = dt
        self.trial_dur = 500
        self.task_len = self.trial_dur // self.dt
        self.timing = {
            'fixation': 100,
            'frame1': 50,
            'frame2': 50,
            'frame3': 50,
            'frame4': 50,
            'frame5': 50,
            'decision': 150
        }
        if timing:
            self.timing.update(timing)

        # Observation space
        self.ob_size = img_size  # H and W of each image
        self.tam_shape = (self.ob_size, self.ob_size)  # (H, W)

        self.observation_space = ngym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.tam_shape,
            dtype=np.float32,
        )

        # Action space
        self.choice_names = {'no_motion': 0, 'left': 1, 'middle': 2, 'right': 3, 'down': 4, 'up': 5, 'fixation': 6}
        self.choices = list(self.choice_names.values())
        self.action_space = ngym.spaces.Discrete(7, name=self.choice_names)

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            similarity: stimuli's similarity (degree of contours matching)
            obs:

        Returns
        -------
        dict : containing trial parameters
        """
        # Frames
        # Select stimuli
        if self._stim_ori == 'horizontal':
            trial_direction = np.random.choice([0, 1, 2, 3])
        else:
            trial_direction = np.random.choice([0, 2, 4, 5])
            for key, val in self._stimuli.items():
                for s, stim_class in enumerate(val):
                    for i, img in enumerate(stim_class):
                        val[s][i] = np.rot90(img, k=-1)

        # Trial info
        trial = {
            'ground_truth': trial_direction,  # random condition
            'box_type': self._box_shape,
            'stim_ori': self._stim_ori,
            'motion_type': self._motion_type,
            "noise": self.sigma
        }
        trial.update(kwargs)  # allows wrapper to modify trial
        ground_truth = trial['ground_truth']

        if trial_direction == 0:  # no motion
            motion_id = 'no_motion'
        elif trial_direction == 2:  # middle
            motion_id = 'middle'
        elif trial_direction in [3, 4]:  # right or down
            motion_id = 'right'
        elif trial_direction in [1, 5]:  # left or up
            motion_id = 'left'

        if motion_id == "no_motion":
            random_dir = random.choice(["left", "right"])
            stim_class = self._stimuli[f"{self._box_shape}-{self._motion_type}-{random_dir}"]
            trial_stim = random.choice(stim_class)
            random_frame = np.random.randint(0, 5)
            frame = trial_stim[random_frame]
            frames = [frame] * 5
        else:
            stim_class = self._stimuli[f"{self._box_shape}-{self._motion_type}-{motion_id}"]
            frames = random.choice(stim_class)

        # fixation screen
        fix_size = 1
        fix = np.zeros(self.tam_shape)  # fixation

        # Fixation
        fix[
            (self.ob_size // 2 - fix_size):(self.ob_size // 2 + fix_size),
            (self.ob_size // 2 - fix_size):(self.ob_size // 2 + fix_size)
        ] = 1

        # Decision period
        decision_stim = np.zeros(self.tam_shape)

        # Add sequential periods
        self.add_period(period=list(self.timing.keys()), duration=list(self.timing.values()))

        # Add them to the observational environment
        self.add_ob(fix, period=['fixation'])
        for i in range(len(frames)):
            self.add_ob(frames[i], period=['frame' + str(i + 1)])
        self.add_ob(decision_stim, period=['decision'])

        # Make some noise!
        # self.add_randn(0, self.sigma)
        self.add_randn(0, self.sigma, period=['frame1', 'frame2', 'frame3', 'frame4', 'frame5'])

        # Ground truth
        self.set_groundtruth(6, period=['fixation', 'frame1', 'frame2', 'frame3', 'frame4', 'frame5'])
        self.set_groundtruth(ground_truth, period=['decision'])
        # self.set_groundtruth(6, period=['fixation'])
        # self.set_groundtruth(ground_truth, period=['frame1', 'frame2', 'frame3', 'frame4', 'frame5', 'decision'])

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has ended, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']

        Parameters
        ----------
        action : int
            The decision that the network made. Should be one of
            2 (fixating), -1 (left motion), 0 (middle motion), 1 (right motion)

        Returns
        -------
        tuple : (observation, reward, ?, dict {new_trial, ground_truth})
        """
        # Initiate new trial var
        new_trial = False

        # Rewards
        reward = 0
        gt = self.gt_now  # ground truth

        # Observations
        if self.in_period('fixation'):
            if action != 6:  # action = 2 means fixating

                # not fixating, so abort and gets no reward
                new_trial = self.abort
                reward += self.rewards['abort']

        elif self.in_period('decision'):
            if action != 6:  # took some action, not just fixating
                new_trial = True

                # correct response
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1

                # incorrect response
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


def visualize_environment(env, n_trials: int = 2, save_dir=None):
    """
    Visualizes an environment by plotting the images and the corresponding labels.

    Parameters
    ----------
    env : Environment
        Environment to be visualized.
    n_trials : int
        Number of trials to visualize.
    save_dir : str
        Directory to save the figure.

    Returns
    -------
    None
    """
    # Number of timesteps and trials
    n_timesteps = env.task_len

    # Figure
    ax_size = 12
    fig, axes = plt.subplots(
        n_trials,
        n_timesteps,
        figsize=(ax_size * n_timesteps, ax_size * n_trials)
    )

    # add space between subplots
    fig.subplots_adjust(hspace=0.3)
    # if env.NAME == 'TAM_TASK':
    #     fig.suptitle(f'{env.stim_type.capitalize()} TAM Stimulus')
    # else:
    #     fig.suptitle('Motion Stimulus')

    # Get images and labels
    env.reset(no_step=True)
    ob = env.ob[0]

    for tr in range(n_trials):
        for step in range(n_timesteps):

            _plot = sns.heatmap(
                ob,
                ax=axes[tr][step],
                cbar=False,
                cmap='gray'
            )

            _plot.set_title(
                f'Ground Truth: {env.gt_now} \n ' +
                f'Time in trial: {env.t}'
            )
            _plot.axis('off')

            # Step the environment
            ob, _, _, _ = env.step(action=6)

    if save_dir is not None:
        plt.savefig(save_dir)


def load_tam_from_image(box_size: int = 32, stim_type: str = "tam", stim_path: Path = None):
    """
    Load a stimulus from an image

    Returns
    -------
    dict
        Stimulus
    """
    stims = defaultdict()
    shapes = ['square', 'circle', 'triangle']
    motion_directions = ['no_motion', 'left', 'right', 'middle', 'init']
    for shape in shapes:
        for mt in motion_directions:
            stims[f'{shape}-{stim_type}-{mt}'] = []

    stim_files = list(stim_path.glob('*.jpg'))
    for sf in stim_files:
        base_name = '-'.join(sf.stem.split('-')[:-1])

        img = Image.open(sf)
        img = img.resize((box_size, box_size))
        img_arr = np.array(img)
        img_arr = img_arr[:, :, 0]
        img_arr = img_arr / 255
        img_arr = 1 - img_arr

        try:
            stims[base_name].append(img_arr)
        except KeyError:
            print(f'Could not find {sf} in stims')
            print(list(stims.keys()))
            break

    return stims


def load_motion_from_image(box_size: int = 32, stim_path: Path = None):
    """
    Load a stimulus from an image

    Returns
    -------
    dict
        Stimulus
    """
    stims = defaultdict()
    shapes = ['square', 'circle', 'triangle']
    stim_types = ['cnt', 'track']
    motion_directions = ['left', 'right', 'middle']
    stim_files = sorted(list(stim_path.glob('*.jpg')))

    for sp in shapes:
        for st in stim_types:
            for mt in motion_directions:

                cond_name = f'{sp}-{st}-{mt}'
                stims[cond_name] = []
                ns = []

                for stim_file in stim_files:
                    # print(cond_name, stim_file.stem)
                    if cond_name in stim_file.stem:
                        ns.append(int(stim_file.stem.split('-')[-2]))  # Add the number of the example frame series
                        # print(cond_name, stim_file.stem.split('-')[-2])

                # print(ns)
                # print(max(ns))
                for i in range(max(ns)):
                    stims[f'{cond_name}'].append([])

    for sf in stim_files:
        base_name = '-'.join(sf.stem.split('-')[:-2])
        idx = int(sf.stem.split('-')[-2]) - 1

        img = Image.open(sf)
        img = img.resize((box_size, box_size))
        img_arr = np.array(img)
        img_arr = img_arr[:, :, 0]
        img_arr = img_arr / 255
        img_arr = 1 - img_arr

        stims[base_name][idx].append(img_arr)

    return stims
