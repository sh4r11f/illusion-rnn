import numpy as np
import pytest

from illusion_rnn.envs import MOTION_CHOICES, MotionTask, TAM_CHOICES, TAMTask, rotate_stimuli
from illusion_rnn.stimuli import MOTION_TYPES, SHAPES, TAM_VARIANTS


def _make_tam(**kwargs):
    defaults = dict(box_shape="square", variant="standard",
                    stim_ori="horizontal", img_size=32)
    defaults.update(kwargs)
    env = TAMTask(**defaults)
    env.seed(0)
    return env


@pytest.mark.parametrize("variant", TAM_VARIANTS)
@pytest.mark.parametrize("stim_ori", ["horizontal", "vertical"])
@pytest.mark.parametrize("shape", SHAPES)
def test_tam_grid(variant, stim_ori, shape):
    env = _make_tam(variant=variant, stim_ori=stim_ori, box_shape=shape)
    ob, _ = env.reset()
    # fixation 100 + 5 frames x 50 + decision 100 = 450 ms; dt 50 -> 9 steps
    assert env.ob.shape == (9, 32, 32)
    assert env.gt.shape == (9,)
    assert env.ob.dtype == np.float32
    # first 3 steps (fixation x2 + frame1) are gt=0; rest are the direction
    assert set(env.gt[:3]) == {TAM_CHOICES["fixation"]}
    allowed = (
        {TAM_CHOICES["left"], TAM_CHOICES["middle"], TAM_CHOICES["right"]}
        if stim_ori == "horizontal"
        else {TAM_CHOICES["middle"], TAM_CHOICES["down"], TAM_CHOICES["up"]}
    )
    assert set(env.gt[3:]) <= allowed
    assert len(set(env.gt[3:])) == 1


def test_action_space_and_labels_frozen():
    env = _make_tam()
    assert env.action_space.n == 6
    assert env.choice_names == {
        "fixation": 0, "left": 1, "middle": 2, "right": 3, "down": 4, "up": 5,
    }


def test_seeded_trials_reproduce():
    a = _make_tam()
    b = _make_tam()
    a.seed(123)
    b.seed(123)
    a.reset()
    b.reset()
    for _ in range(3):
        ta = a.new_trial()
        tb = b.new_trial()
        assert ta["ground_truth"] == tb["ground_truth"]
        np.testing.assert_array_equal(a.ob, b.ob)


def test_direction_variability():
    env = _make_tam()
    env.reset()
    gts = {env.new_trial()["ground_truth"] for _ in range(50)}
    assert gts == {1, 2, 3}


def test_vertical_stimuli_do_not_mutate():
    # Regression: old code rotated the stimulus dict in place every trial.
    env = _make_tam(stim_ori="vertical")
    env.reset()
    before = env._stimuli["square-tam-init"][0].copy()
    for _ in range(5):
        env.new_trial()
    np.testing.assert_array_equal(env._stimuli["square-tam-init"][0], before)


def test_vertical_rotates_once():
    h = _make_tam(stim_ori="horizontal")
    v = _make_tam(stim_ori="vertical")
    np.testing.assert_array_equal(
        v._stimuli["square-tam-init"][0],
        np.rot90(h._stimuli["square-tam-init"][0], k=-1),
    )


def test_rotate_stimuli_nested():
    stims = {"a": [np.eye(3)], "b": [[np.eye(3), np.ones((3, 3))]]}
    out = rotate_stimuli(stims)
    np.testing.assert_array_equal(out["a"][0], np.rot90(np.eye(3), k=-1))
    np.testing.assert_array_equal(out["b"][0][1], np.ones((3, 3)))
    # original untouched
    np.testing.assert_array_equal(stims["a"][0], np.eye(3))


def test_step_contract_and_fixation_index():
    # neurogym 2.x reset() consumes the trial's first timestep internally,
    # so exactly one fixation step remains before frame1.
    env = _make_tam()
    env.reset()
    # Fixating (action 0) during the fixation period must NOT be punished.
    out = env.step(TAM_CHOICES["fixation"])
    assert len(out) == 5
    _, reward, terminated, truncated, info = out
    assert reward == 0.0
    assert terminated is False and truncated is False
    # Fresh trial: breaking fixation (any non-fixation action) on the
    # remaining fixation step draws the abort penalty.
    env.reset()
    _, reward, _, _, _ = env.step(TAM_CHOICES["left"])
    assert reward == pytest.approx(env.rewards["abort"])


def test_decision_reward():
    env = _make_tam()
    env.reset()
    gt_final = int(env.gt[-1])
    for _ in range(7):  # t1..t7: remaining fixation step, frames 1-5, first decision step
        env.step(TAM_CHOICES["fixation"])
    _, reward, _, _, info = env.step(gt_final)  # t8: final decision step
    assert reward == pytest.approx(env.rewards["correct"])
    assert info["new_trial"] is True


def test_custom_stimuli_override():
    frames = {
        f"{s}-tam-{m}": [np.zeros((32, 32))]
        for s in SHAPES
        for m in ("no_motion", "left", "right", "middle", "init")
    }
    env = TAMTask(box_shape="square", variant="standard",
                  stim_ori="horizontal", stimuli=frames, img_size=32)
    env.seed(0)
    env.reset()
    assert env.ob.shape == (9, 32, 32)


def test_invalid_args_raise():
    with pytest.raises(ValueError, match="box_shape"):
        TAMTask(box_shape="hexagon")
    with pytest.raises(ValueError, match="stim_ori"):
        TAMTask(stim_ori="diagonal")
    with pytest.raises(ValueError, match="variant"):
        TAMTask(variant="bogus")


def _make_motion(**kwargs):
    defaults = dict(box_shape="square", motion_type="continuous",
                    stim_ori="horizontal", img_size=32)
    defaults.update(kwargs)
    env = MotionTask(**defaults)
    env.seed(0)
    return env


@pytest.mark.parametrize("motion_type", MOTION_TYPES)
@pytest.mark.parametrize("stim_ori", ["horizontal", "vertical"])
@pytest.mark.parametrize("shape", SHAPES)
def test_motion_grid(motion_type, stim_ori, shape):
    env = _make_motion(motion_type=motion_type, stim_ori=stim_ori, box_shape=shape)
    env.reset()
    # fixation 100 + 5 frames x 50 + decision 150 = 500 ms; dt 50 -> 10 steps
    assert env.ob.shape == (10, 32, 32)
    assert env.gt.shape == (10,)
    # gt is fixation (6) everywhere except the 3 decision steps
    assert set(env.gt[:7]) == {MOTION_CHOICES["fixation"]}
    allowed = {0, 1, 2, 3} if stim_ori == "horizontal" else {0, 2, 4, 5}
    assert set(env.gt[7:]) <= allowed
    assert len(set(env.gt[7:])) == 1


def test_motion_action_space_frozen():
    env = _make_motion()
    assert env.action_space.n == 7
    assert env.choice_names == {
        "no_motion": 0, "left": 1, "middle": 2, "right": 3,
        "down": 4, "up": 5, "fixation": 6,
    }


def test_motion_direction_variability():
    env = _make_motion()
    env.reset()
    gts = {env.new_trial()["ground_truth"] for _ in range(80)}
    assert gts == {0, 1, 2, 3}


def test_no_motion_trials_are_static():
    env = _make_motion()
    env.reset()
    for _ in range(80):
        trial = env.new_trial()
        if trial["ground_truth"] == MOTION_CHOICES["no_motion"]:
            # frames 1-5 are steps 2..6 of the ob; all identical
            frames = env.ob[2:7]
            for i in range(1, 5):
                np.testing.assert_array_equal(frames[i], frames[0])
            break
    else:
        pytest.fail("no no_motion trial sampled in 80 draws")


def test_motion_trials_change_frames():
    env = _make_motion()
    env.reset()
    for _ in range(80):
        trial = env.new_trial()
        if trial["ground_truth"] != MOTION_CHOICES["no_motion"]:
            frames = env.ob[2:7]
            assert any(
                not np.array_equal(frames[i], frames[0]) for i in range(1, 5)
            )
            break
    else:
        pytest.fail("no motion trial sampled in 80 draws")


def test_motion_step_fixation_index():
    # neurogym 2.x reset() consumes the trial's first timestep internally,
    # so exactly one fixation step remains before frame1.
    env = _make_motion()
    env.reset()
    out = env.step(MOTION_CHOICES["fixation"])
    assert len(out) == 5
    assert out[1] == 0.0  # fixating during fixation: no penalty
    env.reset()
    _, reward, _, _, _ = env.step(MOTION_CHOICES["left"])
    assert reward == pytest.approx(env.rewards["abort"])


def test_motion_vertical_stimuli_do_not_mutate():
    env = _make_motion(stim_ori="vertical")
    env.reset()
    before = env._stimuli["square-cnt-left"][0][0].copy()
    for _ in range(5):
        env.new_trial()
    np.testing.assert_array_equal(env._stimuli["square-cnt-left"][0][0], before)


from illusion_rnn.envs import TAMCorrespondenceTask


def _make_corr(**kwargs):
    defaults = dict(split="train", img_size=32)
    defaults.update(kwargs)
    env = TAMCorrespondenceTask(**defaults)
    env.seed(0)
    return env


def test_correspondence_trial_structure():
    """fixation 100 + frame1 50 + frame2 4x50 + decision 100 = 450ms at dt=50."""
    env = _make_corr()
    env.reset()
    assert env.ob.shape == (9, 32, 32)
    assert env.gt.shape == (9,)
    # fixation x2 + frame1 carry the fixation label; direction from frame2 on
    assert set(env.gt[:3]) == {TAM_CHOICES["fixation"]}
    assert set(env.gt[3:]) <= {TAM_CHOICES["left"], TAM_CHOICES["right"]}
    assert len(set(env.gt[3:])) == 1


def test_correspondence_action_space_is_the_frozen_six():
    env = _make_corr()
    assert env.action_space.n == 6
    assert env.choice_names == TAM_CHOICES


def test_frame_indices_locate_the_two_frames_in_the_observation():
    env = _make_corr()
    env.reset()
    f1 = env.ob[env.frame1_index]
    f2s = [env.ob[i] for i in env.frame2_indices]
    assert f1.any(), "frame1 index points at an empty observation"
    for f2 in f2s:
        np.testing.assert_array_equal(f2, f2s[0])   # frame2 is repeated
    assert not np.array_equal(f1, f2s[0])


def test_frame1_and_frame2_always_differ():
    env = _make_corr()
    for _ in range(50):
        env.new_trial()
        assert not np.array_equal(
            env.ob[env.frame1_index], env.ob[env.frame2_indices[0]],
        )


def test_correspondence_is_reproducible_from_a_seed():
    a, b = _make_corr(), _make_corr()
    for _ in range(10):
        a.new_trial(); b.new_trial()
        np.testing.assert_array_equal(a.ob, b.ob)
        np.testing.assert_array_equal(a.gt, b.gt)


def test_correspondence_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="Unknown split"):
        TAMCorrespondenceTask(split="nope")


@pytest.mark.parametrize("split", ["train", "test_position", "test_shape", "test_style"])
def test_every_split_builds_an_env_that_runs(split):
    env = _make_corr(split=split)
    for _ in range(20):
        env.new_trial()
    assert env.ob.shape == (9, 32, 32)
