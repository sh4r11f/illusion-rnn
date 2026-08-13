import numpy as np
import pytest

from illusion_rnn.envs import TAM_CHOICES, TAMTask, rotate_stimuli
from illusion_rnn.stimuli import SHAPES, TAM_VARIANTS


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
