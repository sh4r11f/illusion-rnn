import numpy as np
import pytest

from illusion_rnn.generate import (
    SHAPE_NAMES, erode, ink_mass, outline_mask, shape_mask,
)


@pytest.mark.parametrize("shape", SHAPE_NAMES)
def test_shape_mask_is_nonempty_and_fits_its_box(shape):
    m = shape_mask(shape, 16)
    assert m.shape == (16, 16)
    assert m.dtype == bool
    assert m.any(), f"{shape} rendered empty"
    assert not m.all() or shape == "square", f"{shape} filled its whole box"


@pytest.mark.parametrize("shape", SHAPE_NAMES)
def test_shape_mask_is_left_right_symmetric(shape):
    """Every shape is symmetric about the vertical axis, so shape identity
    cannot leak the left/right direction label."""
    m = shape_mask(shape, 16)
    np.testing.assert_array_equal(m, m[:, ::-1])


def test_shapes_are_distinguishable_from_each_other():
    masks = {s: shape_mask(s, 24) for s in SHAPE_NAMES}
    for a in SHAPE_NAMES:
        for b in SHAPE_NAMES:
            if a < b:
                assert not np.array_equal(masks[a], masks[b]), f"{a} == {b}"


def test_shape_mask_rejects_unknown_shape():
    with pytest.raises(ValueError, match="Unknown shape 'blob'"):
        shape_mask("blob", 16)


def test_erode_shrinks_a_solid_block_by_the_radius():
    m = np.zeros((11, 11), dtype=bool)
    m[2:9, 2:9] = True          # 7x7 block
    e = erode(m, 1)
    assert e.sum() == 25        # shrinks to 5x5
    assert e[3:8, 3:8].all()


def test_erode_removes_a_block_thinner_than_the_radius():
    """Erosion must not silently keep pixels it cannot support."""
    m = np.zeros((11, 11), dtype=bool)
    m[5, 2:9] = True            # 1px-tall line
    assert not erode(m, 1).any()


def test_outline_mask_is_hollow_and_inside_the_original():
    m = shape_mask("square", 20)
    o = outline_mask(m, stroke_width=2)
    assert o.sum() < m.sum(), "outline must be lighter than the filled shape"
    assert (m | o == m).all(), "outline must lie inside the filled shape"
    assert not o[8:12, 8:12].any(), "centre must be hollow"


def test_outline_stroke_width_monotonically_adds_ink():
    m = shape_mask("circle", 32)
    masses = [outline_mask(m, w).sum() for w in (1, 2, 3, 4)]
    assert masses == sorted(masses), "thicker strokes must not lose ink"


def test_ink_mass_sums_pixel_values():
    frame = np.zeros((4, 4))
    frame[0, :2] = 0.5
    assert ink_mass(frame) == pytest.approx(1.0)


from illusion_rnn.generate import DIRECTIONS, TrialStimulus, render_trial, sample_params


def _params(**overrides):
    base = dict(
        shape="square", shape_size=8, bar_left=10, bar_length=24,
        direction="right", family="balanced", transform="growth",
        render="filled", stroke_width=1, ink_match="none",
    )
    base.update(overrides)
    return base


def test_frame2_is_bit_identical_across_directions():
    """THE load-bearing invariant of the design.

    Frame 2 is a function of (bar_left, bar_length, shape_size) only. If this
    ever fails, the frame-2-only baseline is above chance and the whole
    experiment is void.
    """
    right = render_trial(_params(direction="right"))
    left = render_trial(_params(direction="left"))
    np.testing.assert_array_equal(right.frame2, left.frame2)


def test_frame1_differs_across_directions():
    right = render_trial(_params(direction="right"))
    left = render_trial(_params(direction="left"))
    assert not np.array_equal(right.frame1, left.frame1)


def test_frame1_sits_at_the_end_the_bar_grew_from():
    """Rightward growth starts at the bar's left edge, and vice versa."""
    right = render_trial(_params(direction="right"))
    left = render_trial(_params(direction="left"))
    right_cols = np.flatnonzero(right.frame1.any(axis=0))
    left_cols = np.flatnonzero(left.frame1.any(axis=0))
    assert right_cols.min() == 10                  # bar_left
    assert left_cols.max() == 10 + 24 - 1          # bar_left + bar_length - 1


def test_frame1_lies_inside_the_frame2_bar():
    """The shape is absorbed into the bar -- that is the transformation."""
    t = render_trial(_params())
    assert ((t.frame1 > 0) & (t.frame2 == 0)).sum() == 0


def test_frames_respect_the_ink_convention():
    t = render_trial(_params())
    for frame in (t.frame1, t.frame2):
        assert frame.dtype == np.float64
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0
        assert frame.shape == (64, 64)


def test_label_matches_the_requested_direction():
    assert render_trial(_params(direction="left")).label == "left"
    assert render_trial(_params(direction="right")).label == "right"


def test_render_trial_rejects_a_bar_that_runs_off_canvas():
    with pytest.raises(ValueError, match="does not fit"):
        render_trial(_params(bar_left=50, bar_length=30), img_size=64)


def test_render_trial_rejects_a_shape_longer_than_its_bar():
    with pytest.raises(ValueError, match="shape_size"):
        render_trial(_params(shape_size=30, bar_length=24))


def test_sample_params_draws_frame2_geometry_independently_of_direction():
    """Statistical form of the invariant: across many samples, the bar geometry
    distribution must not differ by label."""
    rng = np.random.default_rng(0)
    rows = [sample_params(rng) for _ in range(4000)]
    left = np.array([(r["bar_left"], r["bar_length"]) for r in rows
                     if r["direction"] == "left"], dtype=float)
    right = np.array([(r["bar_left"], r["bar_length"]) for r in rows
                      if r["direction"] == "right"], dtype=float)
    # means must agree to well within sampling noise
    assert np.allclose(left.mean(axis=0), right.mean(axis=0), rtol=0.05)


def test_sample_params_is_reproducible_from_a_seed():
    a = [sample_params(np.random.default_rng(7)) for _ in range(3)]
    b = [sample_params(np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_outline_render_has_less_ink_than_filled():
    filled = render_trial(_params(render="filled"))
    outline = render_trial(_params(render="outline", stroke_width=1))
    assert ink_mass(outline.frame1) < ink_mass(filled.frame1)


def test_energy_matching_equalizes_ink_mass():
    filled = render_trial(_params(render="filled"))
    matched = render_trial(_params(render="outline", ink_match="energy"))
    assert ink_mass(matched.frame1) == pytest.approx(ink_mass(filled.frame1), rel=1e-9)


def test_render_trial_rejects_an_unknown_transform():
    """Nothing should fail silently: a bogus transform must not fall through
    to growth-like behaviour, it must raise."""
    with pytest.raises(ValueError, match="Unknown transform 'bogus-nonsense'"):
        render_trial(_params(transform="bogus-nonsense"))


def test_render_trial_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="Unknown family 'bogus-nonsense'"):
        render_trial(_params(family="bogus-nonsense"))


def test_classic_family_frame2_IS_label_readable():
    """The diagnostic family deliberately leaks -- that is its purpose.

    It reproduces the 2023 geometry where a raised end marks the direction, so
    a frame-2-only model can solve it. Comparing model behaviour across the two
    families is what quantifies the original defect.
    """
    right = render_trial(_params(family="classic", direction="right"))
    left = render_trial(_params(family="classic", direction="left"))
    assert not np.array_equal(right.frame2, left.frame2)


def test_classic_family_frame1_is_constant_across_directions():
    """Matching the 2023 stimulus: frame 1 is two small squares either way."""
    right = render_trial(_params(family="classic", direction="right"))
    left = render_trial(_params(family="classic", direction="left"))
    np.testing.assert_array_equal(right.frame1, left.frame1)


def test_classic_raised_end_is_on_the_labelled_side():
    right = render_trial(_params(family="classic", direction="right"))
    heights = (right.frame2 > 0).sum(axis=0)
    cols = np.flatnonzero(heights)
    assert heights[cols[-1]] > heights[cols[len(cols) // 2]], \
        "rightward trials must raise the right end"


def test_shrink_transform_swaps_the_two_frames():
    grow = render_trial(_params(transform="growth", direction="right"))
    shrink = render_trial(_params(transform="shrink", direction="left"))
    np.testing.assert_array_equal(grow.frame1, shrink.frame2)
    np.testing.assert_array_equal(grow.frame2, shrink.frame1)


def test_growth_plus_shrink_makes_order_load_bearing():
    """A growth-right trial and a shrink-left trial share IDENTICAL frame
    content in reversed temporal order, with opposite labels -- so a model
    that ignores frame order cannot distinguish them, and must be at chance."""
    grow = render_trial(_params(transform="growth", direction="right"))
    shrink = render_trial(_params(transform="shrink", direction="left"))
    np.testing.assert_array_equal(grow.frame1, shrink.frame2)
    np.testing.assert_array_equal(grow.frame2, shrink.frame1)
    assert grow.label != shrink.label


def test_render_trial_rejects_classic_overlapping_end_squares():
    """classic's two end squares must not overlap -- a short bar relative to
    the shape size would silently merge them instead of raising, which
    violates the project's 'nothing fails silently' rule."""
    with pytest.raises(ValueError, match="classic family needs bar_length"):
        render_trial(
            _params(family="classic", shape_size=8, bar_length=10, bar_left=10),
        )


def test_sample_params_growth_plus_shrink_draws_both_transforms():
    rng = np.random.default_rng(0)
    rows = [sample_params(rng, transform="growth+shrink") for _ in range(500)]
    seen = {r["transform"] for r in rows}
    assert seen == {"growth", "shrink"}


def test_sample_params_rejects_unknown_transform():
    with pytest.raises(ValueError, match="Unknown transform"):
        sample_params(np.random.default_rng(0), transform="teleport")
