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
