import numpy as np
import pytest

from illusion_rnn.stimuli import (
    MOTION_DIRECTIONS,
    SHAPES,
    load_motion,
    load_tam,
)


@pytest.mark.parametrize(
    "variant,prefix",
    [("standard", "tam"), ("basic", "tam"), ("outline", "outline")],
)
def test_load_tam_variants(variant, prefix):
    stims = load_tam(24, variant=variant)
    for key, frames in stims.items():
        for frame in frames:
            assert frame.shape == (24, 24)
            assert frame.dtype == np.float64
            assert 0.0 <= frame.min() and frame.max() <= 1.0
    for shape in SHAPES:
        for motion in ("init", "left", "right", "middle"):
            assert len(stims[f"{shape}-{prefix}-{motion}"]) >= 1


@pytest.mark.parametrize(
    "motion_type,prefix",
    [("continuous", "cnt"), ("tracking", "track")],
)
def test_load_motion_types(motion_type, prefix):
    stims = load_motion(24, motion_type=motion_type)
    for shape in SHAPES:
        for direction in MOTION_DIRECTIONS:
            key = f"{shape}-{prefix}-{direction}"
            assert key in stims
            assert len(stims[key]) >= 1
            for exemplar in stims[key]:
                assert len(exemplar) == 5
                for frame in exemplar:
                    assert frame.shape == (24, 24)


def test_ink_is_one():
    # Stimulus JPGs are dark ink on white; loader inverts so ink ~= 1.
    stims = load_tam(48, variant="standard")
    frame = stims["square-tam-init"][0]
    # background (corner) should be near 0, some ink pixels near 1
    assert frame[0, 0] < 0.2
    assert frame.max() > 0.7


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        load_tam(24, variant="bogus")
    with pytest.raises(ValueError, match="motion_type"):
        load_motion(24, motion_type="bogus")


def test_empty_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .jpg stimuli"):
        load_tam(24, source=tmp_path)


def test_bad_filename_raises(tmp_path):
    (tmp_path / "bogus-file.jpg").write_bytes(b"junk")
    with pytest.raises(ValueError, match="expected"):
        load_tam(24, source=tmp_path)
