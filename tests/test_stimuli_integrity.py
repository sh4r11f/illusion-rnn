"""Guards against the variant-contamination defect found on 2026-08-15.

TAM_basic was shipped and documented as a held-out generalization variant, but
every one of its 24 images is a byte-identical copy of a TAM_task image, so
"transfer to basic = 0.967" was measured on training data. These tests record
which variant pairs are genuinely disjoint so the mistake cannot recur silently.
"""
import hashlib
from pathlib import Path

import pytest

from illusion_rnn.stimuli import _default_source


def _hashes(dirname: str) -> set[str]:
    src = _default_source(dirname)
    return {
        hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(Path(src).glob("*.jpg"))
    }


def test_outline_is_genuinely_disjoint_from_standard():
    """outline IS a real held-out variant -- no shared images."""
    assert not (_hashes("TAM_outline") & _hashes("TAM_task"))


def test_basic_is_documented_as_a_subset_not_a_variant():
    """basic is NOT held out: every image also appears in the standard set.

    This assertion is deliberately phrased as the true state of the data. If
    someone regenerates TAM_basic as genuinely novel images, this test fails and
    forces the documentation to be revisited.
    """
    basic, standard = _hashes("TAM_basic"), _hashes("TAM_task")
    assert basic <= standard, (
        "TAM_basic is expected to be a strict subset of TAM_task; if it is no "
        "longer, README/MANIFEST claims about it must be re-checked"
    )
