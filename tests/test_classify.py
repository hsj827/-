"""The region paths: what counts as a defect once a pixel map exists.

`classify` is where a score map becomes a verdict the operator sees, and until now nothing
tested it. Two of its rules were changed against measurements on a labelled benchmark, and
both were silent failures: a region rejected for touching the frame edge returns an empty
list, which looks identical to "no defect found".
"""
from __future__ import annotations

import numpy as np
import pytest

from adkit.postprocess.classify import _inside_band, classify
from adkit.postprocess.fuse import Fused


def fused(map_: np.ndarray) -> Fused:
    return Fused(map=map_.astype(np.float32), image_score=float(map_.max()),
                 is_defect=False, per_detector={"patchcore": float(map_.max())},
                 sources={"patchcore": map_.astype(np.float32)})


def block(shape, y0, x0, h, w, value, base=0.0):
    out = np.full(shape, base, np.float32)
    out[y0:y0 + h, x0:x0 + w] = value
    return out


def test_a_peak_only_cut_cannot_see_a_wide_faint_defect(config):
    """1.0 is the goods extreme, so extent is evidence the maximum is blind to.

    The spike is louder; the spill is the one a person would call a defect. With the peak cut
    above both, only the extent rule separates them, and it has to do so without promoting a
    small faint blob - noise comes in small faint doses too.
    """
    part = np.ones((100, 100), bool)
    cut, extent = 2.0, 0.02

    loud_spike = classify(fused(block((100, 100), 20, 20, 5, 5, 3.0, base=0.9)),
                          config, pixel_threshold=cut, template_mask=part, extent_cut=extent)
    wide_spill = classify(fused(block((100, 100), 20, 20, 40, 40, 1.3, base=0.9)),
                          config, pixel_threshold=cut, template_mask=part, extent_cut=extent)
    small_faint = classify(fused(block((100, 100), 20, 20, 6, 6, 1.3, base=0.9)),
                           config, pixel_threshold=cut, template_mask=part, extent_cut=extent)

    assert len(loud_spike) == 1, "a peak past the cut is still a defect"
    assert len(wide_spill) == 1, "a fourth of the part at 1.3x the goods extreme is a defect"
    assert small_faint == [], "a faint blob nobody would ship a stop-line for stays out"
    assert wide_spill[0].mass > loud_spike[0].mass * 4


def test_the_extent_rule_off_when_no_floor_was_measured(config):
    """Without a measured floor the second rule must not silently invent defects."""
    part = np.ones((100, 100), bool)
    map_ = block((100, 100), 20, 20, 40, 40, 1.3, base=0.9)
    assert classify(fused(map_), config, pixel_threshold=2.0, template_mask=part) == []


def test_only_a_sliver_confined_to_the_band_is_a_cropping_seam(config):
    """The old test rejected anything touching the frame, which is everything on a part that
    fills the canvas: it returned zero regions on all 63 labelled defects of a benchmark."""
    part = np.ones((100, 100), bool)
    big_at_edge = classify(fused(block((100, 100), 20, 0, 40, 40, 3.0, base=0.9)),
                           config, pixel_threshold=2.0, template_mask=part, extent_cut=0.0)
    sliver_at_edge = classify(fused(block((100, 100), 20, 0, 40, 2, 3.0, base=0.9)),
                              config, pixel_threshold=2.0, template_mask=part, extent_cut=0.0)
    assert len(big_at_edge) == 1, "a defect the part reaches to the frame edge is a defect"
    assert sliver_at_edge == [], "a component entirely inside the band is the mask seam"
    assert _inside_band((0, 20, 2, 40), 2, 100, 100)
    assert not _inside_band((0, 20, 40, 40), 2, 100, 100)


def test_nearby_speckles_are_measured_as_one_defect(config):
    """A spill is speckled, and every speckle is under both floors on its own."""
    part = np.ones((100, 100), bool)
    speckled = block((100, 100), 30, 30, 21, 21, 0.9)
    for y in range(30, 50, 3):          # dots 3px apart, each 2x2 at 1.3 -> mass 4*0.3/part
        for x in range(30, 50, 3):
            speckled[y:y + 2, x:x + 2] = 1.3
    merged = classify(fused(speckled), config, pixel_threshold=2.0, template_mask=part,
                      extent_cut=0.005)
    single = classify(fused(speckled), config, pixel_threshold=2.0, template_mask=part,
                      extent_cut=0.0)   # extent rule off -> only a peak can pass
    assert len(merged) == 1, "the dots are one defect once connected"
    assert single == [], "and nothing at all while each dot is judged alone"


def test_regions_are_reported_with_the_mask_that_justifies_the_box(config):
    """The box may grow to cover connected evidence; the mask must not cover the halo."""
    part = np.ones((100, 100), bool)
    found = classify(fused(block((100, 100), 30, 30, 20, 20, 3.0, base=0.9)),
                     config, pixel_threshold=2.0, template_mask=part)
    defect = found[0]
    assert "patchcore" in defect.sources, "the channel that produced it has to be named"
    assert defect.mask_area == pytest.approx(400, rel=0.05),         f"dilation halo counted as evidence: {defect.mask_area} of a {defect.mask.shape} box"
