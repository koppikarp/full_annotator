"""Unit tests for the color calling module."""

import numpy as np
import pytest

from popcrop.color_calling import (
    compute_core_rim_geometry,
    analyze_spatial_colocalization,
    CoreAwareColorCaller,
)


def test_core_rim_geometry():
    # 20x20 square blob
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True

    weights, core_mask, rim_mask, max_dist = compute_core_rim_geometry(mask)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    # Core should be inside rim
    assert np.sum(core_mask) > 0
    assert np.sum(rim_mask) > 0
    assert np.sum(core_mask & rim_mask) == 0
    assert np.sum(core_mask | rim_mask) == np.sum(mask)
    # Deepest point should be near center
    assert weights[20, 20] == pytest.approx(1.0, abs=0.1)


def test_color_caller_single_colors():
    caller = CoreAwareColorCaller()
    T, C, H, W = 4, 2, 30, 30
    stack = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    bg = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    noise = np.ones((T, C, H, W), dtype=np.float32) * 10.0

    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 10:20] = True  # 10x10 blob

    # Cycle 0: Dark (both channels low)
    # Cycle 1: Green only (ch 1 = 500, ch 0 = 100) -> A
    stack[1, 1, 10:20, 10:20] = 500.0
    # Cycle 2: Red only (ch 0 = 500, ch 1 = 100) -> T
    stack[2, 0, 10:20, 10:20] = 500.0
    # Cycle 3: Yellow (ch 0 = 400, ch 1 = 600) -> C
    stack[3, 0, 10:20, 10:20] = 400.0
    stack[3, 1, 10:20, 10:20] = 600.0

    z_scores = np.maximum(0.0, (stack - bg) / noise)

    probs = caller.call_blob_cycle_probabilities(
        sbs_stack=stack,
        background_maps=bg,
        noise_maps=noise,
        z_scores=z_scores,
        component_mask=mask,
        p_real_blob=1.0
    )

    assert len(probs) == 4
    # Cycle 0: Dark
    assert probs[0]["dark"] > 0.80
    # Cycle 1: Green
    assert probs[1]["green"] > 0.85
    # Cycle 2: Red
    assert probs[2]["red"] > 0.85
    # Cycle 3: Yellow
    assert probs[3]["yellow"] > 0.85
