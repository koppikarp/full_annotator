"""Unit tests for the blob detection module."""

import numpy as np
import pytest

from popcrop.blob_detection import (
    compute_zscore_and_energy,
    aggregate_cross_cycle_signal,
    segment_extended_blobs,
)


def test_zscore_and_energy():
    T, C, H, W = 2, 2, 32, 32
    stack = np.ones((T, C, H, W), dtype=np.float32) * 200.0
    bg = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    noise = np.ones((T, C, H, W), dtype=np.float32) * 10.0

    # Put high signal in ch 0 at (16, 16)
    stack[0, 0, 16, 16] = 500.0  # excess = 400, z = 40
    # Put signal in ch 1 at (16, 16)
    stack[0, 1, 16, 16] = 400.0  # excess = 300, z = 30

    z, energy = compute_zscore_and_energy(stack, bg, noise)
    assert z[0, 0, 16, 16] == pytest.approx(40.0, abs=0.1)
    assert z[0, 1, 16, 16] == pytest.approx(30.0, abs=0.1)
    assert energy[0, 16, 16] == pytest.approx(50.0, abs=0.1)


def test_cross_cycle_persistence():
    T, H, W = 4, 32, 32
    energies = np.zeros((T, H, W), dtype=np.float32)
    energies[0, 10, 10] = 20.0
    energies[1, 10, 10] = 25.0
    energies[2, 10, 10] = 15.0
    energies[3, 10, 10] = 0.5  # dark cycle

    top_k, persist, metric = aggregate_cross_cycle_signal(energies, top_k=2, persistence_threshold=2.5)
    assert top_k[10, 10] == pytest.approx(22.5, abs=0.1)
    assert persist[10, 10] == 3
    assert metric[10, 10] == pytest.approx(45.0, abs=0.1)


def test_hysteresis_segmenter():
    H, W = 64, 64
    nuclear_mask = np.zeros((H, W), dtype=np.int32)
    y, x = np.ogrid[:H, :W]
    nuclear_mask[(y - 32)**2 + (x - 32)**2 <= 15**2] = 1

    metric = np.zeros((H, W), dtype=np.float32)
    # Plateau around it first
    metric[25:35, 25:35] = 3.0
    # Seed at (30, 30) second
    metric[28:32, 28:32] = 6.0

    blob_mask, prob_map = segment_extended_blobs(
        combined_metric=metric,
        nuclear_mask=nuclear_mask,
        hysteresis_high=4.0,
        hysteresis_low=2.0,
        min_blob_area=5
    )

    assert blob_mask.max() == 1
    # Full plateau (10x10=100 pixels) should be segmented
    assert np.sum(blob_mask == 1) == 100
    assert prob_map[30, 30] > 0.95
