"""Unit tests for the preprocessing module."""

import numpy as np
import pytest

from popcrop.preprocessing import (
    subtract_camera_offset,
    correct_illumination_stack,
    estimate_illumination_profile,
    register_cycles,
    correct_chromatic_shift,
    match_psf,
    unmix_spectral_crosstalk,
    estimate_background_and_noise,
)


def test_subtract_camera_offset():
    img = np.array([[50.0, 150.0], [200.0, 80.0]], dtype=np.float32)
    res = subtract_camera_offset(img, camera_offset=100.0)
    assert res[0, 0] == 0.0
    assert res[0, 1] == 50.0
    assert res[1, 0] == 100.0
    assert res[1, 1] == 0.0


def test_field_flattening_per_cycle_and_well():
    # 2 cycles, 2 channels, 128x128
    T, C, H, W = 2, 2, 128, 128
    y, x = np.mgrid[:H, :W]
    # Synthetic vignetting pattern (brighter in center, darker at edges)
    vignette = 1.0 - 0.5 * ((y - H/2)**2 + (x - W/2)**2) / (H/2)**2
    
    stack = np.zeros((T, C, H, W), dtype=np.float32)
    for t in range(T):
        for c in range(C):
            stack[t, c] = 500.0 * vignette + np.random.normal(0, 5, (H, W))

    flattened, profiles = correct_illumination_stack(
        stack,
        well_id="Well_B1",
        do_field_flattening=True,
        mode="retrospective",
        per_cycle=True,
        per_well=True,
        filter_size=32
    )

    assert flattened.shape == stack.shape
    assert profiles.shape == stack.shape
    # Center-to-edge ratio should be much flatter after correction
    raw_ratio = np.mean(stack[:, :, 64, 64]) / np.mean(stack[:, :, 10, 10])
    corr_ratio = np.mean(flattened[:, :, 64, 64]) / np.mean(flattened[:, :, 10, 10])
    assert abs(corr_ratio - 1.0) < abs(raw_ratio - 1.0)


def test_cycle_registration():
    # 3 cycles, 2 channels, 64x64
    T, C, H, W = 3, 2, 64, 64
    stack = np.zeros((T, C, H, W), dtype=np.float32)
    
    # Create distinct nuclear pattern at (32, 32)
    y, x = np.ogrid[:H, :W]
    pattern = ((y - 32)**2 + (x - 32)**2 <= 100).astype(np.float32) * 500.0

    # Cycle 0: reference
    stack[0, 0] = pattern
    stack[0, 1] = pattern

    # Cycle 1: shift by (2, 3)
    from scipy.ndimage import shift
    stack[1, 0] = shift(pattern, (2, 3), order=1)
    stack[1, 1] = shift(pattern, (2, 3), order=1)

    # Cycle 2: shift by (-1, 2)
    stack[2, 0] = shift(pattern, (-1, 2), order=1)
    stack[2, 1] = shift(pattern, (-1, 2), order=1)

    reg_stack, qc = register_cycles(stack, registration_channel=0, upsample_factor=10)
    
    assert reg_stack.shape == stack.shape
    assert len(qc) == T
    assert qc[0]["shift_magnitude"] < 0.2
    assert abs(qc[1]["shift_y"] - (-2.0)) < 0.5
    assert abs(qc[1]["shift_x"] - (-3.0)) < 0.5


def test_unmix_spectral_crosstalk():
    # 2 channels: Pure Green (ch 1) bleeds 20% into Red (ch 0)
    # M = [[1.0, 0.2], [0.0, 1.0]]
    M = [[1.0, 0.2], [0.0, 1.0]]
    T, C, H, W = 1, 2, 32, 32
    stack = np.zeros((T, C, H, W), dtype=np.float32)

    # True sources: S_red = 0, S_green = 100
    # Measured: Red = 20, Green = 100
    stack[0, 0] = 20.0
    stack[0, 1] = 100.0

    unmixed = unmix_spectral_crosstalk(stack, unmixing_matrix=M)
    assert np.allclose(unmixed[0, 0], 0.0, atol=1e-3)
    assert np.allclose(unmixed[0, 1], 100.0, atol=1e-3)


def test_background_noise_and_saturation():
    T, C, H, W = 1, 2, 64, 64
    stack = np.ones((T, C, H, W), dtype=np.float32) * 150.0  # background=150
    # Add noise
    stack += np.random.normal(0, 5.0, (T, C, H, W))
    # Add saturated pixel
    stack[0, 0, 10, 10] = 65535.0

    bg, noise, sat = estimate_background_and_noise(stack, saturation_threshold=65520.0)
    assert sat[0, 0, 10, 10] == True
    assert sat[0, 1, 10, 10] == False
    assert np.all(bg > 100.0)
    assert np.all(noise > 2.0)
