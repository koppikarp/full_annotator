"""Illumination correction, camera-offset subtraction, and field flattening."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian
import logging

logger = logging.getLogger(__name__)


def subtract_camera_offset(
    image_stack: np.ndarray,
    camera_offset: float = 100.0
) -> np.ndarray:
    """
    Subtracts baseline camera dark offset from linear image data.
    Input image_stack: shape (..., Y, X), float32.
    """
    if float(camera_offset) == 0.0:
        return image_stack
    return np.maximum(0.0, image_stack - float(camera_offset))


def correct_illumination_stack(
    sbs_stack: np.ndarray,
    well_id: str = "W1",
    do_field_flattening: bool = True,
    mode: str = "retrospective",
    per_cycle: bool = True,
    per_well: bool = True,
    reference_flatfield: Optional[np.ndarray] = None,
    filter_size: int = 128
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Corrects illumination across (cycles, channels, Y, X).
    Ensures field flattening is calculated independently per cycle, per channel, and per well.
    """
    if not do_field_flattening:
        return sbs_stack, None

    T, C, H, W = sbs_stack.shape
    flattened_stack = np.zeros_like(sbs_stack, dtype=np.float32)
    illumination_profiles = np.zeros_like(sbs_stack, dtype=np.float32)

    if mode == "reference" and reference_flatfield is not None:
        ref = np.squeeze(reference_flatfield).astype(np.float32)
        ref_mean = np.mean(ref)
        safe_ref = np.maximum(ref, 1e-3)
        for t in range(T):
            for c in range(C):
                flattened_stack[t, c] = (sbs_stack[t, c] / safe_ref) * ref_mean
                illumination_profiles[t, c] = safe_ref / ref_mean
        return flattened_stack, illumination_profiles

    # Retrospective field flattening per cycle and channel
    for c in range(C):
        if per_cycle:
            for t in range(T):
                plane = sbs_stack[t, c]
                # If channel is at noise floor / dark, keep as is
                if np.mean(plane) < 10.0:
                    flattened_stack[t, c] = plane
                    illumination_profiles[t, c] = np.ones_like(plane, dtype=np.float32)
                    continue

                profile = estimate_illumination_profile(plane, filter_size=filter_size)
                profile_mean = float(np.mean(profile))
                if profile_mean < 5.0:
                    flattened_stack[t, c] = plane
                    illumination_profiles[t, c] = np.ones_like(plane, dtype=np.float32)
                else:
                    safe_profile = np.maximum(profile, 1.0)
                    flattened_stack[t, c] = (plane / safe_profile) * profile_mean
                    illumination_profiles[t, c] = safe_profile
        else:
            cycle_mean = np.mean(sbs_stack[:, c], axis=0)
            if np.mean(cycle_mean) < 10.0:
                for t in range(T):
                    flattened_stack[t, c] = sbs_stack[t, c]
                    illumination_profiles[t, c] = np.ones_like(cycle_mean, dtype=np.float32)
                continue

            profile = estimate_illumination_profile(cycle_mean, filter_size=filter_size)
            profile_mean = float(np.mean(profile))
            if profile_mean < 5.0:
                for t in range(T):
                    flattened_stack[t, c] = sbs_stack[t, c]
                    illumination_profiles[t, c] = np.ones_like(cycle_mean, dtype=np.float32)
            else:
                safe_profile = np.maximum(profile, 1.0)
                for t in range(T):
                    flattened_stack[t, c] = (sbs_stack[t, c] / safe_profile) * profile_mean
                    illumination_profiles[t, c] = safe_profile

    return flattened_stack, illumination_profiles


def estimate_illumination_profile(
    image_2d: np.ndarray,
    filter_size: int = 128
) -> np.ndarray:
    """
    Estimates the low-frequency illumination profile / vignetting of a 2D image.
    Uses median filter and Gaussian smoothing.
    """
    img = image_2d.astype(np.float32)
    if np.mean(img) < 5.0:
        return np.ones_like(img, dtype=np.float32) * max(1.0, float(np.mean(img)))

    f_size = max(9, filter_size // 4)
    med = ndi.median_filter(img, size=f_size)
    sigma = filter_size / 3.0
    profile = gaussian(med, sigma=sigma, preserve_range=True)
    return np.maximum(profile, 1.0).astype(np.float32)
