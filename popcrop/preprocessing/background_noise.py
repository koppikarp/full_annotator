"""Robust background and noise estimation and saturation detection."""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from scipy import ndimage as ndi
import logging

logger = logging.getLogger(__name__)


def estimate_background_and_noise(
    sbs_stack: np.ndarray,
    bg_filter_size: int = 35,
    bg_quantile: float = 0.15,
    saturation_threshold: float = 65520.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes robust 2D background and noise maps for each cycle and channel:
      sbs_stack: shape (T, C, Y, X), float32
      
    Returns:
      background_maps: (T, C, Y, X), float32
      noise_maps: (T, C, Y, X), float32
      saturation_masks: (T, C, Y, X), bool
    """
    T, C, H, W = sbs_stack.shape
    background_maps = np.zeros_like(sbs_stack, dtype=np.float32)
    noise_maps = np.zeros_like(sbs_stack, dtype=np.float32)
    saturation_masks = np.zeros_like(sbs_stack, dtype=bool)

    filter_size = max(5, int(bg_filter_size))

    for t in range(T):
        for c in range(C):
            plane = sbs_stack[t, c].astype(np.float32)

            # 1. Saturation Mask
            sat_mask = plane >= saturation_threshold
            saturation_masks[t, c] = sat_mask

            # 2. Robust Background Estimation
            # Large median filter isolates baseline background floor without noise trough bias
            if min(H, W) > 300 and filter_size >= 15:
                factor = 8 if min(H, W) >= 1000 else 4
                sub = plane[::factor, ::factor]
                sub_fsize = max(3, filter_size // factor)
                bg_sub = ndi.median_filter(sub, size=sub_fsize)
                bg = ndi.zoom(bg_sub, (H / bg_sub.shape[0], W / bg_sub.shape[1]), order=1)[:H, :W]
            else:
                bg = ndi.median_filter(plane, size=filter_size)
            bg = np.maximum(bg, 1.0)
            background_maps[t, c] = bg

            # 3. Robust High-Frequency Noise Estimation (MAD)
            med_filtered = ndi.median_filter(plane, size=3)
            residual = np.abs(plane - med_filtered)
            
            valid_pixels = residual[~sat_mask] if np.any(~sat_mask) else residual
            if len(valid_pixels) > 500000:
                sample_step = max(1, len(valid_pixels) // 200000)
                mad = float(np.median(valid_pixels[::sample_step]))
            else:
                mad = float(np.median(valid_pixels))
            sigma = max(1.0, 1.4826 * mad)

            local_noise = np.sqrt(sigma**2 + np.maximum(0.0, plane - bg))
            noise_maps[t, c] = np.maximum(local_noise, 1.0).astype(np.float32)

    return background_maps, noise_maps, saturation_masks
