"""Chromatic shift correction between Red and Green channels and optional PSF matching."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian
import logging

logger = logging.getLogger(__name__)


def correct_chromatic_shift(
    sbs_stack: np.ndarray,
    shift_red_green: List[float] = (0.0, 0.0),
    red_channel_idx: int = 0,
    green_channel_idx: int = 1
) -> np.ndarray:
    """
    Applies subpixel chromatic shift correction between Red and Green emission channels.
    shift_red_green: (shift_y, shift_x) of Red relative to Green.
    """
    shift_y, shift_x = float(shift_red_green[0]), float(shift_red_green[1])
    if abs(shift_y) < 1e-4 and abs(shift_x) < 1e-4:
        return sbs_stack

    logger.info(f"Applying chromatic shift to Red channel: dy={shift_y:.2f}, dx={shift_x:.2f}")
    out_stack = sbs_stack.copy()
    T, C, H, W = out_stack.shape

    for t in range(T):
        out_stack[t, red_channel_idx] = ndi.shift(
            out_stack[t, red_channel_idx],
            shift=(-shift_y, -shift_x),
            order=1,
            mode="reflect",
            prefilter=False
        )

    return out_stack


def match_psf(
    sbs_stack: np.ndarray,
    target_sigma: float = 1.0,
    channel_sigmas: Tuple[float, float] = (1.2, 1.0)
) -> np.ndarray:
    """
    PSF matching across channels: applies Gaussian smoothing so that all channels
    exhibit equivalent optical resolution / spot spread.
    """
    out_stack = sbs_stack.copy()
    T, C, H, W = out_stack.shape

    for c in range(min(C, len(channel_sigmas))):
        current_sigma = channel_sigmas[c]
        if current_sigma < target_sigma:
            diff_sigma = np.sqrt(target_sigma**2 - current_sigma**2)
            for t in range(T):
                out_stack[t, c] = gaussian(out_stack[t, c], sigma=diff_sigma, preserve_range=True)

    return out_stack
