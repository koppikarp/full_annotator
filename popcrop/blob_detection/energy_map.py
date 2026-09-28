"""Positive signal z-score map and cycle energy map calculation."""

from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_zscore_and_energy(
    sbs_stack: np.ndarray,
    background_maps: np.ndarray,
    noise_maps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates robust background-normalized positive signal maps:
      z[t, c](x) = max(0, (I[t, c](x) - background[t, c](x)) / noise[t, c](x))
      
    Combines channels per cycle into an omni-channel energy map:
      e[t](x) = sqrt(sum_c z[t, c](x)^2)
      
    Args:
      sbs_stack: shape (T, C, Y, X), float32
      background_maps: shape (T, C, Y, X), float32
      noise_maps: shape (T, C, Y, X), float32
      
    Returns:
      z_scores: shape (T, C, Y, X), float32
      cycle_energies: shape (T, Y, X), float32
    """
    safe_noise = np.maximum(noise_maps, 1e-3)
    excess_intensity = np.maximum(0.0, sbs_stack - background_maps)
    
    # Positive SNR map z[t, c](x)
    z_scores = (excess_intensity / safe_noise).astype(np.float32)

    # Combined cycle energy: L2 norm across channels
    # e[t](x) = sqrt(sum_c z[t,c]^2)
    cycle_energies = np.sqrt(np.sum(z_scores**2, axis=1)).astype(np.float32)

    return z_scores, cycle_energies
