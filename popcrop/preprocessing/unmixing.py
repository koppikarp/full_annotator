"""Nonnegative spectral unmixing (NNLS) for multi-channel fluorescence imaging."""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import nnls
import logging

logger = logging.getLogger(__name__)


def unmix_spectral_crosstalk(
    sbs_stack: np.ndarray,
    unmixing_matrix: Optional[np.ndarray | list] = None
) -> np.ndarray:
    """
    Performs nonnegative spectral unmixing on linear multi-channel data.
    
    Given unmixing matrix M (C_measured x C_source):
      I_measured = M * S_source + noise
    Solves for S >= 0:
      min_{S >= 0} ||I_measured - M * S||_2^2
      
    Args:
      sbs_stack: shape (T, C, Y, X), float32
      unmixing_matrix: shape (C, C), float32
      
    Returns:
      unmixed_stack: shape (T, C, Y, X), float32
    """
    if unmixing_matrix is None:
        return sbs_stack

    M = np.asarray(unmixing_matrix, dtype=np.float32)
    T, C, H, W = sbs_stack.shape

    if M.shape != (C, C):
        raise ValueError(f"Unmixing matrix shape {M.shape} does not match channel count {C}x{C}")

    # If M is identity, return unchanged
    if np.allclose(M, np.eye(C), atol=1e-3):
        return sbs_stack

    logger.info(f"Applying nonnegative spectral unmixing with mixing matrix:\n{M}")

    # Fast analytical solution for 2x2 case
    if C == 2:
        return _unmix_2channel_vectorized(sbs_stack, M)

    # General C-channel NNLS
    unmixed = np.zeros_like(sbs_stack, dtype=np.float32)
    for t in range(T):
        # Reshape to (C, H*W)
        measured = sbs_stack[t].reshape(C, -1)
        sources = np.zeros_like(measured)
        for i in range(measured.shape[1]):
            s, _ = nnls(M, measured[:, i])
            sources[:, i] = s
        unmixed[t] = sources.reshape(C, H, W)

    return unmixed


def _unmix_2channel_vectorized(sbs_stack: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Fast vectorized 2-channel NNLS unmixing."""
    T, C, H, W = sbs_stack.shape
    out = np.zeros_like(sbs_stack, dtype=np.float32)

    m11, m12 = float(M[0, 0]), float(M[0, 1])
    m21, m22 = float(M[1, 0]), float(M[1, 1])
    det = m11 * m22 - m12 * m21

    if abs(det) < 1e-6:
        logger.warning("Unmixing matrix is singular or near-singular. Skipping unmixing.")
        return sbs_stack

    inv_det = 1.0 / det

    for t in range(T):
        y1 = sbs_stack[t, 0]
        y2 = sbs_stack[t, 1]

        # Unconstrained linear inversion
        s1 = (m22 * y1 - m12 * y2) * inv_det
        s2 = (-m21 * y1 + m11 * y2) * inv_det

        # KKT conditions for 2D NNLS:
        # Region 1: both s1 >= 0 and s2 >= 0 -> optimal
        # Region 2: s1 < 0 -> project to s1 = 0, s2 = max(0, (M[:,1]^T y) / ||M[:,1]||^2)
        # Region 3: s2 < 0 -> project to s2 = 0, s1 = max(0, (M[:,0]^T y) / ||M[:,0]||^2)
        norm1_sq = m11**2 + m21**2
        norm2_sq = m12**2 + m22**2

        # Safe projection when s1 < 0
        s2_only = np.maximum(0.0, (m12 * y1 + m22 * y2) / norm2_sq)
        # Safe projection when s2 < 0
        s1_only = np.maximum(0.0, (m11 * y1 + m21 * y2) / norm1_sq)

        mask_both = (s1 >= 0) & (s2 >= 0)
        mask_s1_neg = (s1 < 0) & (s2 >= 0)
        mask_s2_neg = (s1 >= 0) & (s2 < 0)
        mask_both_neg = (s1 < 0) & (s2 < 0)

        out_s1 = np.zeros_like(s1)
        out_s2 = np.zeros_like(s2)

        out_s1[mask_both] = s1[mask_both]
        out_s2[mask_both] = s2[mask_both]

        out_s1[mask_s1_neg] = 0.0
        out_s2[mask_s1_neg] = s2_only[mask_s1_neg]

        out_s1[mask_s2_neg] = s1_only[mask_s2_neg]
        out_s2[mask_s2_neg] = 0.0

        out_s1[mask_both_neg] = 0.0
        out_s2[mask_both_neg] = 0.0

        out[t, 0] = out_s1
        out[t, 1] = out_s2

    return out
