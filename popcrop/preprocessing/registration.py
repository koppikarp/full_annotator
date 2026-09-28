"""Cycle-to-cycle subpixel phase cross-correlation registration."""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import logging

logger = logging.getLogger(__name__)


def register_cycles(
    sbs_stack: np.ndarray,
    dapi_image: Optional[np.ndarray] = None,
    registration_channel: int = 0,
    reference_cycle_idx: int = 0,
    upsample_factor: int = 10,
    max_allowed_shift: float = 25.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Registers SBS cycles (T, C, Y, X) using subpixel phase cross-correlation.
    
    If dapi_image is provided and reference_cycle_idx == 0, registers all cycles
    against DAPI or against cycle 0 of the SBS stack.
    
    Returns:
      registered_stack: shape (T, C, Y, X), float32
      registration_qc: list of dicts with shift_y, shift_x, error, shift_magnitude, qc_flag per cycle.
    """
    T, C, H, W = sbs_stack.shape
    registered_stack = np.zeros_like(sbs_stack, dtype=np.float32)
    registration_qc = []

    # Choose reference image
    if dapi_image is not None:
        ref_image = dapi_image.astype(np.float32)
    else:
        ref_image = sbs_stack[reference_cycle_idx, registration_channel].astype(np.float32)

    logger.info(f"Starting cycle-to-cycle registration for {T} cycles with upsample_factor={upsample_factor}.")

    for t in range(T):
        moving_image = sbs_stack[t, registration_channel].astype(np.float32)

        try:
            # Calculate subpixel translation shift
            shift, error, diffphase = phase_cross_correlation(
                ref_image,
                moving_image,
                upsample_factor=upsample_factor
            )
            shift_y, shift_x = float(shift[0]), float(shift[1])
            shift_mag = float(np.sqrt(shift_y**2 + shift_x**2))
            is_valid = shift_mag <= max_allowed_shift
        except Exception as e:
            logger.warning(f"Registration failed on cycle {t}: {e}. Keeping zero shift.")
            shift_y, shift_x = 0.0, 0.0
            shift_mag = 0.0
            error = 1.0
            is_valid = False

        qc = {
            "cycle": t,
            "shift_y": shift_y,
            "shift_x": shift_x,
            "shift_magnitude": shift_mag,
            "error": float(error),
            "qc_passed": bool(is_valid)
        }
        registration_qc.append(qc)

        # Apply shift to all channels of cycle t
        for c in range(C):
            if shift_mag > 1e-4:
                registered_stack[t, c] = ndi.shift(
                    sbs_stack[t, c],
                    shift=(shift_y, shift_x),
                    order=1,
                    mode="reflect",
                    prefilter=False
                )
            else:
                registered_stack[t, c] = sbs_stack[t, c]

    logger.info("Cycle-to-cycle registration completed.")
    return registered_stack, registration_qc
