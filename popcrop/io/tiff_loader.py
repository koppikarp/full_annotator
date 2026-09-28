"""Multicycle, multichannel TIFF and OME-TIFF loader with axis normalization."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import tifffile
import logging

logger = logging.getLogger(__name__)


def load_raw_tiff(
    tiff_path: str | Path,
    expected_channels: int = 2,
    has_dapi_cycle: bool = True,
    dapi_channel_idx: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads a multicycle, multichannel linear TIFF file and normalizes axes to:
      sbs_stack: (n_cycles, n_channels, Y, X), float32
      dapi_image: (Y, X), float32 (or None if no DAPI cycle)
      metadata: dict of original metadata and dimensions

    Supported input shapes:
      - (T, Y, X, C)
      - (T, C, Y, X)
      - (T, C, Z, Y, X) -> max-projected across Z
      - (N_pages, Y, X) -> reshaped based on expected_channels
    """
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF file not found: {path}")

    logger.info(f"Loading TIFF from {path}")
    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        n_pages = len(pages)
        first_page = pages[0]
        raw_shape = first_page.shape
        raw_dtype = first_page.dtype

        # Check linear 16-bit / raw properties
        if raw_dtype == np.uint8:
            logger.warning("Raw image is 8-bit. Standard raw linear data should ideally be 16-bit.")

        # Preserve raw integer pixels and use read-only file-backed pages when
        # possible. A float32 copy of this experiment alone is almost 5 GB.
        try:
            arr = tifffile.memmap(path, mode="r")
        except (ValueError, TypeError):
            arr = tif.asarray(out="memmap")

    logger.info(f"Loaded raw TIFF with shape {arr.shape} and dtype {arr.dtype}")
    metadata = {
        "source_path": str(path),
        "raw_shape": arr.shape,
        "raw_dtype": str(arr.dtype),
        "n_pages": n_pages,
    }

    # Squeeze singleton dimensions
    arr = np.squeeze(arr)

    # Normalize to (T, C, Y, X)
    norm_arr = _standardize_axes(arr, expected_channels)

    # Separate DAPI if present
    if has_dapi_cycle:
        if norm_arr.shape[0] < 2:
            raise ValueError(
                f"Image has only {norm_arr.shape[0]} cycle(s), but has_dapi_cycle=True requires at least 2 cycles."
            )
        # Handle RGB composite stacks (where channels are Red=0, Green=1, Blue/DAPI=2)
        if norm_arr.shape[1] >= 3:
            # Cycle 1 Page Blue is the clean reference DAPI image
            dapi_image = norm_arr[1, 2]
            metadata["reference_dapi"] = norm_arr[0, 2]
            metadata["dapi_stack"] = norm_arr[1:, 2]
            sbs_stack = norm_arr[1:, :expected_channels]
        else:
            # Cycle 0 is DAPI/phenotype
            dapi_raw = norm_arr[0]
            # Extract the requested DAPI channel index
            if dapi_raw.shape[0] > dapi_channel_idx:
                dapi_image = dapi_raw[dapi_channel_idx]
            else:
                dapi_image = dapi_raw[0]
            sbs_stack = norm_arr[1:, :expected_channels]
    else:
        dapi_image = None
        sbs_stack = norm_arr[:, :expected_channels]

    logger.info(
        f"Standardized SBS stack: {sbs_stack.shape} (T, C, Y, X)"
        + (f", DAPI image: {dapi_image.shape}" if dapi_image is not None else "")
    )
    return sbs_stack, dapi_image, metadata


def _standardize_axes(arr: np.ndarray, expected_channels: int = 2) -> np.ndarray:
    """Standardizes arbitrary TIFF dimension order to (T, C, Y, X)."""
    # Max-project across Z if 5D: (T, C, Z, Y, X) or (T, Z, Y, X, C)
    if arr.ndim == 5:
        # Check which dimension is Z (typically index 2 or 1)
        if arr.shape[1] == expected_channels:
            # (T, C, Z, Y, X)
            arr = arr.max(axis=2)
        elif arr.shape[-1] == expected_channels:
            # (T, Z, Y, X, C)
            arr = arr.max(axis=1)
            arr = np.moveaxis(arr, -1, 1)  # (T, C, Y, X)
        else:
            # Generic max projection across the smallest remaining spatial dimension
            z_axis = int(np.argmin(arr.shape[1:3])) + 1
            arr = arr.max(axis=z_axis)

    if arr.ndim == 4:
        # Check if channel is last: (T, Y, X, C)
        if arr.shape[-1] == expected_channels or arr.shape[-1] <= 4:
            # Move channel to index 1: (T, C, Y, X)
            arr = np.moveaxis(arr, -1, 1)
        # Now arr has shape (T, C, Y, X)
        return arr

    if arr.ndim == 3:
        # Could be (C, Y, X) for single cycle, or (N_pages, Y, X)
        if arr.shape[0] == expected_channels:
            # Single cycle (1, C, Y, X)
            return arr[None, ...]
        elif arr.shape[0] % expected_channels == 0:
            # Flattened cycles and channels: reshape to (T, C, Y, X)
            n_cycles = arr.shape[0] // expected_channels
            return arr.reshape(n_cycles, expected_channels, arr.shape[1], arr.shape[2])
        else:
            # Assume 1 channel per page -> (T, 1, Y, X)
            return arr[:, None, :, :]

    if arr.ndim == 2:
        # Single plane: (1, 1, Y, X)
        return arr[None, None, ...]

    raise ValueError(f"Unsupported array dimensionality: {arr.ndim} (shape: {arr.shape})")
