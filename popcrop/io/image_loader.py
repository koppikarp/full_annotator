"""Unified image loader supporting both Nikon ND2 and linear TIFF/OME-TIFF files."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
import logging

from popcrop.io.tiff_loader import load_raw_tiff
from popcrop.io.nd2_loader import load_raw_nd2

logger = logging.getLogger(__name__)


def load_raw_image(
    image_path_or_paths: Union[str, Path, List[Union[str, Path]]],
    expected_channels: int = 2,
    has_dapi_cycle: bool = True,
    dapi_channel_idx: int = 0,
    position_idx: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Universal image loader for optical pooled screening.
    Automatically detects whether inputs are Nikon ND2 or TIFF/OME-TIFF format,
    standardizing the output to:
      sbs_stack: shape (T, C, Y, X), float32
      dapi_image: shape (Y, X), float32 (or None)
      metadata: dictionary with original dimensions and microscope parameters
    """
    # Detect format
    is_nd2 = False
    if isinstance(image_path_or_paths, (list, tuple)):
        first = str(image_path_or_paths[0]).lower()
        if first.endswith(".nd2"):
            is_nd2 = True
    else:
        path_str = str(image_path_or_paths).lower()
        p = Path(image_path_or_paths)
        if path_str.endswith(".nd2") or (p.is_dir() and any(p.glob("*.nd2"))):
            is_nd2 = True

    if is_nd2:
        logger.info("Detected Nikon ND2 format. Dispatching to load_raw_nd2.")
        return load_raw_nd2(
            file_or_files=image_path_or_paths,
            position_idx=position_idx,
            expected_channels=expected_channels,
            has_dapi_cycle=has_dapi_cycle,
            dapi_channel_idx=dapi_channel_idx
        )
    else:
        logger.info("Detected TIFF format. Dispatching to load_raw_tiff.")
        return load_raw_tiff(
            tiff_path=image_path_or_paths if not isinstance(image_path_or_paths, (list, tuple)) else image_path_or_paths[0],
            expected_channels=expected_channels,
            has_dapi_cycle=has_dapi_cycle,
            dapi_channel_idx=dapi_channel_idx
        )
