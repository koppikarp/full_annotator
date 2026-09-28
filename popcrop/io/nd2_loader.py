"""Nikon ND2 file loader with multi-position, multi-cycle, and Z-stack projection support."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
import logging
import glob
import re

logger = logging.getLogger(__name__)

try:
    import nd2
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False


def load_raw_nd2(
    file_or_files: Union[str, Path, List[Union[str, Path]]],
    position_idx: int = 0,
    expected_channels: int = 2,
    has_dapi_cycle: bool = True,
    dapi_channel_idx: int = 0,
    channel_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads Nikon ND2 file(s) and standardizes to:
      sbs_stack: shape (T, C, Y, X), float32
      dapi_image: shape (Y, X), float32 (or None)
      metadata: dictionary of microscope and acquisition metadata

    Supports:
      1. A single ND2 file containing multiple cycles (T dimension) and/or positions (P dimension).
      2. A list of ND2 files (e.g. one file per SBS cycle).
      3. A directory containing ND2 files sorted by cycle.
      4. Multi-plane Z-stacks (automatically max-projected across Z).
    """
    if not ND2_AVAILABLE:
        raise ImportError(
            "The 'nd2' library is required to read Nikon ND2 files. Install via 'pip install nd2'."
        )

    # 1. Resolve file list
    file_list = _resolve_nd2_files(file_or_files)
    logger.info(f"Loading ND2 data from {len(file_list)} file(s).")

    metadata: Dict[str, Any] = {
        "source_files": [str(f) for f in file_list],
        "format": "nd2"
    }

    # Case A: Multiple ND2 files (one per cycle)
    if len(file_list) > 1:
        cycle_arrays = []
        for f_path in file_list:
            c_arr, meta_f = _read_single_nd2(f_path, position_idx=position_idx)
            cycle_arrays.append(c_arr)
            metadata.setdefault("nd2_meta", meta_f)

        # Stack cycles: shape (T, C, Y, X)
        full_stack = np.stack(cycle_arrays, axis=0)

    # Case B: Single ND2 file
    else:
        full_stack, meta_f = _read_single_nd2(file_list[0], position_idx=position_idx)
        metadata["nd2_meta"] = meta_f

    logger.info(f"Extracted raw ND2 stack with shape: {full_stack.shape}")

    # Channel selection / reordering if specified
    if channel_indices is not None:
        full_stack = full_stack[:, channel_indices, :, :]

    # Handle DAPI cycle separation
    if has_dapi_cycle:
        if full_stack.shape[0] < 2:
            # If only 1 cycle loaded, but has DAPI channel inside it
            if full_stack.shape[1] > expected_channels and dapi_channel_idx < full_stack.shape[1]:
                dapi_image = full_stack[0, dapi_channel_idx].astype(np.float32)
                # Keep the other channels as SBS
                sbs_channels = [c for c in range(full_stack.shape[1]) if c != dapi_channel_idx]
                sbs_stack = full_stack[:, sbs_channels, :, :].astype(np.float32)
            else:
                raise ValueError(
                    f"ND2 has only {full_stack.shape[0]} cycle(s), but has_dapi_cycle=True requires >= 2 cycles "
                    "or a separate DAPI channel in the stack."
                )
        else:
            # Cycle 0 is DAPI/phenotype
            dapi_plane = full_stack[0]
            if dapi_plane.shape[0] > dapi_channel_idx:
                dapi_image = dapi_plane[dapi_channel_idx].astype(np.float32)
            else:
                dapi_image = dapi_plane[0].astype(np.float32)

            sbs_stack = full_stack[1:].astype(np.float32)
    else:
        dapi_image = None
        sbs_stack = full_stack.astype(np.float32)

    logger.info(
        f"Standardized ND2 SBS stack: {sbs_stack.shape} (T, C, Y, X)"
        + (f", DAPI image: {dapi_image.shape}" if dapi_image is not None else "")
    )
    return sbs_stack, dapi_image, metadata


def _read_single_nd2(
    f_path: Path,
    position_idx: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads a single ND2 file, collapses Z via max projection, and returns (T, C, Y, X)."""
    with nd2.ND2File(str(f_path)) as nd2_file:
        sizes = dict(nd2_file.sizes)
        logger.info(f"Reading ND2 {f_path.name} with axes: {sizes}")

        arr = nd2_file.asarray()
        meta = {
            "sizes": sizes,
            "dtype": str(nd2_file.dtype),
            "is_rgb": bool(nd2_file.is_rgb),
        }

        # Extract calibration info if available
        try:
            vox = nd2_file.voxel_size()
            meta["pixel_size_um"] = {"x": vox.x, "y": vox.y, "z": vox.z}
        except Exception:
            pass

    # Dimensional axes mapping
    # Standard ND2 axes order in sizes dict: e.g. T, P, Z, C, Y, X
    # Select position if P exists
    if "P" in sizes and sizes["P"] > 1:
        p_axis = list(sizes.keys()).index("P")
        p_idx = min(position_idx, sizes["P"] - 1)
        arr = np.take(arr, indices=p_idx, axis=p_axis)
        sizes.pop("P")

    # Max-project across Z if Z exists
    if "Z" in sizes and sizes["Z"] > 1:
        z_axis = list(sizes.keys()).index("Z")
        arr = np.max(arr, axis=z_axis)
        sizes.pop("Z")
    elif "Z" in sizes:
        z_axis = list(sizes.keys()).index("Z")
        arr = np.squeeze(arr, axis=z_axis)
        sizes.pop("Z")

    # Squeeze any remaining singleton dimensions
    arr = np.squeeze(arr)

    # Standardize to (T, C, Y, X)
    if arr.ndim == 4:
        # Check if shape is (T, C, Y, X) or (T, Y, X, C)
        if arr.shape[-1] <= 6:
            # Channel is last: (T, Y, X, C) -> move to index 1
            arr = np.moveaxis(arr, -1, 1)
    elif arr.ndim == 3:
        # Could be (C, Y, X) or (T, Y, X)
        if "C" in sizes and sizes["C"] > 1:
            # Shape is (C, Y, X) -> expand T dimension
            arr = arr[None, ...]
        elif "T" in sizes and sizes["T"] > 1:
            # Shape is (T, Y, X) -> expand C dimension
            arr = arr[:, None, :, :]
        else:
            # Default: treat axis 0 as channel (1, C, Y, X)
            arr = arr[None, ...]
    elif arr.ndim == 2:
        # (Y, X) -> (1, 1, Y, X)
        arr = arr[None, None, ...]

    return arr.astype(np.float32), meta


def _resolve_nd2_files(file_or_files: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
    """Resolves input path, directory, glob pattern, or list to sorted Path objects."""
    if isinstance(file_or_files, (list, tuple)):
        return sorted([Path(f) for f in file_or_files])

    path = Path(file_or_files)
    if path.is_dir():
        # Directory of ND2 files
        matches = sorted(path.glob("*.nd2"))
        if not matches:
            raise FileNotFoundError(f"No .nd2 files found in directory: {path}")
        # Sort naturally by cycle number if present
        return _sort_by_cycle_number(matches)

    if "*" in str(file_or_files):
        # Glob pattern
        matches = sorted([Path(p) for p in glob.glob(str(file_or_files))])
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {file_or_files}")
        return _sort_by_cycle_number(matches)

    if path.exists():
        return [path]

    raise FileNotFoundError(f"ND2 file or path not found: {file_or_files}")


def _sort_by_cycle_number(paths: List[Path]) -> List[Path]:
    """Sorts paths by cycle number in filename (e.g. C-1, cycle_02, etc.)."""
    def extract_cycle(p: Path) -> int:
        match = re.search(r"[cC](?:ycle)?[-_]?(\d+)", p.name)
        return int(match.group(1)) if match else 0

    has_cycles = any(re.search(r"[cC](?:ycle)?[-_]?(\d+)", p.name) for p in paths)
    if has_cycles:
        return sorted(paths, key=extract_cycle)
    return sorted(paths)
