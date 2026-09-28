"""Nucleus mask loader and adapter for precomputed, Cellpose, or classical segmentations."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tifffile
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
import logging

logger = logging.getLogger(__name__)


def load_or_generate_nuclear_mask(
    mask_path: Optional[str | Path] = None,
    dapi_image: Optional[np.ndarray] = None,
    min_nucleus_size: int = 80,
    max_nucleus_size: int = 15000,
) -> np.ndarray:
    """
    Loads an existing nuclear mask or segments DAPI using a classical Otsu-watershed pipeline.
    
    Returns:
      labeled_mask: 2D integer ndarray (Y, X) where each nucleus has unique integer ID > 0.
    """
    if mask_path is not None:
        path = Path(mask_path)
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {path}")

        logger.info(f"Loading nuclear mask from {path}")
        if path.suffix == ".npy":
            raw_data = np.load(path, allow_pickle=True)
            if isinstance(raw_data, np.ndarray) and raw_data.dtype == object and raw_data.shape == ():
                # Cellpose dict format
                cpose_dict = raw_data.item()
                if "masks" in cpose_dict:
                    return cpose_dict["masks"].astype(np.int32)
            if isinstance(raw_data, np.ndarray):
                return raw_data.astype(np.int32)
        elif path.suffix in [".tif", ".tiff"]:
            mask = tifffile.imread(path)
            return np.squeeze(mask).astype(np.int32)
        else:
            raise ValueError(f"Unsupported mask file format: {path.suffix}")

    # If no mask path provided, use classical segmentation on DAPI
    if dapi_image is None:
        raise ValueError("Neither mask_path nor dapi_image was provided for nuclear segmentation.")

    logger.info("No precomputed nuclear mask provided. Running classical DAPI nuclear segmentation.")
    return segment_dapi_classical(
        dapi_image,
        min_size=min_nucleus_size,
        max_size=max_nucleus_size
    )


def segment_dapi_classical(
    dapi_image: np.ndarray,
    min_size: int = 80,
    max_size: int = 15000,
    smoothing_sigma: float = 2.0
) -> np.ndarray:
    """
    Classical nuclear segmentation using Gaussian smoothing, Otsu thresholding,
    distance transform, and watershed separation.
    """
    dapi = np.squeeze(dapi_image).astype(np.float32)
    smoothed = gaussian(dapi, sigma=smoothing_sigma, preserve_range=True)
    
    # Check dynamic range
    if smoothed.max() <= smoothed.min():
        logger.warning("DAPI image has zero contrast. Returning empty mask.")
        return np.zeros_like(dapi, dtype=np.int32)

    try:
        thresh = threshold_otsu(smoothed)
    except ValueError:
        thresh = np.mean(smoothed)

    binary = smoothed > thresh
    binary = remove_small_objects(binary, min_size=min_size)

    # Separate touching nuclei using distance transform & watershed
    distance = ndi.distance_transform_edt(binary)
    # Local maxima as seeds
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    coords = peak_local_max(distance, min_distance=15, labels=binary)
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)

    if markers.max() > 0:
        labeled = watershed(-distance, markers, mask=binary)
    else:
        labeled = label(binary)

    # Filter by size
    unique_labels, counts = np.unique(labeled, return_counts=True)
    valid_labels = unique_labels[(counts >= min_size) & (counts <= max_size) & (unique_labels > 0)]
    
    # Relabel once, instead of scanning 32 million pixels once per nucleus.
    lookup = np.zeros(int(labeled.max()) + 1, dtype=np.int32)
    lookup[valid_labels] = np.arange(1, len(valid_labels) + 1)
    out_mask = lookup[labeled]
    new_id = len(valid_labels) + 1

    logger.info(f"Segmented {new_id - 1} nuclei from DAPI image.")
    return out_mask
