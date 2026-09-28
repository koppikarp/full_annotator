"""Extended region blob segmentation using hysteresis thresholding and connected components."""

from __future__ import annotations
from typing import Tuple, Dict, List, Any
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops
import logging

logger = logging.getLogger(__name__)


def segment_extended_blobs(
    combined_metric: np.ndarray,
    nuclear_mask: np.ndarray,
    hysteresis_high: float = 4.0,
    hysteresis_low: float = 2.0,
    min_blob_area: int = 8,
    max_blob_fraction: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects extended fluorescent blobs inside segmented nuclei using hysteresis thresholding.
    
    Operates without assuming point-like local maxima. Accurately segments broad,
    irregular, off-center, fragmented, or multi-component plateaus.
    
    Args:
      combined_metric: shape (Y, X), float32 (energy * sqrt(1 + persistence))
      nuclear_mask: shape (Y, X), int32 (labeled nuclei)
      hysteresis_high: seed threshold for core detection
      hysteresis_low: boundary expansion threshold
      min_blob_area: minimum component size in pixels
      max_blob_fraction: maximum fraction of nuclear area a blob can occupy
      
    Returns:
      blob_mask: shape (Y, X), int32 (unique label per detected component > 0)
      blob_prob_map: shape (Y, X), float32 (soft probability map [0.0, 1.0])
    """
    H, W = combined_metric.shape
    in_nuclei = nuclear_mask > 0

    # 1. Soft Blob Probability Map: Calibrated sigmoid on combined metric
    # Midpoint at (high + low) / 2
    midpoint = (hysteresis_high + hysteresis_low) / 2.0
    scale = max(0.5, (hysteresis_high - hysteresis_low) / 2.0)
    blob_prob_map = 1.0 / (1.0 + np.exp(-((combined_metric - midpoint) / scale)))
    # Set probability outside nuclei to zero
    blob_prob_map[~in_nuclei] = 0.0

    # 2. Hysteresis Thresholding: detects extended regions from high seeds down to low threshold
    # Restrict to pixels inside nuclei
    metric_in_nuclei = np.where(in_nuclei, combined_metric, 0.0)
    raw_binary = apply_hysteresis_threshold(
        metric_in_nuclei,
        low=float(hysteresis_low),
        high=float(hysteresis_high)
    )

    # 3. Component Extraction per Nucleus in a single vectorized pass
    blob_mask = np.zeros((H, W), dtype=np.int32)
    next_component_id = 1

    # Precompute nuclear areas once via regionprops
    nuc_areas = {p.label: p.area for p in regionprops(nuclear_mask)}

    # Single-pass connected component labeling across the entire image
    labeled_all, num_comp = ndi.label(raw_binary, structure=np.ones((3, 3), dtype=bool))
    
    if num_comp > 0:
        for p in regionprops(labeled_all):
            coords = p.coords
            coords_nuc = nuclear_mask[coords[:, 0], coords[:, 1]]
            for n_id in np.unique(coords_nuc):
                if n_id == 0:
                    continue
                sub_coords = coords[coords_nuc == n_id]
                c_area = len(sub_coords)
                if c_area < min_blob_area:
                    continue
                n_area = nuc_areas.get(n_id, 1000)
                if (c_area / n_area) > max_blob_fraction:
                    continue
                blob_mask[sub_coords[:, 0], sub_coords[:, 1]] = next_component_id
                next_component_id += 1

    logger.info(f"Segmented {next_component_id - 1} extended blob components across {len(nuc_areas)} nuclei.")
    return blob_mask, blob_prob_map
