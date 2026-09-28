"""Distance transform and core-versus-rim spatial decomposition for fluorescent blobs."""

from __future__ import annotations
from typing import Dict, Tuple, Any
import numpy as np
from scipy import ndimage as ndi
import logging

logger = logging.getLogger(__name__)


def compute_core_rim_geometry(
    component_mask: np.ndarray,
    distance_percentile: float = 0.35,
    min_core_pixels: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Decomposes a 2D blob component into soft core weights, a discrete core mask,
    and a discrete rim mask using the Euclidean distance transform.
    
    Args:
      component_mask: 2D boolean array representing the single blob component.
      distance_percentile: fraction of max boundary distance defining core boundary.
      min_core_pixels: minimum pixels to assign to core if component is small.
      
    Returns:
      core_weights: continuous float32 array in [0.0, 1.0], increasing toward interior.
      core_mask: 2D boolean array for the interior core.
      rim_mask: 2D boolean array for the exterior rim.
      max_distance: maximum distance from boundary (in pixels).
    """
    if not np.any(component_mask):
        empty = np.zeros_like(component_mask, dtype=bool)
        return empty.astype(np.float32), empty, empty, 0.0

    # Euclidean distance transform from boundary inward
    dist = ndi.distance_transform_edt(component_mask).astype(np.float32)
    max_dist = float(np.max(dist))

    if max_dist <= 1e-4:
        # 1-pixel or flat component
        return component_mask.astype(np.float32), component_mask.copy(), np.zeros_like(component_mask, dtype=bool), max_dist

    # Normalized soft core weights in [0, 1]
    core_weights = np.where(component_mask, dist / max_dist, 0.0).astype(np.float32)

    # Core threshold: pixels whose distance to edge is >= distance_percentile of max_dist
    thresh = max_dist * float(distance_percentile)
    core_mask = (dist >= thresh) & component_mask

    # If core has fewer than min_core_pixels, select top min_core_pixels deepest pixels
    n_core = np.sum(core_mask)
    if n_core < min_core_pixels:
        flat_dist = dist[component_mask]
        if len(flat_dist) > 0:
            top_k = min(len(flat_dist), min_core_pixels)
            cutoff = np.partition(flat_dist, -top_k)[-top_k]
            core_mask = (dist >= cutoff) & component_mask

    rim_mask = component_mask & (~core_mask)
    return core_weights, core_mask, rim_mask, max_dist


def analyze_spatial_colocalization(
    red_plane: np.ndarray,
    green_plane: np.ndarray,
    component_mask: np.ndarray,
    core_mask: np.ndarray,
    z_red: np.ndarray,
    z_green: np.ndarray,
    snr_thresh: float = 2.0
) -> Dict[str, Any]:
    """
    Evaluates spatial co-localization vs separation between Red and Green signals within a blob.
    
    Computes:
      - Overlap (IoU) between Red-positive and Green-positive sub-regions.
      - Center-of-mass Euclidean distance between Red and Green positive patches.
      - Largest connected dual-positive core area.
      - Core vs rim mean and median intensities.
    """
    r_pos = (z_red >= snr_thresh) & component_mask
    g_pos = (z_green >= snr_thresh) & component_mask
    dual_pos = r_pos & g_pos

    # 1. IoU Overlap
    intersection = np.sum(dual_pos)
    union = np.sum(r_pos | g_pos)
    iou = float(intersection / max(1, union))

    # 2. Spatial Separation (distance between centers of mass)
    if np.any(r_pos) and np.any(g_pos):
        com_r = ndi.center_of_mass(red_plane * r_pos)
        com_g = ndi.center_of_mass(green_plane * g_pos)
        dist_com = float(np.sqrt((com_r[0] - com_g[0])**2 + (com_r[1] - com_g[1])**2))
    else:
        dist_com = 0.0

    # 3. Largest connected dual-positive core area
    dual_core = dual_pos & core_mask
    if np.any(dual_core):
        labeled_core, num_cores = ndi.label(dual_core, structure=np.ones((3, 3), dtype=bool))
        core_counts = np.bincount(labeled_core.ravel())[1:]
        largest_yellow_core = int(np.max(core_counts)) if len(core_counts) > 0 else 0
    else:
        largest_yellow_core = 0

    # 4. Intensity summaries in Core vs Rim
    core_r = float(np.mean(red_plane[core_mask])) if np.any(core_mask) else 0.0
    core_g = float(np.mean(green_plane[core_mask])) if np.any(core_mask) else 0.0
    rim_r = float(np.mean(red_plane[component_mask & (~core_mask)])) if np.any(component_mask & (~core_mask)) else 0.0
    rim_g = float(np.mean(green_plane[component_mask & (~core_mask)])) if np.any(component_mask & (~core_mask)) else 0.0

    return {
        "colocalization_iou": iou,
        "spatial_separation_dist": dist_com,
        "largest_yellow_core_area": largest_yellow_core,
        "core_red_mean": core_r,
        "core_green_mean": core_g,
        "rim_red_mean": rim_r,
        "rim_green_mean": rim_g,
        "dual_positive_pixel_count": int(intersection)
    }
