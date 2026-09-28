"""Object-level morphometric, photometric, and temporal feature extraction for blob classification."""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion
import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "area",
    "fraction_nucleus_occupied",
    "integrated_excess_intensity",
    "median_contrast",
    "upper_quantile_contrast",
    "peak_contrast",
    "edge_to_background_contrast",
    "spatial_smoothness",
    "solidity",
    "eccentricity",
    "perimeter_to_area_ratio",
    "persistence_cycles",
    "distance_from_nuclear_boundary",
    "saturated_pixel_fraction",
    "component_count",
]

BLOB_FEATURE_NAMES = FEATURE_NAMES + [
    "green_detected",
    "red_detected",
    "green_area",
    "red_area",
    "dist_to_center_norm",
    "ownership_ratio",
    "dapi_under_blob_mean",
    "peak_excess_green",
    "peak_excess_red",
    "max_peak_excess",
    "fold_change_green",
    "fold_change_red",
    "max_fold_change",
    "bg_green",
    "bg_red",
    "noise_green",
    "noise_red",
]


def extract_component_features(
    sbs_stack: np.ndarray,
    background_maps: np.ndarray,
    noise_maps: np.ndarray,
    saturation_masks: np.ndarray,
    blob_mask: np.ndarray,
    nuclear_mask: np.ndarray,
    top_k_map: np.ndarray,
    persistence_map: np.ndarray
) -> pd.DataFrame:
    """Extracts comprehensive object-level features for every segmented blob component."""
    props = regionprops(blob_mask)
    if not props:
        return pd.DataFrame(columns=["component_id", "nucleus_id"] + FEATURE_NAMES)

    in_nuclei = nuclear_mask > 0
    eroded_nuclei = binary_erosion(in_nuclei)
    boundary = in_nuclei & (~eroded_nuclei)
    dist_to_nuc_boundary = ndi.distance_transform_edt(~boundary)

    comp_counts_per_nucleus: Dict[int, int] = {}
    nucleus_of_comp: Dict[int, int] = {}
    for p in props:
        comp_id = p.label
        coords = p.coords
        nuc_id = int(nuclear_mask[coords[0, 0], coords[0, 1]])
        nucleus_of_comp[comp_id] = nuc_id
        comp_counts_per_nucleus[nuc_id] = comp_counts_per_nucleus.get(nuc_id, 0) + 1

    nuc_props_map = {p.label: (p.centroid, p.area) for p in regionprops(nuclear_mask)}

    T, C, H, W = sbs_stack.shape
    any_channel_sat = np.any(saturation_masks, axis=(0, 1))

    records: List[Dict[str, Any]] = []

    for p in props:
        comp_id = p.label
        coords = p.coords
        cy, cx = p.centroid
        nuc_id = nucleus_of_comp[comp_id]
        nuc_info = nuc_props_map.get(nuc_id)
        nuc_area = float(nuc_info[1]) if nuc_info is not None else 1000.0
        area = float(p.area)

        # Morphometrics
        frac_nuc = min(1.0, area / max(1.0, nuc_area))
        solidity = float(p.solidity)
        eccentricity = float(p.eccentricity)
        perimeter = float(p.perimeter)
        perimeter_to_area = perimeter / max(1.0, area)

        # Distance to boundary
        min_dist_boundary = float(np.min(dist_to_nuc_boundary[coords[:, 0], coords[:, 1]]))

        # Photometrics
        blob_snr = top_k_map[coords[:, 0], coords[:, 1]]
        median_contrast = float(np.median(blob_snr))
        p90_contrast = float(np.percentile(blob_snr, 90))
        peak_contrast = float(np.max(blob_snr))

        excess = np.maximum(
            0.0,
            sbs_stack[:, :, coords[:, 0], coords[:, 1]] - background_maps[:, :, coords[:, 0], coords[:, 1]]
        )
        integrated_excess = float(np.sum(excess))

        # Bounding box crop for fast rim ring calculation
        by0, by1 = max(0, p.bbox[0] - 2), min(H, p.bbox[2] + 2)
        bx0, bx1 = max(0, p.bbox[1] - 2), min(W, p.bbox[3] + 2)
        c_crop = blob_mask[by0:by1, bx0:bx1] == comp_id
        dilated_c = binary_dilation(c_crop)
        nuc_crop = nuclear_mask[by0:by1, bx0:bx1] == nuc_id
        rim_ring = dilated_c & (~c_crop) & nuc_crop
        if np.any(rim_ring):
            bg_snr_ring = top_k_map[by0:by1, bx0:bx1][rim_ring]
            edge_to_bg = float(median_contrast / max(1e-2, np.mean(bg_snr_ring)))
        else:
            edge_to_bg = float(median_contrast)

        sub_snr = top_k_map[coords[:, 0], coords[:, 1]]
        spatial_smoothness = float(1.0 / (1.0 + np.var(sub_snr)))

        # Temporal persistence: use 80th percentile so non-full plateaus aren't masked
        blob_persist = persistence_map[coords[:, 0], coords[:, 1]]
        persistence_cycles = float(np.percentile(blob_persist, 80))

        sat_pix_count = np.sum(any_channel_sat[coords[:, 0], coords[:, 1]])
        saturated_frac = float(sat_pix_count / max(1.0, area))

        # Per-channel detection attributes if dual-channel stack is present
        g_detected, r_detected = False, False
        g_area, r_area = 0, 0
        if C >= 2:
            r_excess = np.maximum(0.0, sbs_stack[:, 0, coords[:, 0], coords[:, 1]] - background_maps[:, 0, coords[:, 0], coords[:, 1]])
            g_excess = np.maximum(0.0, sbs_stack[:, 1, coords[:, 0], coords[:, 1]] - background_maps[:, 1, coords[:, 0], coords[:, 1]])
            r_noise = np.maximum(1e-3, noise_maps[:, 0, coords[:, 0], coords[:, 1]])
            g_noise = np.maximum(1e-3, noise_maps[:, 1, coords[:, 0], coords[:, 1]])
            
            r_snr_max = float(np.max(r_excess / r_noise)) if len(coords) > 0 else 0.0
            g_snr_max = float(np.max(g_excess / g_noise)) if len(coords) > 0 else 0.0
            
        peak_excess_r, peak_excess_g = 0.0, 0.0
        bg_r, bg_g = 0.0, 0.0
        noise_r, noise_g = 0.0, 0.0
        fold_change_r, fold_change_g = 1.0, 1.0
        if C >= 2 and len(coords) > 0:
            peak_excess_r = float(np.max(r_excess))
            peak_excess_g = float(np.max(g_excess))
            bg_r = float(np.median(background_maps[:, 0, coords[:, 0], coords[:, 1]]))
            bg_g = float(np.median(background_maps[:, 1, coords[:, 0], coords[:, 1]]))
            noise_r = float(np.median(r_noise))
            noise_g = float(np.median(g_noise))
            fold_change_r = (peak_excess_r + bg_r) / max(1.0, bg_r)
            fold_change_g = (peak_excess_g + bg_g) / max(1.0, bg_g)

        max_peak_excess = max(peak_excess_g, peak_excess_r)
        max_fold_change = max(fold_change_g, fold_change_r)

        # Distance to nuclear center normalized
        if nuc_info is not None:
            nuc_cy, nuc_cx = nuc_info[0]
            nuc_radius = np.sqrt(nuc_area / np.pi)
            dist_to_center_norm = float(np.hypot(cy - nuc_cy, cx - nuc_cx) / max(1.0, nuc_radius))
        else:
            dist_to_center_norm = 0.0

        rec = {
            "component_id": comp_id,
            "nucleus_id": nuc_id,
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "area": area,
            "fraction_nucleus_occupied": frac_nuc,
            "integrated_excess_intensity": integrated_excess,
            "median_contrast": median_contrast,
            "upper_quantile_contrast": p90_contrast,
            "peak_contrast": peak_contrast,
            "edge_to_background_contrast": edge_to_bg,
            "spatial_smoothness": spatial_smoothness,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "perimeter_to_area_ratio": perimeter_to_area,
            "persistence_cycles": persistence_cycles,
            "distance_from_nuclear_boundary": min_dist_boundary,
            "saturated_pixel_fraction": saturated_frac,
            "component_count": float(comp_counts_per_nucleus[nuc_id]),
            "green_detected": g_detected,
            "red_detected": r_detected,
            "green_area": g_area,
            "red_area": r_area,
            "dist_to_center_norm": dist_to_center_norm,
            "ownership_ratio": float(area / max(1.0, nuc_area)),
            "dapi_under_blob_mean": 0.0,
            "peak_excess_green": peak_excess_g,
            "peak_excess_red": peak_excess_r,
            "max_peak_excess": max_peak_excess,
            "fold_change_green": fold_change_g,
            "fold_change_red": fold_change_r,
            "max_fold_change": max_fold_change,
            "bg_green": bg_g,
            "bg_red": bg_r,
            "noise_green": noise_g,
            "noise_red": noise_r,
        }
        records.append(rec)

    return pd.DataFrame(records)


def extract_blob_features_from_crop(
    green_plane: np.ndarray,
    red_plane: np.ndarray,
    blob_mask: np.ndarray,
    energy_map: np.ndarray,
    nuc_mask: Optional[np.ndarray] = None,
    persistence_cycles: float = 1.0,
    green_mask: Optional[np.ndarray] = None,
    red_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts features for a candidate blob directly from crop planes,
    using the exact spot engine from the annotation app.
    """
    from popcrop.annotation.spot_engine import extract_blob_validation_features
    return extract_blob_validation_features(
        green_plane=green_plane,
        red_plane=red_plane,
        blob_mask=blob_mask,
        energy_map=energy_map,
        nuc_mask=nuc_mask,
        persistence_cycles=persistence_cycles,
        green_mask=green_mask,
        red_mask=red_mask
    )
