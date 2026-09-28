"""Shared raw-intensity analysis used by the annotation UI and batch pipeline.

The circle is an analysis ROI, not a segmentation of the biological nucleus.
No display LUT, illumination correction, or image-wide background enters this
contract. Masks and image crops are local, so callers can discard them per cell.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

from .spot_engine import (
    segment_cycle_spots, extract_blob_contours,
    extract_basecaller_distribution_features, extract_blob_validation_features,
)

ANALYSIS_VERSION = "manual_raw_circle_v1"
DEFAULT_DETECTOR_PARAMS = {
    "high_thresh": 4.0, "low_thresh": 2.0, "min_area": 6,
    "max_area_fraction": 0.60, "alpha": 0.28, "min_excess": 650.0,
    "green_min_excess": 3500.0, "min_snr": 4.0,
    "min_p95_excess": 400.0, "min_fold": 2.0,
}


def anchor_nuclear_center(dapi_crop, reference_dapi_crop, center_yx, radius):
    """Reproduce the annotator's per-cycle DAPI centroid anchoring exactly."""
    cy, cx = map(float, center_yx)
    dapi_ref = dapi_crop
    if dapi_ref is None or not dapi_ref.size or dapi_ref.max() <= dapi_ref.min() + 10:
        dapi_ref = reference_dapi_crop
    if dapi_ref is None or not dapi_ref.size or dapi_ref.max() <= dapi_ref.min() + 10:
        return cy, cx
    d_mask = remove_small_objects(
        ndi.binary_fill_holes(dapi_ref > threshold_otsu(dapi_ref)), min_size=80,
    )
    d_labeled, count = ndi.label(d_mask)
    if count:
        _, indices = ndi.distance_transform_edt(d_mask == 0, return_indices=True)
        iy = int(np.clip(round(cy), 0, d_mask.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, d_mask.shape[1] - 1))
        label_id = d_labeled[indices[0, iy, ix], indices[1, iy, ix]]
        selected = d_labeled == label_id
        if label_id > 0 and selected.sum() >= 80:
            ny, nx = ndi.center_of_mass(selected)
            if np.hypot(ny - cy, nx - cx) <= radius:
                return float(ny), float(nx)
    return cy, cx


def flatten_base_features(features):
    """Preserve every scalar statistic, histogram and representative sample."""
    result = {k: v for k, v in features.items() if k not in ("green", "red")}
    for channel in ("green", "red"):
        for name, value in features[channel].items():
            if name not in ("histogram", "samples"):
                result[f"{channel}_{name}"] = value
        result[f"{channel}_distribution"] = features[channel]["histogram"]
        result[f"{channel}_samples"] = features[channel]["samples"]
    return result


def analyze_nucleus_cycle(
    green, red, dapi, center_yx, nuclear_diameter=55.0,
    reference_dapi=None, radius_scale=1.6, detector_params=None,
    neighbor_centers_yx=None, include_contours=False,
):
    """Analyze one cycle from raw full-image channel planes without copying them.

    ``neighbor_centers_yx`` contains other nuclei only, in full-image coordinates.
    Returns UI-compatible features, flattened base model inputs, local masks,
    crop views, and exact analysis provenance. Spatial features use the anchored
    ROI center and current-cycle DAPI consistently at training and inference.
    """
    green, red = np.asarray(green), np.asarray(red)
    if green.ndim != 2 or red.shape != green.shape:
        raise ValueError("Green and red must be same-shape 2D raw planes")
    for plane in (dapi, reference_dapi):
        if plane is not None and np.shape(plane) != green.shape:
            raise ValueError("DAPI planes must share the raw image coordinates")
    if not np.isfinite(nuclear_diameter) or nuclear_diameter <= 0:
        raise ValueError("nuclear_diameter must be finite and positive")
    if not np.isfinite(radius_scale) or radius_scale < 1:
        raise ValueError("radius_scale must be finite and at least 1")
    cy, cx = map(float, center_yx)
    if not np.isfinite([cy, cx]).all() or not (0 <= cy < green.shape[0] and 0 <= cx < green.shape[1]):
        raise ValueError("Nucleus center must lie inside the image")
    params = dict(DEFAULT_DETECTOR_PARAMS)
    if detector_params:
        unknown = set(detector_params) - set(params)
        if unknown:
            raise ValueError(f"Unknown detector parameters: {sorted(unknown)}")
        params.update(detector_params)
    if not all(np.isfinite(value) for value in params.values()):
        raise ValueError("Detector settings must be finite")
    if not 0 < params["max_area_fraction"] <= 1 or not 0 < params["alpha"] <= 1:
        raise ValueError("Area fraction and plateau alpha must be in (0, 1]")
    if params["min_area"] < 1 or int(params["min_area"]) != params["min_area"]:
        raise ValueError("Minimum spot area must be a positive integer")
    if any(params[key] < 0 for key in ("high_thresh", "low_thresh", "min_excess", "green_min_excess", "min_snr", "min_p95_excess", "min_fold")):
        raise ValueError("Detection gates cannot be negative")
    radius = max(8.0, nuclear_diameter / 2.0)
    crop_radius = max(24, int(round(nuclear_diameter * radius_scale / 2.0)))
    yc, xc = int(round(cy)), int(round(cx))
    y0, y1 = max(0, yc - crop_radius), min(green.shape[0], yc + crop_radius)
    x0, x1 = max(0, xc - crop_radius), min(green.shape[1], xc + crop_radius)
    bounds = np.s_[y0:y1, x0:x1]
    g_crop, r_crop = green[bounds], red[bounds]
    b_crop = dapi[bounds] if dapi is not None else np.zeros(g_crop.shape, dtype=green.dtype)
    ref_crop = reference_dapi[bounds] if reference_dapi is not None else b_crop
    anchored_y, anchored_x = anchor_nuclear_center(b_crop, ref_crop, (yc-y0, xc-x0), radius)
    yy, xx = np.ogrid[:g_crop.shape[0], :g_crop.shape[1]]
    nuc_mask = (yy-anchored_y)**2 + (xx-anchored_x)**2 <= radius**2
    spots = segment_cycle_spots(g_crop, r_crop, nuc_mask=nuc_mask, **params)
    g_mask, r_mask, union = spots["green_mask"], spots["red_mask"], spots["composite_mask"]
    base = extract_basecaller_distribution_features(
        g_crop, r_crop, union, nuc_mask, g_mask, r_mask,
    )
    blob = extract_blob_validation_features(
        g_crop, r_crop, union, spots["energy_map"], nuc_mask,
        green_mask=g_mask, red_mask=r_mask,
    )
    if union.any():
        sy, sx = ndi.center_of_mass(union)
        dist_target = float(np.hypot(sy-anchored_y, sx-anchored_x))
        dist_norm = dist_target / radius
        neighbors = np.asarray(neighbor_centers_yx if neighbor_centers_yx is not None else [], dtype=float).reshape(-1, 2)
        nearest = float(np.hypot(neighbors[:, 0]-(sy+y0), neighbors[:, 1]-(sx+x0)).min()) if len(neighbors) else None
        # Missing neighbors is explicit. Batch/training must supply the same context.
        ownership = dist_target / max(1e-2, nearest) if nearest is not None else np.nan
        dapi_mean = float(np.mean(b_crop[union]))
    else:
        dist_norm, ownership = 0.0, 0.0
        dapi_mean = float(np.mean(b_crop[nuc_mask])) if nuc_mask.any() else 0.0
    spatial = {"dist_to_center_norm": round(dist_norm, 4),
               "ownership_ratio": round(ownership, 4),
               "dapi_under_blob_mean": round(dapi_mean, 1)}
    base.update(spatial)
    blob.update(spatial)
    detection = {
        "green_detected": spots["green_detected"], "red_detected": spots["red_detected"],
        "green_area": spots["green_area"], "red_area": spots["red_area"],
        "green_peak_z": float(spots["green_stats"]["peak_z"]),
        "red_peak_z": float(spots["red_stats"]["peak_z"]),
        "any_detected": spots["any_blob_detected"], "both_detected": spots["both_blobs_detected"],
        "composite_detected": spots["composite_detected"], "composite_area": spots["composite_area"],
    }
    crop_box = {"y_min": y0, "y_max": y1, "x_min": x0, "x_max": x1,
                "width": x1-x0, "height": y1-y0, "center_x": anchored_x,
                "center_y": anchored_y, "nuclear_diameter": float(nuclear_diameter)}
    provenance = {"analysis_version": ANALYSIS_VERSION, "input_domain": "raw",
                  "nuclear_diameter": float(nuclear_diameter), "radius_scale": float(radius_scale),
                  "nuclear_area": int(nuc_mask.sum()), "center_input_yx": [cy, cx],
                  "center_anchored_yx": [anchored_y+y0, anchored_x+x0],
                  "detector_params": params, "crop_box": crop_box}
    return {
        "g_crop": g_crop, "r_crop": r_crop, "b_crop": b_crop, "c0_b_crop": ref_crop,
        "crop_box": crop_box, "provenance": provenance,
        "base_features": base, "base_row": flatten_base_features(base), "blob_features": blob,
        "detection_summary": detection, "detection": detection,
        "segmentation_stats": {"blob_detected": spots["any_blob_detected"],
                               "green_detected": spots["green_detected"], "red_detected": spots["red_detected"],
                               "green_area": spots["green_area"], "red_area": spots["red_area"],
                               "blob_area": spots["composite_area"], "nuc_area": int(nuc_mask.sum())},
        "nuc_mask": nuc_mask, "green_mask": g_mask, "red_mask": r_mask, "composite_mask": union,
        "green_stats": spots["green_stats"], "red_stats": spots["red_stats"],
        "contours": extract_blob_contours(union, nuc_mask, g_mask, r_mask) if include_contours else {},
    }
