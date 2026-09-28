"""High-precision local spot/blob segmentation and multi-tier distributional featurization."""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, gaussian
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, disk


def compute_robust_bg_and_noise(plane: np.ndarray, nuc_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Calculates robust local background (median) and MAD noise.
    When a nuclear mask is provided, calculates baseline strictly from the lower 40%
    of intranuclear pixels, ensuring an uncorrupted chromatin/autofluorescence floor
    even in the presence of bright fluorophore amplicons.
    """
    p = np.asarray(plane, dtype=np.float32)
    if nuc_mask is not None and np.any(nuc_mask):
        sample = p[nuc_mask]
        if len(sample) > 0:
            p40 = float(np.percentile(sample, 40))
            lower_half = sample[sample <= p40]
            if len(lower_half) > 0:
                med = float(np.median(lower_half))
                mad = float(np.median(np.abs(lower_half - med)))
                noise = max(1e-3, 1.4826 * mad)
                return med, noise
            med = float(np.median(sample))
            mad = float(np.median(np.abs(sample - med)))
            return med, max(1e-3, 1.4826 * mad)

    if nuc_mask is not None and np.any(~nuc_mask):
        sample = p[~nuc_mask]
    else:
        sample = p.ravel()

    if len(sample) == 0:
        return 0.0, 1.0

    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    noise = max(1e-3, 1.4826 * mad)
    return med, noise


def segment_channel_spot(
    channel_plane: np.ndarray,
    nuc_mask: Optional[np.ndarray] = None,
    high_thresh: float = 3.5,
    low_thresh: float = 1.8,
    min_area: int = 6,
    max_area_fraction: float = 0.60,
    alpha: float = 0.28,
    min_excess: float = 650.0,
    min_snr: float = 4.0,
    min_p95_excess: float = 400.0,
    min_fold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Performs high-precision single-channel spot segmentation within the nucleus.
    Calibrates baseline against the nuclear interior, verifies spot existence via
    photometric gating (excess, SNR, 95th percentile contrast), and delineates a
    tight, continuous contour hugging the amplicon plateau.

    Returns:
      spot_mask: 2D boolean mask of the primary segmented spot in this channel
      z_map: 2D float32 positive z-score map
      stats: dictionary of spot metrics (detected, area, peak_z, bg, noise, etc.)
    """
    H, W = channel_plane.shape
    if nuc_mask is None:
        cy, cx = H / 2.0, W / 2.0
        r = min(H, W) * 0.38
        y_grid, x_grid = np.ogrid[:H, :W]
        nuc_mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= (r ** 2)
    else:
        nuc_mask = np.asarray(nuc_mask, dtype=bool)

    nuc_area = max(1, int(np.sum(nuc_mask)))

    # 1. Mild spatial smoothing (sigma=1.0) to enforce continuity and reject single-pixel noise spikes
    sm = gaussian(channel_plane.astype(np.float32), sigma=1.0, preserve_range=True)

    # 2. Robust intranuclear background and noise estimation
    bg, noise = compute_robust_bg_and_noise(sm, nuc_mask)
    z_map = np.maximum(0.0, (channel_plane.astype(np.float32) - bg) / noise)

    nuc_vals = sm[nuc_mask]
    if len(nuc_vals) == 0:
        return np.zeros((H, W), dtype=bool), z_map, {
            "spot_detected": False, "spot_area": 0, "nuc_area": nuc_area,
            "fraction_nuc": 0.0, "bg": float(bg), "noise": float(noise),
            "peak_z": 0.0, "mean_z": 0.0
        }

    masked_sm = np.where(nuc_mask, sm, -1e9)
    peak_loc = np.unravel_index(np.argmax(masked_sm), sm.shape)
    peak_val = float(sm[peak_loc])
    peak_excess = peak_val - bg
    peak_snr = peak_excess / noise
    p95_val = float(np.percentile(nuc_vals, 95))
    p95_excess = p95_val - bg
    fold = peak_val / max(1.0, bg)

    # Effective SNR gate allows high_thresh to modulate sensitivity if explicitly passed
    effective_snr_gate = min(min_snr, float(high_thresh))

    # 3. Dual-gated spot presence verification
    has_spot = (
        (peak_excess >= min_excess) and
        (peak_snr >= effective_snr_gate) and
        (p95_excess >= min_p95_excess) and
        (fold >= min_fold)
    )

    spot_mask = np.zeros((H, W), dtype=bool)
    if has_spot:
        # 4. Plateau boundary cutoff with spatial continuity (alpha=0.28 captures full shoulder)
        th = bg + alpha * peak_excess
        th = max(th, bg + 3.0 * noise)

        cand = (sm >= th) & nuc_mask
        labeled, num_comp = ndi.label(cand, structure=np.ones((3, 3), dtype=bool))

        peak_comp = labeled[peak_loc]
        if peak_comp > 0:
            c_mask = (labeled == peak_comp)
        else:
            # Fallback to highest energy component containing signal
            best_c = 0
            best_energy = -1.0
            for c in range(1, num_comp + 1):
                comp_m = (labeled == c)
                e = float(np.sum(sm[comp_m]))
                if e > best_energy:
                    best_energy = e
                    best_c = c
            c_mask = (labeled == best_c) if best_c > 0 else np.zeros((H, W), dtype=bool)

        # Morphological closing to ensure watertight, continuous contour
        if np.any(c_mask):
            c_mask = binary_closing(c_mask, footprint=disk(1)) & nuc_mask

        area = int(np.sum(c_mask))
        if min_area <= area <= int(max_area_fraction * nuc_area):
            spot_mask = c_mask
            has_spot = True
        else:
            has_spot = False

    stats = {
        "spot_detected": bool(has_spot),
        "total_components": 1 if has_spot else 0,
        "spot_area": int(np.sum(spot_mask)),
        "nuc_area": nuc_area,
        "fraction_nuc": float(np.sum(spot_mask) / nuc_area) if nuc_area > 0 else 0.0,
        "bg": float(bg),
        "noise": float(noise),
        "peak_raw": peak_val,
        "peak_excess": peak_excess,
        "peak_z": float(peak_snr) if has_spot else 0.0,
        "mean_z": float(np.mean(z_map[spot_mask])) if np.any(spot_mask) else 0.0
    }

    return spot_mask, z_map, stats


def segment_cycle_spots(
    green_plane: np.ndarray,
    red_plane: np.ndarray,
    nuc_mask: Optional[np.ndarray] = None,
    high_thresh: float = 3.5,
    low_thresh: float = 1.8,
    min_area: int = 6,
    max_area_fraction: float = 0.60,
    alpha: float = 0.28,
    min_excess: float = 650.0,
    min_snr: float = 4.0,
    min_p95_excess: float = 400.0,
    min_fold: float = 2.0,
    green_min_excess: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs independent spot detection for the Green channel and Red channel for a cycle.
    In optical pooled screening (A=Green only, T=Red only, C=Both, G=Neither),
    spots shift, scale, or vanish independently in each channel.
    """
    g_excess_gate = float(green_min_excess) if green_min_excess is not None else float(min_excess)
    g_mask, g_z, g_stats = segment_channel_spot(
        channel_plane=green_plane,
        nuc_mask=nuc_mask,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        min_area=min_area,
        max_area_fraction=max_area_fraction,
        alpha=alpha,
        min_excess=g_excess_gate,
        min_snr=min_snr,
        min_p95_excess=min_p95_excess,
        min_fold=min_fold
    )

    r_mask, r_z, r_stats = segment_channel_spot(
        channel_plane=red_plane,
        nuc_mask=nuc_mask,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        min_area=min_area,
        max_area_fraction=max_area_fraction,
        alpha=alpha,
        min_excess=min_excess,
        min_snr=min_snr,
        min_p95_excess=min_p95_excess,
        min_fold=min_fold
    )

    composite_mask = g_mask | r_mask
    energy_map = np.sqrt(g_z ** 2 + r_z ** 2).astype(np.float32)

    return {
        "green_mask": g_mask,
        "green_z": g_z,
        "green_stats": g_stats,
        "green_detected": bool(g_stats["spot_detected"]),
        "green_area": int(g_stats["spot_area"]),
        "red_mask": r_mask,
        "red_z": r_z,
        "red_stats": r_stats,
        "red_detected": bool(r_stats["spot_detected"]),
        "red_area": int(r_stats["spot_area"]),
        "composite_mask": composite_mask,
        "composite_detected": bool(g_stats["spot_detected"] or r_stats["spot_detected"]),
        "composite_area": int(np.sum(composite_mask)),
        "energy_map": energy_map,
        "any_blob_detected": bool(g_stats["spot_detected"] or r_stats["spot_detected"]),
        "both_blobs_detected": bool(g_stats["spot_detected"] and r_stats["spot_detected"])
    }


def segment_crop_blob(
    green_plane: np.ndarray,
    red_plane: np.ndarray,
    nuc_mask: Optional[np.ndarray] = None,
    high_thresh: float = 4.0,
    low_thresh: float = 2.0,
    min_area: int = 6,
    max_area_fraction: float = 0.60,
    alpha: float = 0.28,
    min_excess: float = 650.0,
    min_snr: float = 4.0,
    min_p95_excess: float = 400.0,
    min_fold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Dual-channel segmentation wrapper for backward compatibility.
    Runs independent per-channel segmentation and returns composite mask and stats.
    """
    cycle_res = segment_cycle_spots(
        green_plane=green_plane,
        red_plane=red_plane,
        nuc_mask=nuc_mask,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        min_area=min_area,
        max_area_fraction=max_area_fraction,
        alpha=alpha,
        min_excess=min_excess,
        min_snr=min_snr,
        min_p95_excess=min_p95_excess,
        min_fold=min_fold
    )

    comp_mask = cycle_res["composite_mask"]
    energy_map = cycle_res["energy_map"]
    g_stats = cycle_res["green_stats"]
    r_stats = cycle_res["red_stats"]

    stats = {
        "blob_detected": cycle_res["any_blob_detected"],
        "green_detected": g_stats["spot_detected"],
        "red_detected": r_stats["spot_detected"],
        "green_area": g_stats["spot_area"],
        "red_area": r_stats["spot_area"],
        "blob_area": int(np.sum(comp_mask)),
        "nuc_area": g_stats["nuc_area"],
        "bg_green": g_stats["bg"],
        "noise_green": g_stats["noise"],
        "bg_red": r_stats["bg"],
        "noise_red": r_stats["noise"],
        "peak_energy": float(np.max(energy_map[comp_mask])) if np.any(comp_mask) else 0.0
    }

    return comp_mask, energy_map, stats


def extract_blob_contours(
    blob_mask: Optional[np.ndarray] = None,
    nuc_mask: Optional[np.ndarray] = None,
    green_mask: Optional[np.ndarray] = None,
    red_mask: Optional[np.ndarray] = None
) -> Dict[str, List[List[List[float]]]]:
    """
    Extracts boundary polygon coordinates for canvas rendering in the browser.
    Returns: {
      "green_blob": [[[x, y], ...]],
      "red_blob": [[[x, y], ...]],
      "blob": [[[x, y], ...]],
      "nuc": [[[x, y], ...]]
    }
    """
    contours_dict: Dict[str, List[List[List[float]]]] = {
        "green_blob": [],
        "red_blob": [],
        "blob": [],
        "nuc": []
    }

    def _extract_poly(mask: Optional[np.ndarray]) -> List[List[List[float]]]:
        if mask is None or not np.any(mask):
            return []
        out = []
        raw_cnts = find_contours(mask.astype(float), 0.5)
        for c in raw_cnts:
            step = max(1, len(c) // 120)
            pts = [[round(float(pt[1]), 1), round(float(pt[0]), 1)] for pt in c[::step]]
            if len(pts) >= 3:
                out.append(pts)
        return out

    contours_dict["green_blob"] = _extract_poly(green_mask)
    contours_dict["red_blob"] = _extract_poly(red_mask)
    contours_dict["blob"] = _extract_poly(blob_mask)
    contours_dict["nuc"] = _extract_poly(nuc_mask)

    return contours_dict


def extract_basecaller_distribution_features(
    green_plane: np.ndarray,
    red_plane: np.ndarray,
    blob_mask: Optional[np.ndarray] = None,
    nuc_mask: Optional[np.ndarray] = None,
    green_blob_mask: Optional[np.ndarray] = None,
    red_blob_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts multi-tier distributional features for the Base Caller.
    In optical pooled screening (A=Green, T=Red, C=Both, G=Neither):
      - Green distribution is measured within green_blob_mask if present
      - Red distribution is measured within red_blob_mask if present
      - If no blob is detected in a channel, falls back to nucleus/nuclear center
    """
    # Resolve green eval mask
    if green_blob_mask is not None and np.any(green_blob_mask):
        g_eval_mask = green_blob_mask
        g_has_blob = True
    elif blob_mask is not None and np.any(blob_mask):
        g_eval_mask = blob_mask
        g_has_blob = True
    else:
        g_eval_mask = nuc_mask if (nuc_mask is not None and np.any(nuc_mask)) else np.ones_like(green_plane, dtype=bool)
        g_has_blob = False

    # Resolve red eval mask
    if red_blob_mask is not None and np.any(red_blob_mask):
        r_eval_mask = red_blob_mask
        r_has_blob = True
    elif blob_mask is not None and np.any(blob_mask):
        r_eval_mask = blob_mask
        r_has_blob = True
    else:
        r_eval_mask = nuc_mask if (nuc_mask is not None and np.any(nuc_mask)) else np.ones_like(red_plane, dtype=bool)
        r_has_blob = False

    g_vals = green_plane[g_eval_mask].astype(np.float64)
    r_vals = red_plane[r_eval_mask].astype(np.float64)

    # Nuclear background reference
    if nuc_mask is not None and np.any(nuc_mask):
        bg_g_mask = nuc_mask & (~g_eval_mask)
        nuc_g_vals = green_plane[bg_g_mask].astype(np.float64) if np.any(bg_g_mask) else g_vals
        bg_r_mask = nuc_mask & (~r_eval_mask)
        nuc_r_vals = red_plane[bg_r_mask].astype(np.float64) if np.any(bg_r_mask) else r_vals
    else:
        nuc_g_vals = g_vals
        nuc_r_vals = r_vals

    def calc_channel_distribution(vals: np.ndarray, nuc_bg_vals: np.ndarray) -> Dict[str, Any]:
        if len(vals) == 0:
            return {
                "min": 0.0, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0,
                "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                "iqr": 0.0, "trimmed_mean": 0.0, "upper_core_median": 0.0,
                "nuc_median": 0.0, "excess_median": 0.0, "integrated_excess": 0.0,
                "histogram": {"counts": [], "bin_edges": []}, "samples": []
            }

        p10, p25, p50, p75, p90, p95 = np.percentile(vals, [10, 25, 50, 75, 90, 95])
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))

        # Trimmed Mean: Average of values between 50th and 90th percentile (the stable plateau core)
        core_window = vals[(vals >= p50) & (vals <= p90)]
        trimmed_mean = float(np.mean(core_window)) if len(core_window) > 0 else float(p50)
        upper_core_med = float(p75)

        # Contrast over nuclear background
        nuc_med = float(np.median(nuc_bg_vals)) if len(nuc_bg_vals) > 0 else 0.0
        excess_med = max(0.0, float(p50) - nuc_med)
        integrated_excess = float(np.sum(np.maximum(0.0, vals - nuc_med)))

        # Tier 2: 32-bin histogram across [min_v, max_v]
        hist_range = (min_v, max_v if max_v > min_v else min_v + 1.0)
        counts, bin_edges = np.histogram(vals, bins=32, range=hist_range)

        # Tier 2 sample representation: up to 64 representative sample values
        step = max(1, len(vals) // 64)
        samples = [round(float(v), 1) for v in vals[::step][:64]]

        return {
            "min": round(min_v, 1),
            "p10": round(float(p10), 1),
            "p25": round(float(p25), 1),
            "median": round(float(p50), 1),
            "p75": round(float(p75), 1),
            "p90": round(float(p90), 1),
            "p95": round(float(p95), 1),
            "max": round(max_v, 1),
            "mean": round(mean_v, 1),
            "std": round(std_v, 1),
            "iqr": round(float(p75 - p25), 1),
            "trimmed_mean": round(trimmed_mean, 1),
            "upper_core_median": round(upper_core_med, 1),
            "nuc_median": round(nuc_med, 1),
            "excess_median": round(excess_med, 1),
            "integrated_excess": round(integrated_excess, 1),
            "histogram": {
                "counts": counts.tolist(),
                "bin_edges": [round(float(b), 1) for b in bin_edges]
            },
            "samples": samples
        }

    green_dist = calc_channel_distribution(g_vals, nuc_g_vals)
    red_dist = calc_channel_distribution(r_vals, nuc_r_vals)

    # Cross-channel distribution ratios
    rg_ratio_median = round(red_dist["median"] / max(1.0, green_dist["median"]), 3)
    rg_ratio_trimmed = round(red_dist["trimmed_mean"] / max(1.0, green_dist["trimmed_mean"]), 3)

    comp_mask = blob_mask if blob_mask is not None else (g_eval_mask | r_eval_mask)

    return {
        "has_blob": bool(g_has_blob or r_has_blob),
        "green_has_blob": bool(g_has_blob),
        "red_has_blob": bool(r_has_blob),
        "blob_area": int(np.sum(comp_mask)),
        "green_blob_area": int(np.sum(green_blob_mask)) if green_blob_mask is not None else 0,
        "red_blob_area": int(np.sum(red_blob_mask)) if red_blob_mask is not None else 0,
        "green": green_dist,
        "red": red_dist,
        "rg_ratio_median": rg_ratio_median,
        "rg_ratio_trimmed": rg_ratio_trimmed
    }


def extract_blob_validation_features(
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
    Extracts comprehensive object-level morphometric, photometric, and quality
    features for the Blob Signal Validator (Real Blob vs Noise/Artifact).
    Now includes per-channel detection attributes (Green spot vs Red spot).
    """
    has_blob = np.any(blob_mask)
    g_has_spot = bool(green_mask is not None and np.any(green_mask))
    r_has_spot = bool(red_mask is not None and np.any(red_mask))

    if not has_blob and not g_has_spot and not r_has_spot:
        return {
            "has_blob": False,
            "green_detected": False,
            "red_detected": False,
            "green_area": 0,
            "red_area": 0,
            "area": 0,
            "fraction_nucleus_occupied": 0.0,
            "median_contrast": 0.0,
            "peak_contrast": 0.0,
            "upper_quantile_contrast": 0.0,
            "integrated_excess_intensity": 0.0,
            "spatial_smoothness": 0.0,
            "solidity": 0.0,
            "eccentricity": 0.0,
            "perimeter_to_area_ratio": 0.0,
            "persistence_cycles": float(persistence_cycles),
            "distance_from_nuclear_boundary": 0.0,
            "saturated_pixel_fraction": 0.0,
            "component_count": 0,
            "peak_excess_green": 0.0,
            "peak_excess_red": 0.0,
            "max_peak_excess": 0.0,
            "fold_change_green": 1.0,
            "fold_change_red": 1.0,
            "max_fold_change": 1.0,
            "bg_green": 0.0,
            "bg_red": 0.0,
            "noise_green": 0.0,
            "noise_red": 0.0,
            "p_real_blob": 0.0
        }

    effective_mask = blob_mask if has_blob else ((green_mask if g_has_spot else 0) | (red_mask if r_has_spot else 0))
    area = int(np.sum(effective_mask))
    nuc_area = int(np.sum(nuc_mask)) if (nuc_mask is not None and np.any(nuc_mask)) else area * 3
    frac_nuc = min(1.0, float(area / max(1, nuc_area)))

    # Photometrics on energy map
    blob_energy = energy_map[effective_mask]
    median_contrast = float(np.median(blob_energy)) if len(blob_energy) > 0 else 0.0
    peak_contrast = float(np.max(blob_energy)) if len(blob_energy) > 0 else 0.0
    p90_contrast = float(np.percentile(blob_energy, 90.0)) if len(blob_energy) > 0 else 0.0
    integrated_excess = float(np.sum(blob_energy)) if len(blob_energy) > 0 else 0.0
    spatial_smoothness = float(1.0 / (1.0 + np.var(blob_energy))) if len(blob_energy) > 0 else 0.0

    # Morphometrics via regionprops
    props = regionprops(effective_mask.astype(np.int32))
    if props:
        p = props[0]
        solidity = float(p.solidity)
        eccentricity = float(p.eccentricity)
        perimeter = float(p.perimeter)
        perimeter_to_area = float(perimeter / max(1.0, area))
    else:
        solidity = 0.8
        eccentricity = 0.5
        perimeter_to_area = 0.2

    # Distance to nuclear boundary
    if nuc_mask is not None and np.any(nuc_mask):
        eroded = binary_erosion(nuc_mask)
        boundary = nuc_mask & (~eroded)
        dist_to_boundary = ndi.distance_transform_edt(~boundary)
        coords = np.argwhere(effective_mask)
        min_dist_boundary = float(np.min(dist_to_boundary[coords[:, 0], coords[:, 1]])) if len(coords) > 0 else 3.0
    else:
        min_dist_boundary = 3.0

    # Saturation
    sat_g = green_plane >= 65000
    sat_r = red_plane >= 65000
    sat_pix = np.sum((sat_g | sat_r) & effective_mask)
    saturated_frac = float(sat_pix / max(1, area))

    # Channel-specific background and raw photometrics
    bg_g, noise_g = compute_robust_bg_and_noise(green_plane, nuc_mask)
    bg_r, noise_r = compute_robust_bg_and_noise(red_plane, nuc_mask)

    if np.any(effective_mask):
        peak_raw_g = float(np.max(green_plane[effective_mask]))
        peak_raw_r = float(np.max(red_plane[effective_mask]))
    else:
        peak_raw_g = float(bg_g)
        peak_raw_r = float(bg_r)

    peak_excess_g = max(0.0, peak_raw_g - bg_g)
    peak_excess_r = max(0.0, peak_raw_r - bg_r)
    max_peak_excess = max(peak_excess_g, peak_excess_r)

    fold_change_g = peak_raw_g / max(1.0, bg_g)
    fold_change_r = peak_raw_r / max(1.0, bg_r)
    max_fold_change = max(fold_change_g, fold_change_r)

    # Analytical P(real_blob) calibrated score
    snr_term = 1.0 / (1.0 + np.exp(-1.0 * (median_contrast - 2.5)))
    persist_term = np.clip(float(persistence_cycles) / 2.0, 0.0, 1.0)
    area_term = np.clip(area / 8.0, 0.0, 1.0)
    solidity_term = np.clip(solidity, 0.4, 1.0)
    border_term = np.clip(0.85 + 0.15 * (min_dist_boundary / 2.0), 0.85, 1.0)
    sat_penalty = 1.0 - 0.4 * saturated_frac

    p_real = (
        0.50 * snr_term
        + 0.25 * persist_term
        + 0.15 * (p90_contrast / (p90_contrast + 2.5))
        + 0.10 * (0.5 * area_term + 0.5 * solidity_term)
    ) * border_term * sat_penalty
    p_real = float(np.clip(p_real, 0.0, 1.0))

    return {
        "has_blob": True,
        "green_detected": g_has_spot,
        "red_detected": r_has_spot,
        "green_area": int(np.sum(green_mask)) if green_mask is not None else 0,
        "red_area": int(np.sum(red_mask)) if red_mask is not None else 0,
        "area": area,
        "fraction_nucleus_occupied": round(frac_nuc, 3),
        "median_contrast": round(median_contrast, 2),
        "peak_contrast": round(peak_contrast, 2),
        "upper_quantile_contrast": round(p90_contrast, 2),
        "integrated_excess_intensity": round(integrated_excess, 1),
        "spatial_smoothness": round(spatial_smoothness, 3),
        "solidity": round(solidity, 3),
        "eccentricity": round(eccentricity, 3),
        "perimeter_to_area_ratio": round(perimeter_to_area, 3),
        "persistence_cycles": float(persistence_cycles),
        "distance_from_nuclear_boundary": round(min_dist_boundary, 1),
        "saturated_pixel_fraction": round(saturated_frac, 3),
        "component_count": (1 if g_has_spot else 0) + (1 if r_has_spot else 0),
        "peak_excess_green": round(peak_excess_g, 1),
        "peak_excess_red": round(peak_excess_r, 1),
        "max_peak_excess": round(max_peak_excess, 1),
        "fold_change_green": round(fold_change_g, 2),
        "fold_change_red": round(fold_change_r, 2),
        "max_fold_change": round(max_fold_change, 2),
        "bg_green": round(bg_g, 1),
        "bg_red": round(bg_r, 1),
        "noise_green": round(noise_g, 2),
        "noise_red": round(noise_r, 2),
        "p_real_blob": round(p_real, 3)
    }
