"""Core-aware color calling engine with calibrated per-cycle probabilities."""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging

from pathlib import Path
import pandas as pd
from scipy import ndimage as ndi
from popcrop.color_calling.core_rim import compute_core_rim_geometry, analyze_spatial_colocalization
from popcrop.models.trainer import load_trained_model, BASE_DEFAULT_FEATURES
from popcrop.annotation.spot_engine import extract_basecaller_distribution_features

logger = logging.getLogger(__name__)

COLOR_CLASSES = ["red", "green", "yellow", "dark", "ambiguous"]
BASE_TO_COLOR = {
    "A": "green",
    "T": "red",
    "C": "yellow",
    "G": "dark",
}


class ModelBasedBaseCaller:
    """Evaluates base calling probabilities from distribution features using a trained pipeline."""

    def __init__(self, model_path: Optional[str | Path] = None, strict_features: bool = False):
        self.strict_features = strict_features
        self.pipeline = None
        self.metadata = {}
        self.feature_names = list(BASE_DEFAULT_FEATURES)
        self.classes = ["A", "C", "G", "T"]
        if model_path is not None and Path(model_path).exists():
            self.load(model_path)

    def load(self, model_path: Union[str, Path]) -> None:
        self.pipeline, self.metadata = load_trained_model(model_path)
        if "feature_names" in self.metadata:
            self.feature_names = self.metadata["feature_names"]
        if hasattr(self.pipeline, "classes_"):
            self.classes = [str(c) for c in self.pipeline.classes_]
        logger.info(f"Loaded trained base caller model from {model_path} with classes: {self.classes}")

    def predict_cycle_probabilities(self, df_features: pd.DataFrame) -> List[Dict[str, float]]:
        """Predicts calibrated base probabilities for each row and maps them to color classes."""
        if self.pipeline is None or df_features.empty:
            return []

        # Prepare X with all expected features
        X_df = pd.DataFrame(index=df_features.index)
        for f in self.feature_names:
            if f in df_features.columns:
                X_df[f] = df_features[f].astype(float)
            else:
                if self.strict_features:
                    raise ValueError(f"Required base model feature is missing: {f}")
                X_df[f] = 0.0

        probs = self.pipeline.predict_proba(X_df.values)
        out = []
        for p_row in probs:
            base_p = {c: float(p) for c, p in zip(self.classes, p_row)}
            color_p = {
                "green": base_p.get("A", 0.0),
                "red": base_p.get("T", 0.0),
                "yellow": base_p.get("C", 0.0),
                "dark": base_p.get("G", 0.0),
                "ambiguous": 0.0,
            }
            # Normalize
            tot = sum(color_p.values())
            if tot > 0:
                color_p = {k: v / tot for k, v in color_p.items()}
            out.append(color_p)
        return out


class CoreAwareColorCaller:
    """
    Infers base color probabilities for validated fluorescent blobs.
    Enforces spatial core dominance, dual-positive connectivity, and separates spatial artifacts.
    Supports seamless execution with a trained Base Caller model or calibrated analytical fallback.
    """

    def __init__(
        self,
        min_connected_yellow_core: int = 4,
        min_colocalization_overlap: float = 0.20,
        max_spatial_separation: float = 4.0,
        yellow_core_min_ratio: float = 0.20,
        green_rim_override_protection: bool = True,
        calibrated_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str | Path] = None
    ):
        self.min_connected_yellow_core = int(min_connected_yellow_core)
        self.min_colocalization_overlap = float(min_colocalization_overlap)
        self.max_spatial_separation = float(max_spatial_separation)
        self.yellow_core_min_ratio = float(yellow_core_min_ratio)
        self.green_rim_override_protection = bool(green_rim_override_protection)
        self.params = calibrated_params or {
            "expected_green_red_ratio_yellow": 2.5,
            "dark_snr_cutoff": 2.0,
        }
        self.model_caller: Optional[ModelBasedBaseCaller] = None
        if model_path is not None and Path(model_path).exists():
            self.model_caller = ModelBasedBaseCaller(model_path)

    def call_blob_cycle_probabilities(
        self,
        sbs_stack: np.ndarray,
        background_maps: np.ndarray,
        noise_maps: np.ndarray,
        z_scores: np.ndarray,
        component_mask: np.ndarray,
        p_real_blob: float = 1.0,
        prob_threshold: float = 0.50,
        nuclear_mask: Optional[np.ndarray] = None,
        component_properties: Optional[Dict[str, Any]] = None,
        precomputed_features: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, float]]:
        """Calculates calibrated per-cycle probabilities for a validated blob component."""
        T, C, H, W = sbs_stack.shape

        # Rule 1: Only call color for validated blobs.
        if p_real_blob < prob_threshold or not np.any(component_mask):
            return [
                {"red": 0.0, "green": 0.0, "yellow": 0.0, "dark": 0.5, "ambiguous": 0.5}
                for _ in range(T)
            ]

        # Mode A: Model-Based Base Caller if a trained model is loaded
        if self.model_caller is not None and self.model_caller.pipeline is not None and C >= 2:
            if precomputed_features is not None:
                feat_rows = precomputed_features
            else:
                feat_rows = self.extract_cycle_distribution_features(
                    sbs_stack,
                    component_mask,
                    nuclear_mask=nuclear_mask,
                    component_properties=component_properties
                )
            df_dist = pd.DataFrame(feat_rows)
            return self.model_caller.predict_cycle_probabilities(df_dist)

        # Mode B: Core/Rim Spatial Geometry Calibrated Heuristic
        core_weights, core_mask, rim_mask, max_dist = compute_core_rim_geometry(component_mask)
        core_pixel_count = int(np.sum(core_mask))

        cycle_probs: List[Dict[str, float]] = []

        for t in range(T):
            r_plane = sbs_stack[t, 0]
            g_plane = sbs_stack[t, 1]
            z_r = z_scores[t, 0]
            z_g = z_scores[t, 1]

            dark_thresh = float(self.params["dark_snr_cutoff"])

            # Spatial co-localization metrics inside blob
            coloc = analyze_spatial_colocalization(
                r_plane, g_plane, component_mask, core_mask, z_r, z_g,
                snr_thresh=dark_thresh
            )

            # Core-weighted SNR
            weight_sum = np.sum(core_weights[component_mask])
            safe_weights = core_weights / max(1e-4, weight_sum)
            w_snr_r = float(np.sum(z_r[component_mask] * safe_weights[component_mask]))
            w_snr_g = float(np.sum(z_g[component_mask] * safe_weights[component_mask]))

            # Active pixels in entire component
            r_active_pixels = int(np.sum((z_r >= dark_thresh) & component_mask))
            g_active_pixels = int(np.sum((z_g >= dark_thresh) & component_mask))

            r_active = r_active_pixels >= 4 or w_snr_r >= dark_thresh
            g_active = g_active_pixels >= 4 or w_snr_g >= dark_thresh

            largest_yellow_core = coloc["largest_yellow_core_area"]
            dist_sep = coloc["spatial_separation_dist"]

            # Compute evidence
            p_red, p_green, p_yellow, p_dark, p_ambiguous = self._evaluate_color_evidence(
                w_snr_r=w_snr_r,
                w_snr_g=w_snr_g,
                r_active_pixels=r_active_pixels,
                g_active_pixels=g_active_pixels,
                r_active=r_active,
                g_active=g_active,
                largest_yellow_core=largest_yellow_core,
                dist_sep=dist_sep,
            )

            # Normalize probabilities to sum to 1.0
            probs = np.array([p_red, p_green, p_yellow, p_dark, p_ambiguous], dtype=np.float32)
            total = float(np.sum(probs))
            if total > 0:
                probs /= total
            else:
                probs = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            cycle_probs.append({
                "red": float(probs[0]),
                "green": float(probs[1]),
                "yellow": float(probs[2]),
                "dark": float(probs[3]),
                "ambiguous": float(probs[4]),
            })

        return cycle_probs

    def _evaluate_color_evidence(
        self,
        w_snr_r: float,
        w_snr_g: float,
        r_active_pixels: int,
        g_active_pixels: int,
        r_active: bool,
        g_active: bool,
        largest_yellow_core: int,
        dist_sep: float,
    ) -> Tuple[float, float, float, float, float]:
        """Evaluates unnormalized posterior evidence applying all critical labeling rules."""

        # 1. DARK STATE: Neither channel active
        if not r_active and not g_active:
            snr_max = max(w_snr_r, w_snr_g)
            p_dark = float(np.exp(-snr_max))
            return 0.01, 0.01, 0.00, p_dark, 0.02

        # 2. YELLOW CORE RULE:
        # "A blob with a spatially coherent yellow center and a stronger or larger green rim must be called yellow."
        # If there is a connected dual-positive core >= min_connected_yellow_core:
        if largest_yellow_core >= self.min_connected_yellow_core:
            # Yellow core is confirmed!
            # High yellow probability, suppressing green halo
            p_yellow = 100.0
            p_green = 0.05
            p_red = 0.05
            return p_red, p_green, p_yellow, 0.001, 0.01

        # 3. SPATIALLY SEPARATE RED AND GREEN REGIONS:
        # If both are active, but disconnected and spatially separated:
        # "Spatially separate red and green regions must not automatically be called yellow; return mixed or ambiguous"
        if r_active and g_active and dist_sep > self.max_spatial_separation and largest_yellow_core == 0:
            return 0.05, 0.05, 0.00, 0.01, 0.89

        # 4. TINY YELLOW SPECK ON GREEN BLOB:
        # "A tiny yellow speck must not override an otherwise green blob."
        if g_active and g_active_pixels >= 10 and r_active_pixels < self.min_connected_yellow_core:
            p_green = float(np.exp(min(12.0, w_snr_g)))
            return 0.01, p_green, 0.01, 0.005, 0.02

        # 5. TINY YELLOW SPECK ON RED BLOB:
        if r_active and r_active_pixels >= 10 and g_active_pixels < self.min_connected_yellow_core:
            p_red = float(np.exp(min(12.0, w_snr_r)))
            return p_red, 0.01, 0.01, 0.005, 0.02

        # 6. SINGLE-COLOR GREEN (A):
        if g_active and not r_active:
            p_green = float(np.exp(min(12.0, w_snr_g)))
            return 0.01, p_green, 0.01, 0.01, 0.02

        # 7. SINGLE-COLOR RED (T):
        if r_active and not g_active:
            p_red = float(np.exp(min(12.0, w_snr_r)))
            return p_red, 0.01, 0.01, 0.01, 0.02

        # 8. Dual active with modest overlap / ambiguous
        p_r = float(np.exp(min(6.0, w_snr_r)))
        p_g = float(np.exp(min(6.0, w_snr_g)))
        return p_r, p_g, 0.1, 0.01, max(p_r, p_g) * 0.9

    def extract_cycle_distribution_features(
        self,
        sbs_stack: np.ndarray,
        component_mask: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        component_properties: Optional[Dict[str, Any]] = None,
        dapi_stack: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts multi-tier distributional features for every cycle in sbs_stack,
        matching the exact feature structure extracted by the annotation app.
        If dapi_stack is provided, dynamically anchors nuclear and component masks
        to the local biological nucleus centroid on each cycle.
        """
        T, C, H, W = sbs_stack.shape
        records = []

        # Nominal center and size of target nucleus
        nuc_coords = np.argwhere(nuclear_mask) if nuclear_mask is not None else None
        if nuc_coords is not None and len(nuc_coords) > 0:
            cy_nom = float(np.mean(nuc_coords[:, 0]))
            cx_nom = float(np.mean(nuc_coords[:, 1]))
            r_nuc = float(np.sqrt(len(nuc_coords) / np.pi))
        else:
            comp_coords = np.argwhere(component_mask)
            if len(comp_coords) > 0:
                cy_nom = float(np.mean(comp_coords[:, 0]))
                cx_nom = float(np.mean(comp_coords[:, 1]))
                r_nuc = 14.0
            else:
                cy_nom, cx_nom, r_nuc = H / 2.0, W / 2.0, 14.0

        # Bounding box crop for fast local shift & feature extraction
        max_shift = r_nuc + 6.0
        pad = int(np.ceil(max_shift)) + 8

        combined_coords = []
        if nuc_coords is not None and len(nuc_coords) > 0:
            combined_coords.append(nuc_coords)
        comp_coords = np.argwhere(component_mask)
        if len(comp_coords) > 0:
            combined_coords.append(comp_coords)

        if combined_coords:
            all_c = np.vstack(combined_coords)
            y_min, x_min = all_c[:, 0].min(), all_c[:, 1].min()
            y_max, x_max = all_c[:, 0].max(), all_c[:, 1].max()
            y0, y1 = max(0, int(y_min - pad)), min(H, int(y_max + pad + 1))
            x0, x1 = max(0, int(x_min - pad)), min(W, int(x_max + pad + 1))
        else:
            y_c, x_c = int(round(cy_nom)), int(round(cx_nom))
            y0, y1 = max(0, y_c - pad), min(H, y_c + pad + 1)
            x0, x1 = max(0, x_c - pad), min(W, x_c + pad + 1)

        c_mask_crop = component_mask[y0:y1, x0:x1]
        n_mask_crop = nuclear_mask[y0:y1, x0:x1] if nuclear_mask is not None else None

        for t in range(T):
            r_plane_crop = sbs_stack[t, 0, y0:y1, x0:x1]
            g_plane_crop = sbs_stack[t, 1, y0:y1, x0:x1] if C >= 2 else r_plane_crop

            # Apply cycle-specific local DAPI anchoring if available
            c_mask_t = c_mask_crop
            n_mask_t = n_mask_crop
            if dapi_stack is not None and t < dapi_stack.shape[0]:
                dy, dx = self._compute_local_dapi_shift(
                    dapi_stack[t], cy_nom, cx_nom, max_shift=max_shift
                )
                if dy != 0 or dx != 0:
                    c_mask_t = ndi.shift(c_mask_crop.astype(np.float32), (dy, dx), order=0) > 0.5
                    if n_mask_crop is not None:
                        n_mask_t = ndi.shift(n_mask_crop.astype(np.float32), (dy, dx), order=0) > 0.5

            dist_feat = extract_basecaller_distribution_features(
                green_plane=g_plane_crop,
                red_plane=r_plane_crop,
                blob_mask=c_mask_t,
                nuc_mask=n_mask_t
            )
            g_d = dist_feat.get("green", {})
            r_d = dist_feat.get("red", {})
            row = {
                "cycle": t + 1,
                "blob_area": int(dist_feat.get("blob_area", np.sum(component_mask))),
                "green_median": float(g_d.get("median", 0.0)),
                "green_trimmed_mean": float(g_d.get("trimmed_mean", 0.0)),
                "green_p25": float(g_d.get("p25", 0.0)),
                "green_p75": float(g_d.get("p75", 0.0)),
                "green_p90": float(g_d.get("p90", 0.0)),
                "green_p95": float(g_d.get("p95", 0.0)),
                "green_mean": float(g_d.get("mean", 0.0)),
                "green_std": float(g_d.get("std", 0.0)),
                "green_excess_median": float(g_d.get("excess_median", 0.0)),
                "green_integrated_excess": float(g_d.get("integrated_excess", 0.0)),
                "red_median": float(r_d.get("median", 0.0)),
                "red_trimmed_mean": float(r_d.get("trimmed_mean", 0.0)),
                "red_p25": float(r_d.get("p25", 0.0)),
                "red_p75": float(r_d.get("p75", 0.0)),
                "red_p90": float(r_d.get("p90", 0.0)),
                "red_p95": float(r_d.get("p95", 0.0)),
                "red_mean": float(r_d.get("mean", 0.0)),
                "red_std": float(r_d.get("std", 0.0)),
                "red_excess_median": float(r_d.get("excess_median", 0.0)),
                "red_integrated_excess": float(r_d.get("integrated_excess", 0.0)),
                "rg_ratio_median": float(dist_feat.get("rg_ratio_median", 0.0)),
                "rg_ratio_trimmed": float(dist_feat.get("rg_ratio_trimmed", 0.0)),
            }
            if component_properties:
                for k in ["dist_to_center_norm", "ownership_ratio", "dapi_under_blob_mean"]:
                    if k in component_properties:
                        row[k] = float(component_properties[k])
            records.append(row)
        return records

    @staticmethod
    def _compute_local_dapi_shift(
        dapi_plane: np.ndarray,
        cy: float,
        cx: float,
        crop_radius: int = 30,
        max_shift: float = 20.0,
        min_size: int = 40
    ) -> Tuple[int, int]:
        """
        Dynamically finds the local nucleus centroid shift in dapi_plane relative to (cy, cx).
        Matches the dynamic Otsu centroid anchoring from the manual annotator.
        """
        H, W = dapi_plane.shape
        y_c, x_c = int(round(cy)), int(round(cx))
        y0, y1 = max(0, y_c - crop_radius), min(H, y_c + crop_radius)
        x0, x1 = max(0, x_c - crop_radius), min(W, x_c + crop_radius)
        crop = dapi_plane[y0:y1, x0:x1]
        if crop.size == 0 or crop.max() <= crop.min() + 10:
            return 0, 0
        try:
            from skimage.filters import threshold_otsu
            from skimage.morphology import remove_small_objects
            d_th = threshold_otsu(crop)
            d_mask = remove_small_objects(ndi.binary_fill_holes(crop > d_th), min_size=min_size)
            d_labeled, d_num = ndi.label(d_mask)
            if d_num > 0:
                d_dist, d_inds = ndi.distance_transform_edt(d_mask == 0, return_indices=True)
                iy = max(0, min(crop.shape[0] - 1, y_c - y0))
                ix = max(0, min(crop.shape[1] - 1, x_c - x0))
                c_y = d_inds[0, iy, ix]
                c_x = d_inds[1, iy, ix]
                c_lbl = d_labeled[c_y, c_x]
                if c_lbl > 0 and np.sum(d_labeled == c_lbl) >= min_size:
                    com_y, com_x = ndi.center_of_mass(d_labeled == c_lbl)
                    shift_dist = np.hypot(com_y - (y_c - y0), com_x - (x_c - x0))
                    if shift_dist <= max_shift:
                        return int(round(com_y - (y_c - y0))), int(round(com_x - (x_c - x0)))
        except Exception:
            pass
        return 0, 0
