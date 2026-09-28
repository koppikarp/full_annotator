"""Bounded-memory, one-read-per-nucleus analysis on shared annotator features."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from popcrop.annotation.analysis import analyze_nucleus_cycle, ANALYSIS_VERSION, DEFAULT_DETECTOR_PARAMS
from popcrop.color_calling.caller import ModelBasedBaseCaller
from popcrop.models.trainer import load_trained_model
from popcrop.decoding import BarcodeDecoder, call_bases_from_probabilities
from popcrop.io.codebook import COLOR_TO_BASE, reverse_complement

logger = logging.getLogger(__name__)


def _json_compatible(value):
    """Represent missing measurements as JSON null, without inventing zeros."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


class ManualNucleusAnalyzer:
    def __init__(self, config):
        self.config = config
        self.settings = config.manual_analysis
        p = config.preprocessing
        if (p.camera_offset != 0 or p.do_field_flattening or p.registration_method != "none"
                or p.chromatic_registration or p.psf_matching or p.unmixing_matrix is not None):
            raise ValueError("Manual analysis requires raw intensities; disable preprocessing corrections.")
        if not Path(self.settings.base_model_path).is_file():
            raise FileNotFoundError(
                f"Missing annotator-compatible base model: {self.settings.base_model_path}. "
                "Run scripts/retrain_manual_parity.py first."
            )
        self.base_caller = ModelBasedBaseCaller(self.settings.base_model_path, strict_features=True)
        self._validate_model_provenance(self.base_caller.metadata, "base")
        self.blob_model, self.blob_metadata = None, {}
        if self.settings.blob_model_path:
            if not Path(self.settings.blob_model_path).is_file():
                raise FileNotFoundError(self.settings.blob_model_path)
            self.blob_model, self.blob_metadata = load_trained_model(self.settings.blob_model_path)
            self._validate_model_provenance(self.blob_metadata, "blob")
        self.df_cycles = pd.DataFrame()

    def _validate_model_provenance(self, metadata, name):
        provenance = metadata.get("provenance", {})
        expected_type = "base_caller" if name == "base" else "blob_classifier"
        if metadata.get("model_type") != expected_type:
            raise ValueError(f"Expected a {expected_type} model")
        classes = set(map(str, self.base_caller.classes if name == "base" else self.blob_model.classes_))
        if classes != (set("ACGT") if name == "base" else {"0", "1"}):
            raise ValueError(f"Unexpected {name} model classes: {classes}")
        if name == "blob" and metadata.get("target_col") != "any_signal_present":
            raise ValueError("Blob model must learn corrected signal presence")
        if provenance.get("analysis_version") != ANALYSIS_VERSION or provenance.get("input_domain") != "raw":
            raise ValueError(f"{name} model does not declare compatible raw shared-analysis features; retrain it")
        expected = {**DEFAULT_DETECTOR_PARAMS, **self.settings.detector_params}
        if (provenance.get("nuclear_diameter") != self.settings.nuclear_diameter
                or provenance.get("radius_scale") != self.settings.radius_scale
                or provenance.get("detector_params") != expected):
            raise ValueError(f"{name} model was trained with different ROI/detector settings. "
                             "Retrain with the requested settings to avoid feature distribution mismatch.")

    def run(self, stack, nuclear_mask, dapi_image=None, dapi_stack=None, codebook=None, reference_dapi=None):
        started = time.monotonic()
        if stack.ndim != 4 or stack.shape[1] != 2:
            raise ValueError("Expected raw SBS stack shaped (cycles, 2, height, width), ordered red, green")
        T, _, H, W = stack.shape
        if T < 1:
            raise ValueError("At least one SBS cycle is required")
        if nuclear_mask.shape != (H, W) or not np.issubdtype(nuclear_mask.dtype, np.integer):
            raise ValueError("Nuclear mask must be integer labels matching image dimensions")
        if (nuclear_mask < 0).any():
            raise ValueError("Nuclear labels must be nonnegative")
        if dapi_stack is not None and dapi_stack.shape != (T, H, W):
            raise ValueError("DAPI stack must have exactly one image per SBS cycle")
        if codebook is not None and codebook.barcode_length != T:
            raise ValueError(f"Image has {T} SBS cycles, library has {codebook.barcode_length}-base barcodes")
        settings = self.settings
        if settings.batch_size < 1 or not 0 <= settings.signal_probability_threshold <= 1:
            raise ValueError("Invalid batch size or signal probability threshold")
        props = regionprops(nuclear_mask)
        centers = np.array([p.centroid for p in props], dtype=float).reshape(-1, 2)
        decoder = BarcodeDecoder(codebook, self.config.decoding.min_score_margin,
                                 self.config.decoding.min_confidence,
                                 max_hamming_distance=self.config.decoding.max_hamming_distance) if codebook else None
        blob_mask = np.zeros((H, W), dtype=np.int32)
        prob_map = np.zeros((H, W), dtype=np.float32)
        nuclei, components, cycles = [], [], []
        for start in range(0, len(props), settings.batch_size):
            pending = []
            for offset, prop in enumerate(props[start:start+settings.batch_size]):
                index = start + offset
                # Same ownership context as training/UI, including sparse fields.
                neighbors = np.concatenate((centers[:index], centers[index+1:]))
                cell_rows = []
                footprint = []
                for t in range(T):
                    analysis = analyze_nucleus_cycle(
                        stack[t, 1], stack[t, 0], dapi_stack[t] if dapi_stack is not None else dapi_image,
                        centers[index], settings.nuclear_diameter,
                        reference_dapi=reference_dapi if reference_dapi is not None else dapi_image,
                        radius_scale=settings.radius_scale, detector_params=settings.detector_params,
                        neighbor_centers_yx=neighbors,
                    )
                    box = analysis["crop_box"]
                    mask = analysis["composite_mask"]
                    detected = analysis["detection_summary"]["any_detected"]
                    pixels = np.flatnonzero(mask)
                    # Store exact local masks as pixel indices; all crops remain views.
                    row = {
                        "nucleus_id": int(prop.label), "cycle": t+1,
                        **analysis["base_row"], **analysis["detection_summary"],
                        "blob_features": analysis["blob_features"],
                        "green_detection_stats": analysis["green_stats"],
                        "red_detection_stats": analysis["red_stats"],
                        "green_mask_pixels": np.flatnonzero(analysis["green_mask"]).tolist(),
                        "red_mask_pixels": np.flatnonzero(analysis["red_mask"]).tolist(),
                        "analysis_provenance": analysis["provenance"],
                        "saturated_pixel_fraction": float(np.mean(
                            (analysis["g_crop"][analysis["nuc_mask"]] >= self.config.preprocessing.saturation_threshold)
                            | (analysis["r_crop"][analysis["nuc_mask"]] >= self.config.preprocessing.saturation_threshold))),
                    }
                    if detected:
                        yy, xx = np.unravel_index(pixels, mask.shape)
                        footprint.append((yy+box["y_min"], xx+box["x_min"]))
                    cell_rows.append(row)
                pending.append((prop, cell_rows, footprint))
            flat_rows = [r for _, rows, _ in pending for r in rows]
            probabilities = self.base_caller.predict_cycle_probabilities(pd.DataFrame(flat_rows))
            blob_probs = [None] * len(flat_rows)
            if self.blob_model is not None:
                names = self.blob_metadata["feature_names"]
                blob_df = pd.DataFrame([r["blob_features"] for r in flat_rows])
                missing = set(names) - set(blob_df)
                if missing:
                    raise ValueError(f"Blob model features absent from shared analysis: {sorted(missing)}")
                positive = list(self.blob_model.classes_).index(1)
                blob_probs = self.blob_model.predict_proba(blob_df[names].to_numpy(dtype=float))[:, positive]
            cursor = 0
            for prop, rows, footprint in pending:
                probs = probabilities[cursor:cursor+T]
                p_active = [1.0 - p["dark"] for p in probs]
                called = call_bases_from_probabilities(probs)
                raw_bases = list(called)
                signal_cycles = sum(p >= settings.signal_probability_threshold for p in p_active)
                has_signal = signal_cycles > 0
                detected_cycles = sum(r["any_detected"] for r in rows)
                for t, row in enumerate(rows):
                    row["called_base"] = raw_bases[t]
                    row["cycle_probabilities"] = probs[t]
                    row["signal_probability"] = p_active[t]
                    row["blob_presence_probability"] = None if blob_probs[cursor+t] is None else float(blob_probs[cursor+t])
                cursor += T
                if decoder:
                    # Decode all observations, then exclude no-signal cells from
                    # assignment while preserving their raw calls and diagnostics.
                    decoded = decoder.decode_component(probs)
                    if not has_signal:
                        decoded["decoding_status"] = "no_signal"
                else:
                    decoded = {"called_sequence": called, "called_library_barcode": reverse_complement(called) if self.config.decoding.match_reverse_complement else called,
                               "best_barcode": "UNASSIGNED", "best_gene": "NONE",
                               "decoding_status": "unmapped" if has_signal else "no_signal",
                               "posterior_confidence": 0.0, "score_margin": 0.0}
                matched = decoded["decoding_status"] == "matched"
                barcode_matched = decoded["decoding_status"] in ("matched", "ambiguous_gene")
                flags = []
                saturation = max(r["saturated_pixel_fraction"] for r in rows)
                if saturation > 0:
                    flags.append("SATURATED")
                if has_signal and not detected_cycles:
                    flags.append("SIGNAL_WITHOUT_DETECTOR_MASK")
                if detected_cycles and not has_signal:
                    flags.append("DETECTOR_WITHOUT_CLASSIFIED_SIGNAL")
                if any(b == "G" and p >= settings.signal_probability_threshold for b,p in zip(raw_bases,p_active)):
                    flags.append("SIGNAL_WITH_DARK_HARDCALL")
                shift = max(np.linalg.norm(np.asarray(r["analysis_provenance"]["center_anchored_yx"])-np.asarray(prop.centroid)) for r in rows)
                if shift > settings.nuclear_diameter/4:
                    flags.append("LARGE_DAPI_RECENTERING")
                record = {
                    "plate": self.config.plate_id, "well": self.config.well_id, "tile": self.config.tile_id,
                    "nucleus_id": int(prop.label), "centroid_y": float(prop.centroid[0]),
                    "centroid_x": float(prop.centroid[1]), "nucleus_area": int(prop.area),
                    "analysis_diameter": settings.nuclear_diameter, "analysis_version": ANALYSIS_VERSION,
                    "has_detectable_spot": bool(detected_cycles), "has_signal": has_signal,
                    "detected_cycles": detected_cycles, "signal_cycles": signal_cycles,
                    "max_cycle_signal_probability": max(p_active),
                    # Compatibility column, explicitly a max over cycle scores.
                    "p_real_blob": max(p_active), "num_components": int(bool(detected_cycles)),
                    "saturated_pixel_fraction": saturation, "max_dapi_center_shift": float(shift),
                    **decoded, "assigned_barcode": decoded["best_barcode"] if barcode_matched else "UNASSIGNED",
                    "assigned_gene": decoded["best_gene"] if matched else "NONE",
                    "qc_flags": ";".join(flags) if flags else "PASS",
                }
                nuclei.append(record)
                if detected_cycles or has_signal:
                    components.append({**record, "component_id": int(prop.label), "read_unit": "nucleus"})
                for yy, xx in footprint:
                    empty = blob_mask[yy, xx] == 0
                    blob_mask[yy[empty], xx[empty]] = prop.label
                    prob_map[yy, xx] = np.maximum(prob_map[yy, xx], max(p_active))
                for row in rows:
                    # JSON avoids schema instability in nested feature dictionaries,
                    # retaining every statistic, histogram, sample and exact mask.
                    for key, value in list(row.items()):
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(_json_compatible(value), separators=(",", ":"), allow_nan=False)
                    cycles.append(row)
            logger.info("Analyzed %d/%d nuclei (%d cycles), elapsed %.1fs, peak RSS %.0f MiB",
                        min(start+settings.batch_size, len(props)), len(props), len(cycles),
                        time.monotonic()-started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                        (1024**2 if sys.platform == "darwin" else 1024))
        self.df_cycles = pd.DataFrame(cycles)
        nuc_table = pd.DataFrame(nuclei) if nuclei else pd.DataFrame(columns=[
            "nucleus_id", "decoding_status", "has_signal", "has_detectable_spot", "best_gene"])
        return nuc_table, pd.DataFrame(components), {
            "nuclear_mask": nuclear_mask, "blob_mask": blob_mask, "blob_prob_map": prob_map,
        }
