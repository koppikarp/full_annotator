"""Master end-to-end pipeline coordinator for optical pooled screening."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import logging
import json
import resource
import sys
import time

from popcrop.config import PipelineConfig
from popcrop.io import (
    load_raw_image,
    load_or_generate_nuclear_mask,
    Codebook,
    export_results,
)
from popcrop.preprocessing import (
    subtract_camera_offset,
    correct_illumination_stack,
    register_cycles,
    correct_chromatic_shift,
    match_psf,
    unmix_spectral_crosstalk,
    estimate_background_and_noise,
)
from popcrop.blob_detection import (
    compute_zscore_and_energy,
    aggregate_cross_cycle_signal,
    segment_extended_blobs,
)
from popcrop.blob_classifier import (
    extract_component_features,
    BlobClassifier,
)
from popcrop.color_calling import (
    CoreAwareColorCaller,
)
from popcrop.decoding import (
    BarcodeDecoder,
)
from popcrop.visualization import (
    generate_diagnostic_overlays,
)

logger = logging.getLogger(__name__)


class OpsPipeline:
    """End-to-end production pipeline for extended fluorescent signal analysis in OPS."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        np.random.seed(self.config.random_seed)
        self.df_cycles = pd.DataFrame()
        if self.config.analysis_mode == "manual":
            from popcrop.manual_pipeline import ManualNucleusAnalyzer
            self.manual_analyzer = ManualNucleusAnalyzer(self.config)
            return
        if self.config.analysis_mode != "legacy":
            raise ValueError("analysis_mode must be 'manual' or 'legacy'")

        self.classifier = BlobClassifier(
            model_path=self.config.blob_classifier.model_path,
            prob_threshold=self.config.blob_classifier.prob_threshold,
            classifier_type=self.config.blob_classifier.classifier_type
        )
        self.color_caller = CoreAwareColorCaller(
            min_connected_yellow_core=self.config.color_calling.min_connected_yellow_core,
            min_colocalization_overlap=self.config.color_calling.min_colocalization_overlap,
            max_spatial_separation=self.config.color_calling.max_spatial_separation,
            yellow_core_min_ratio=self.config.color_calling.yellow_core_min_ratio,
            green_rim_override_protection=self.config.color_calling.green_rim_override_protection,
            model_path=self.config.color_calling.calibrated_color_model_path
        )

    def run_on_arrays(
        self,
        sbs_stack: np.ndarray,
        dapi_image: Optional[np.ndarray] = None,
        nuclear_mask: Optional[np.ndarray] = None,
        codebook: Optional[Codebook] = None,
        dapi_stack: Optional[np.ndarray] = None,
        reference_dapi: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Executes end-to-end analysis on in-memory numpy arrays.
        """
        T, C, H, W = sbs_stack.shape
        logger.info(f"Starting OpsPipeline run on stack: T={T}, C={C}, H={H}, W={W}")
        if self.config.analysis_mode == "manual":
            if nuclear_mask is None:
                nuclear_mask = load_or_generate_nuclear_mask(dapi_image=dapi_image)
            result = self.manual_analyzer.run(sbs_stack, nuclear_mask, dapi_image, dapi_stack, codebook, reference_dapi)
            self.df_cycles = self.manual_analyzer.df_cycles
            return result

        # Compute raw saturation mask and noise on raw unclipped linear data
        raw_bg_maps, raw_noise_maps, raw_sat_masks = estimate_background_and_noise(
            sbs_stack,
            bg_filter_size=self.config.blob_detection.bg_filter_size,
            bg_quantile=self.config.blob_detection.bg_quantile,
            saturation_threshold=self.config.preprocessing.saturation_threshold
        )

        # Ensure nuclear mask is available
        if nuclear_mask is None:
            if dapi_image is None:
                raise ValueError("Both nuclear_mask and dapi_image are missing. Cannot segment nuclei.")
            nuclear_mask = load_or_generate_nuclear_mask(dapi_image=dapi_image)

        # -------------------------------------------------------------
        # STEP 1: PREPROCESSING
        # -------------------------------------------------------------
        logger.info("Executing Preprocessing: Offset, Field Flattening, Registration, Unmixing...")
        stack = subtract_camera_offset(
            sbs_stack,
            camera_offset=self.config.preprocessing.camera_offset
        )

        stack, _ = correct_illumination_stack(
            stack,
            well_id=self.config.well_id,
            do_field_flattening=self.config.preprocessing.do_field_flattening,
            mode=self.config.preprocessing.field_flattening_mode,
            per_cycle=self.config.preprocessing.field_flattening_per_cycle,
            per_well=self.config.preprocessing.field_flattening_per_well
        )

        # Cycle-to-cycle registration: only run when DAPI reference image is provided
        if self.config.preprocessing.registration_method == "phase_cross_correlation" and dapi_image is not None:
            stack, reg_qc = register_cycles(
                stack,
                dapi_image=dapi_image,
                registration_channel=self.config.preprocessing.registration_channel
            )
        else:
            reg_qc = [{"cycle": t, "qc_passed": True, "shift_magnitude": 0.0} for t in range(T)]

        if self.config.preprocessing.chromatic_registration:
            stack = correct_chromatic_shift(
                stack,
                shift_red_green=self.config.preprocessing.chromatic_shift_red_green
            )
        if self.config.preprocessing.psf_matching:
            stack = match_psf(stack, target_sigma=self.config.preprocessing.psf_target_sigma)

        if self.config.preprocessing.unmixing_matrix is not None:
            stack = unmix_spectral_crosstalk(
                stack,
                unmixing_matrix=self.config.preprocessing.unmixing_matrix
            )

        preprocessing_applied = (
            self.config.preprocessing.camera_offset != 0.0
            or self.config.preprocessing.do_field_flattening
            or self.config.preprocessing.registration_method != "none"
            or self.config.preprocessing.chromatic_registration
            or self.config.preprocessing.psf_matching
            or self.config.preprocessing.unmixing_matrix is not None
        )
        if not preprocessing_applied:
            bg_maps = raw_bg_maps
        else:
            bg_maps, _, _ = estimate_background_and_noise(
                stack,
                bg_filter_size=self.config.blob_detection.bg_filter_size,
                bg_quantile=self.config.blob_detection.bg_quantile,
                saturation_threshold=self.config.preprocessing.saturation_threshold
            )
        noise_maps = raw_noise_maps
        sat_masks = raw_sat_masks

        # -------------------------------------------------------------
        # STEP 2: COLOR-INDEPENDENT BLOB DETECTION
        # -------------------------------------------------------------
        logger.info("Executing Color-Independent Blob Detection...")
        z_scores, cycle_energies = compute_zscore_and_energy(stack, bg_maps, noise_maps)
        top_k_map, persistence_map, combined_metric = aggregate_cross_cycle_signal(
            cycle_energies,
            top_k=self.config.blob_detection.top_k_cycles,
            persistence_threshold=self.config.blob_detection.persistence_threshold
        )
        blob_mask, blob_prob_map = segment_extended_blobs(
            combined_metric=combined_metric,
            nuclear_mask=nuclear_mask,
            hysteresis_high=self.config.blob_detection.hysteresis_high,
            hysteresis_low=self.config.blob_detection.hysteresis_low,
            min_blob_area=self.config.blob_detection.min_blob_area,
            max_blob_fraction=self.config.blob_detection.max_blob_fraction
        )

        # -------------------------------------------------------------
        # STEP 3: REAL-BLOB VS ARTIFACT CLASSIFICATION
        # -------------------------------------------------------------
        logger.info("Extracting object-level features and scoring P(real_blob)...")
        df_features = extract_component_features(
            sbs_stack=stack,
            background_maps=bg_maps,
            noise_maps=noise_maps,
            saturation_masks=sat_masks,
            blob_mask=blob_mask,
            nuclear_mask=nuclear_mask,
            top_k_map=top_k_map,
            persistence_map=persistence_map
        )

        if not df_features.empty:
            p_real = self.classifier.predict_proba(df_features)
            df_features["p_real_blob"] = p_real
        else:
            df_features["p_real_blob"] = []

        # -------------------------------------------------------------
        # STEP 4: CORE-AWARE COLOR CALLING & STEP 5: DECODING
        # -------------------------------------------------------------
        logger.info("Calling core-aware colors and decoding barcodes...")
        decoder = BarcodeDecoder(
            codebook=codebook,
            min_score_margin=self.config.decoding.min_score_margin,
            min_confidence=self.config.decoding.min_confidence,
            max_hamming_distance=self.config.decoding.max_hamming_distance,
        ) if codebook is not None else None

        comp_records: List[Dict[str, Any]] = []
        nuc_comp_map: Dict[int, List[Dict[str, Any]]] = {}

        # Precompute bounding boxes for fast component evaluation
        nuc_props_map = {p.label: p for p in regionprops(nuclear_mask)}
        blob_props_map = {p.label: p for p in regionprops(blob_mask)}

        for _, row in df_features.iterrows():
            comp_id = int(row["component_id"])
            nuc_id = int(row["nucleus_id"])
            p_blob = float(row["p_real_blob"])
            comp_props = row.to_dict()

            p_nuc = nuc_props_map.get(nuc_id)
            p_comp = blob_props_map.get(comp_id)

            if p_nuc is not None and p_comp is not None:
                margin = int(np.sqrt(p_nuc.area / np.pi) + 35)
                y_min = min(p_nuc.bbox[0], p_comp.bbox[0])
                y_max = max(p_nuc.bbox[2], p_comp.bbox[2])
                x_min = min(p_nuc.bbox[1], p_comp.bbox[1])
                x_max = max(p_nuc.bbox[3], p_comp.bbox[3])
                y0, y1 = max(0, y_min - margin), min(H, y_max + margin)
                x0, x1 = max(0, x_min - margin), min(W, x_max + margin)

                c_mask_crop = blob_mask[y0:y1, x0:x1] == comp_id
                n_mask_crop = nuclear_mask[y0:y1, x0:x1] == nuc_id
                sbs_crop = stack[:, :, y0:y1, x0:x1]
                dapi_crop = dapi_stack[:, y0:y1, x0:x1] if dapi_stack is not None else None
            else:
                c_mask_crop = blob_mask == comp_id
                n_mask_crop = nuclear_mask == nuc_id
                sbs_crop = stack
                dapi_crop = dapi_stack

            # Extract full cycle distribution features matching annotator
            cycle_features = self.color_caller.extract_cycle_distribution_features(
                sbs_stack=sbs_crop,
                component_mask=c_mask_crop,
                nuclear_mask=n_mask_crop,
                component_properties=comp_props,
                dapi_stack=dapi_crop
            )

            # Color calling
            cycle_probs = self.color_caller.call_blob_cycle_probabilities(
                sbs_stack=sbs_crop,
                background_maps=bg_maps[:, :, y0:y1, x0:x1] if (p_nuc and p_comp) else bg_maps,
                noise_maps=noise_maps[:, :, y0:y1, x0:x1] if (p_nuc and p_comp) else noise_maps,
                z_scores=z_scores[:, :, y0:y1, x0:x1] if (p_nuc and p_comp) else z_scores,
                component_mask=c_mask_crop,
                p_real_blob=p_blob,
                prob_threshold=self.config.blob_classifier.prob_threshold,
                nuclear_mask=n_mask_crop,
                component_properties=comp_props,
                precomputed_features=cycle_features
            )

            # Decoding
            if decoder is not None:
                decode_res = decoder.decode_component(
                    cycle_probabilities=cycle_probs,
                    p_real_blob=p_blob,
                    prob_threshold=self.config.blob_classifier.prob_threshold
                )
            else:
                decode_res = {
                    "best_barcode": "N/A",
                    "best_gene": "N/A",
                    "best_score": 0.0,
                    "second_best_score": 0.0,
                    "score_margin": 0.0,
                    "posterior_confidence": 0.0,
                    "decoding_status": "unmapped",
                    "called_sequence": "".join([max(p, key=p.get)[0].upper() for p in cycle_probs])
                }

            comp_data = {
                "plate": self.config.plate_id,
                "well": self.config.well_id,
                "tile": self.config.tile_id,
                "nucleus_id": nuc_id,
                "component_id": comp_id,
                "p_real_blob": p_blob,
                **row.to_dict(),
                **decode_res,
                "cycle_probabilities": cycle_probs,
                "cycle_distribution_features": cycle_features
            }
            comp_records.append(comp_data)
            nuc_comp_map.setdefault(nuc_id, []).append(comp_data)

        df_components = pd.DataFrame(comp_records)

        # -------------------------------------------------------------
        # STEP 6: ASSEMBLE PER-NUCLEUS SUMMARY
        # -------------------------------------------------------------
        nuc_props = regionprops(nuclear_mask)
        nuc_records: List[Dict[str, Any]] = []

        any_sat_global = np.any(sat_masks)
        sat_sum_2d = np.sum(sat_masks, axis=(0, 1)) if any_sat_global else None

        for p in nuc_props:
            n_id = p.label
            cy, cx = p.centroid
            n_area = p.area
            comps = nuc_comp_map.get(n_id, [])

            if any_sat_global:
                sl = p.slice
                n_sub = nuclear_mask[sl] == n_id
                nuc_sat_count = int(np.sum(sat_sum_2d[sl][n_sub]))
                nuc_sat_frac = float(nuc_sat_count / max(1.0, n_area * T * C))
            else:
                nuc_sat_count = 0
                nuc_sat_frac = 0.0

            if comps:
                p_max = float(max(c["p_real_blob"] for c in comps))
                if decoder is not None:
                    nuc_decode = decoder.decode_nucleus_multicomponent(comps, p_nucleus=p_max)
                else:
                    nuc_decode = comps[0]
                num_comps = len(comps)
                comp_sat = float(max(c["saturated_pixel_fraction"] for c in comps))
                sat_frac = max(nuc_sat_frac, comp_sat)
            else:
                p_max = 0.0
                num_comps = 0
                sat_frac = nuc_sat_frac
                nuc_decode = {
                    "best_barcode": "UNASSIGNED",
                    "best_gene": "NONE",
                    "best_score": -999.0,
                    "second_best_score": -999.0,
                    "score_margin": 0.0,
                    "posterior_confidence": 0.0,
                    "decoding_status": "no_blob",
                    "called_sequence": "NNNNNNNNNNNN"
                }

            qc_flags = []
            if sat_frac > 0.01 or nuc_sat_count > 0:
                qc_flags.append("SATURATED")
            if any(not r.get("qc_passed", True) for r in reg_qc):
                qc_flags.append("REGISTRATION_WARN")
            if nuc_decode.get("decoding_status") == "mixed":
                qc_flags.append("MIXED_COMPONENTS")
            qc_str = ";".join(qc_flags) if qc_flags else "PASS"

            nuc_rec = {
                "plate": self.config.plate_id,
                "well": self.config.well_id,
                "tile": self.config.tile_id,
                "nucleus_id": n_id,
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "nucleus_area": int(n_area),
                "p_real_blob": p_max,
                "num_components": num_comps,
                "saturated_pixel_fraction": sat_frac,
                "best_barcode": nuc_decode.get("best_barcode", "UNASSIGNED"),
                "best_gene": nuc_decode.get("best_gene", "NONE"),
                "best_score": nuc_decode.get("best_score", -999.0),
                "second_best_score": nuc_decode.get("second_best_score", -999.0),
                "score_margin": nuc_decode.get("score_margin", 0.0),
                "posterior_confidence": nuc_decode.get("posterior_confidence", 0.0),
                "decoding_status": nuc_decode.get("decoding_status", "no_blob"),
                "called_sequence": nuc_decode.get("called_sequence", "NNNNNNNNNNNN"),
                "qc_flags": qc_str
            }
            nuc_records.append(nuc_rec)

        df_nuclei = pd.DataFrame(nuc_records)
        masks_dict = {
            "nuclear_mask": nuclear_mask,
            "blob_mask": blob_mask,
            "blob_prob_map": blob_prob_map,
        }

        logger.info(f"OpsPipeline run completed successfully. Processed {len(df_nuclei)} nuclei.")
        return df_nuclei, df_components, masks_dict

    def run_on_files(
        self,
        image_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        tiff_path: Optional[str | Path] = None,
        mask_path: Optional[str | Path] = None,
        codebook_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        position_idx: int = 0
    ) -> Dict[str, Any]:
        """Run file analysis in a new, uniquely named child of output_dir."""
        started = time.monotonic()
        input_target = image_path or tiff_path
        if input_target is None:
            raise ValueError("Must provide either image_path or tiff_path to run_on_files.")

        from popcrop.io.run_identity import allocate_run, add_run_columns, write_run_manifest
        out_dir, prefix, run_metadata = allocate_run(
            output_dir or self.config.output.output_dir, input_target, self.config, position_idx,
        )
        manifest_path = out_dir / f"{prefix}_run_manifest.json"
        write_run_manifest(manifest_path, run_metadata, "started")
        logger.info("Run %s; results directory: %s", run_metadata["run_id"], out_dir)

        # Universal image loader (handles both ND2 and TIFF automatically)
        sbs_stack, dapi_image, meta = load_raw_image(
            image_path_or_paths=input_target,
            has_dapi_cycle=self.config.has_dapi_cycle,
            position_idx=position_idx
        )

        nuclear_mask = load_or_generate_nuclear_mask(
            mask_path=mask_path,
            dapi_image=dapi_image
        )

        cb_path = codebook_path or self.config.decoding.codebook_path
        if cb_path:
            self.config.decoding.codebook_path = str(Path(cb_path).resolve())
        if cb_path and not Path(cb_path).exists():
            raise FileNotFoundError(f"Configured codebook does not exist: {cb_path}")
        codebook = (
            Codebook.from_file(cb_path, barcode_col=self.config.decoding.barcode_column,
                               match_reverse_complement=self.config.decoding.match_reverse_complement)
            if cb_path and Path(cb_path).exists()
            else None
        )

        df_nuclei, df_components, masks_dict = self.run_on_arrays(
            sbs_stack=sbs_stack,
            dapi_image=dapi_image,
            nuclear_mask=nuclear_mask,
            codebook=codebook,
            dapi_stack=meta.get("dapi_stack"),
            reference_dapi=meta.get("reference_dapi"),
        )

        for frame in (df_nuclei, df_components, self.df_cycles):
            add_run_columns(frame, run_metadata)

        saved_files = export_results(
            output_dir=out_dir,
            df_nuclei=df_nuclei,
            df_components=df_components,
            masks_dict={k: v for k, v in masks_dict.items()
                        if (self.config.output.save_probability_maps if 'prob' in k else self.config.output.save_masks)},
            export_format=self.config.output.export_format,
            prefix=prefix
        )
        if not self.df_cycles.empty:
            cycles_path = out_dir / f"{prefix}_cycles.parquet"
            self.df_cycles.to_parquet(cycles_path, index=False, compression="zstd")
            saved_files["cycles"] = str(cycles_path)
        from popcrop.reporting import write_run_report
        saved_files.update(write_run_report(out_dir, df_nuclei, df_components, self.config, meta,
                                           prefix=prefix, run_metadata=run_metadata))

        if self.config.output.save_diagnostics and not df_nuclei.empty:
            diag_path = out_dir / f"{prefix}_diagnostics.png"
            generate_diagnostic_overlays(
                dapi_image=dapi_image,
                sbs_stack=sbs_stack,
                nuclear_mask=nuclear_mask,
                blob_mask=masks_dict["blob_mask"],
                df_nuclei=df_nuclei,
                output_path=diag_path,
                max_cells=self.config.output.diagnostic_max_cells
            )
            saved_files["diagnostics_png"] = str(diag_path)

        # Finalize runtime metadata before exposing a completed run to collectors.
        summary_path = Path(saved_files["summary_json"])
        summary = json.loads(summary_path.read_text())
        summary["runtime_seconds"] = round(time.monotonic() - started, 2)
        summary["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                        (1024**2 if sys.platform == "darwin" else 1024), 1)
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)
        saved_files["manifest"] = str(manifest_path)
        write_run_manifest(manifest_path, run_metadata, "complete", saved_files)
        return {
            "output_dir": str(out_dir),
            "run_metadata": run_metadata,
            "df_nuclei": df_nuclei,
            "df_components": df_components,
            "masks_dict": masks_dict,
            "saved_files": saved_files
        }
