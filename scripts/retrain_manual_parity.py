#!/usr/bin/env python3
"""Regenerate raw shared features without modifying annotations, then calibrate by nucleus.

Only annotation labels and coordinates are reused. Historical correctness judgements
are converted to presence using the historical detector bits BEFORE feature regeneration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "popcrop-matplotlib"))
warnings.filterwarnings("ignore", category=FutureWarning, message=".*deprecated.*")

import numpy as np
import pandas as pd
import tifffile
import joblib
from threadpoolctl import threadpool_limits
from sklearn.metrics import accuracy_score, classification_report
from popcrop.annotation.analysis import analyze_nucleus_cycle, ANALYSIS_VERSION, DEFAULT_DETECTOR_PARAMS
from popcrop.annotation.app import extract_channel_frame, describe_tiff_array
from popcrop.calibration.ground_truth_adapter import derive_blob_presence_labels, annotation_groups
from popcrop.models.trainer import BaseModelTrainer, BlobModelTrainer


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coordinates(profile_path, image_sha):
    database = json.loads(Path(profile_path).read_text())
    profile = database.get("images", {}).get(image_sha)
    if profile is None or not profile.get("cells"):
        raise ValueError("No saved coordinate profile matches this TIFF's SHA256; provide its verified profile")
    return pd.DataFrame(profile["cells"])


def regenerate(base_source, blob_source, image, coordinates, diameter):
    # Presence is recovered from immutable source before replacing any detector feature.
    blob_labels = derive_blob_presence_labels(blob_source)
    centers = coordinates[["i", "j"]].to_numpy(float)
    coord_ids = coordinates["cell"].to_numpy(float)
    reference_dapi = extract_channel_frame(image, cycle=0, channel_idx=0)
    n_cycles = describe_tiff_array(image)["sequencing_cycles"]
    bases, blobs, auxiliary, provenance = [], [], [], None
    scalar_base_features, scalar_blob_features = None, None
    for label_kind, source, output in (("explicit_base", base_source, bases), ("correctness", blob_labels, blobs)):
        for index, row in source.iterrows():
            cycle = int(row["cycle"])
            if not 1 <= cycle <= n_cycles:
                raise ValueError(f"Invalid sequencing cycle {cycle} in annotation row {index}")
            # Source saved coordinates are authoritative, even if a later profile moved the nucleus.
            center = np.array([float(row["y"]), float(row["x"])])
            neighbors = centers[coord_ids != float(row["cell"])] if pd.notna(row["cell"]) else centers[np.linalg.norm(centers-center, axis=1) > 1.0]
            result = analyze_nucleus_cycle(
                extract_channel_frame(image, cycle, 1), extract_channel_frame(image, cycle, 2),
                extract_channel_frame(image, cycle, 0), center,
                nuclear_diameter=diameter, reference_dapi=reference_dapi,
                neighbor_centers_yx=neighbors,
            )
            if provenance is None:
                provenance = {"analysis_version": ANALYSIS_VERSION, "input_domain": "raw",
                    "nuclear_diameter": diameter, "radius_scale": 1.6,
                    "detector_params": DEFAULT_DETECTOR_PARAMS}
                scalar_base_features = [k for k,v in result["base_row"].items() if np.isscalar(v)]
                scalar_blob_features = [k for k,v in result["blob_features"].items() if np.isscalar(v) and k != "p_real_blob"]
            identity = {name: row[name] for name in ("cell", "cycle", "tile", "well", "x", "y", "tiff_filename", "timestamp") if name in row}
            identity.update(source_row=int(index), label_source=label_kind,
                analysis_provenance_json=json.dumps(result["provenance"], sort_keys=True))
            base_row = dict(identity, **result["base_row"])
            for key, value in list(base_row.items()):
                if isinstance(value, (list, dict)):
                    base_row[key+"_json"] = json.dumps(value)
                    del base_row[key]
            if label_kind == "explicit_base":
                base_row["annotated_base"] = row["annotated_base"]
                output.append(base_row)
            else:
                blob_row = dict(identity, **result["blob_features"])
                for channel in ("green", "red"):
                    blob_row[f"original_{channel}_detected"] = bool(row[f"{channel}_detected"])
                    blob_row[f"original_{channel}_called_correctly"] = bool(row[f"{channel}_called_correctly"])
                    blob_row[f"{channel}_signal_present"] = bool(row[f"{channel}_signal_present"])
                blob_row.update(any_signal_present=bool(row["any_signal_present"]),
                    derived_base=row["derived_base"], original_is_real_blob=bool(row["is_real_blob"]),
                    original_call_status=row["call_status"], label_semantics=row["label_semantics"])
                output.append(blob_row)
                base_row["annotated_base"] = row["derived_base"]
                auxiliary.append(base_row)
            if (index+1) % 200 == 0 or index == len(source)-1:
                print(f"Regenerated {label_kind}: {index+1}/{len(source)}", flush=True)
    return pd.DataFrame(bases), pd.DataFrame(blobs), pd.DataFrame(auxiliary), scalar_base_features, scalar_blob_features, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "wellb2_002_002_RGB.tif")
    parser.add_argument("--base-input", type=Path, default=PROJECT_ROOT / "annotated_base_ground_truth.csv")
    parser.add_argument("--blob-input", type=Path, default=PROJECT_ROOT / "annotated_blob_ground_truth.csv")
    parser.add_argument("--coordinate-profile", type=Path, default=PROJECT_ROOT / "saved_coordinate_profiles.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/manual_parity")
    parser.add_argument("--nuclear-diameter", type=float, default=55.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"image_sha256": file_sha256(args.image), "base_source_sha256": file_sha256(args.base_input),
        "blob_source_sha256": file_sha256(args.blob_input), "coordinate_profile_sha256": file_sha256(args.coordinate_profile)}
    coordinates = load_coordinates(args.coordinate_profile, hashes["image_sha256"])
    base_source, blob_source = pd.read_csv(args.base_input), pd.read_csv(args.blob_input)
    filenames = set(base_source.tiff_filename) | set(blob_source.tiff_filename)
    if filenames != {args.image.name}:
        raise ValueError(f"Annotation image identities {filenames} do not match supplied image")
    # The compressed source is decoded into a temporary disk mapping, never a giant float array.
    with tempfile.TemporaryDirectory(prefix="popcrop-training-") as temporary, threadpool_limits(limits=args.threads):
        try:
            image = tifffile.memmap(args.image, mode="r")
        except (ValueError, TypeError):
            image = tifffile.imread(args.image, out=str(Path(temporary) / "image.memmap"))
        print(f"Image {image.shape} {image.dtype}, {len(coordinates)} verified neighbor nuclei", flush=True)
        base, blob, recovered, base_features, blob_features, provenance = regenerate(
            base_source, blob_source, image, coordinates, args.nuclear_diameter)
        del image
        provenance.update(hashes)
        provenance.update(label_semantics="presence_from_original_detection_and_correctness_v1",
            coordinate_source=str(args.coordinate_profile), image_path=str(args.image),
            base_source_path=str(args.base_input), blob_source_path=str(args.blob_input),
            base_training_labels="explicit_annotations_only", excluded_base_labels=["N"])
        base.to_parquet(args.output_dir / "regenerated_base_features.parquet", index=False)
        blob.to_parquet(args.output_dir / "regenerated_blob_features.parquet", index=False)
        recovered.to_parquet(args.output_dir / "recovered_base_features.parquet", index=False)
        base_trainer = BaseModelTrainer(features=base_features, strict_features=True)
        base_metrics = base_trainer.train(base, args.output_dir, provenance=provenance)
        joblib.dump(base_trainer.holdout_model, args.output_dir / "evaluation_base_caller.joblib")
        evaluation_metadata = dict(base_trainer.metadata,
            artifact_role="heldout_evaluation_only", final_refit_samples=base_metrics["train_samples"],
            train_groups=sorted(base_trainer.holdout_train_groups),
            test_groups=sorted(base_trainer.holdout_test_groups))
        (args.output_dir / "evaluation_base_caller_metadata.json").write_text(json.dumps(evaluation_metadata, indent=2))
        blob_trainer = BlobModelTrainer(features=blob_features, target_col="any_signal_present", strict_features=True)
        blob_metrics = blob_trainer.train(blob, args.output_dir, provenance=provenance)
        base_trainer.holdout_predictions.to_csv(args.output_dir / "base_holdout_predictions.csv", index=False)
        blob_trainer.holdout_predictions.to_csv(args.output_dir / "blob_holdout_predictions.csv", index=False)
        # These are manually corrected labels on nuclei unseen by the explicit-base holdout model.
        unseen = ~np.isin(annotation_groups(recovered), list(base_trainer.holdout_train_groups))
        external = recovered.loc[unseen]
        auxiliary_metrics = {"samples": len(external)}
        if len(external):
            prediction = base_trainer.holdout_model.predict(external[base_features].to_numpy(float))
            auxiliary_metrics.update(accuracy=float(accuracy_score(external.annotated_base, prediction)),
                classification_report=classification_report(external.annotated_base, prediction, output_dict=True, zero_division=0))
        corrected = derive_blob_presence_labels(blob_source)
        historical_area = blob_source.loc[blob_source.fraction_nucleus_occupied > 0, "area"] / blob_source.loc[blob_source.fraction_nucleus_occupied > 0, "fraction_nucleus_occupied"]
        report = {"provenance": provenance, "source_base_rows": len(base_source),
            "discarded_N_rows": int(base_source.annotated_base.eq("N").sum()),
            "source_blob_rows": len(blob_source), "corrected_positive_blob_rows": int(corrected.any_signal_present.sum()),
            "changed_historical_blob_labels": int((corrected.any_signal_present != corrected.is_real_blob).sum()),
            "historical_inferred_roi_area_median": float(historical_area.median()),
            "historical_inferred_roi_diameter_median": float(2*np.sqrt(historical_area.median()/np.pi)),
            "historical_blob_area_gt300": int(blob_source.area.gt(300).sum()),
            "historical_blob_area_gt600": int(blob_source.area.gt(600).sum()),
            "historical_blob_area_max": int(blob_source.area.max()),
            "base_grouped_holdout": base_metrics, "blob_grouped_holdout": blob_metrics,
            "recovered_base_external_evaluation": auxiliary_metrics,
            "elapsed_seconds": time.perf_counter()-started}
        (args.output_dir / "training_audit.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    # Explicitly verify that source annotation records remained byte-identical.
    assert file_sha256(args.base_input) == hashes["base_source_sha256"]
    assert file_sha256(args.blob_input) == hashes["blob_source_sha256"]


if __name__ == "__main__":
    main()
