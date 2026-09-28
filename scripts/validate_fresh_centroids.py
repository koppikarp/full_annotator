#!/usr/bin/env python3
"""Evaluate held-out annotations at fresh automated DAPI centers, without model fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import warnings
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "popcrop-matplotlib"))
warnings.filterwarnings("ignore", category=FutureWarning, message=".*deprecated.*")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from popcrop.annotation.analysis import analyze_nucleus_cycle
from popcrop.calibration.ground_truth_adapter import annotation_groups
from popcrop.io.image_loader import load_raw_image
from popcrop.io.masks import load_or_generate_nuclear_mask
from popcrop.models.trainer import load_trained_model


def metrics(frame, prediction_column):
    if len(frame) == 0:
        return {"rows": 0}
    return {"rows": len(frame), "nuclei": frame.group.nunique(),
        "accuracy": float(accuracy_score(frame.annotated_base, frame[prediction_column])),
        "balanced_accuracy": float(balanced_accuracy_score(frame.annotated_base, frame[prediction_column])),
        "class_order": ["A", "C", "G", "T"],
        "confusion_matrix": confusion_matrix(frame.annotated_base, frame[prediction_column], labels=["A", "C", "G", "T"]).tolist(),
        "classification_report": classification_report(frame.annotated_base, frame[prediction_column], output_dict=True, zero_division=0)}


def load_validation_inputs(image_path, mask_path=None):
    """Use production image views and either regenerate or load nuclear labels."""
    sbs_stack, dapi_image, image_metadata = load_raw_image(image_path)
    if sbs_stack.shape[1] != 2:
        raise ValueError("Fresh-centroid validation requires red and green channels")
    if mask_path is not None and Path(mask_path).suffix.lower() == ".npz":
        with np.load(mask_path, allow_pickle=False) as arrays:
            if "nuclear_mask" not in arrays:
                raise ValueError("Production mask archive must contain nuclear_mask")
            mask = arrays["nuclear_mask"]
    else:
        mask = load_or_generate_nuclear_mask(mask_path=mask_path, dapi_image=dapi_image)
    if mask.ndim != 2 or mask.shape != sbs_stack.shape[-2:]:
        raise ValueError("Nuclear mask must match the image's two-dimensional spatial shape")
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("Nuclear mask must contain nonnegative integer labels")
    return sbs_stack, dapi_image, image_metadata, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=None,
        help="Optional nuclear labels (.npy, .tif/.tiff, or production .npz); otherwise segment fresh DAPI")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "wellb2_002_002_RGB.tif")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/manual_parity")
    parser.add_argument("--max-distance", type=float, default=27.5)
    args = parser.parse_args()
    model, metadata = load_trained_model(args.model_dir / "evaluation_base_caller.joblib")
    feature_names = metadata["feature_names"]
    base = pd.read_parquet(args.model_dir / "regenerated_base_features.parquet")
    base = base[base.annotated_base.isin(["A", "C", "G", "T"])].copy()
    base["group"] = annotation_groups(base)
    sbs_stack, dapi_image, image_metadata, mask = load_validation_inputs(args.image, args.mask)
    props = pd.DataFrame(regionprops_table(mask, properties=["label", "centroid"]))
    centers = props[["centroid-0", "centroid-1"]].to_numpy()
    labels = props.label.to_numpy()
    if not len(centers):
        raise ValueError("No nuclei found in the validation mask")
    distances, nearest = cKDTree(centers).query(base[["y", "x"]].to_numpy())
    base["centroid_distance_px"], base["fresh_index"] = distances, nearest
    base["fresh_label"] = labels[nearest]
    base["matched"] = distances <= args.max_distance
    train_labels = set(base.loc[base.group.isin(metadata["train_groups"]) & base.matched, "fresh_label"])
    duplicate_labels = set(base.loc[base.matched].groupby("fresh_label").group.nunique().loc[lambda s:s>1].index)
    test = base[base.group.isin(metadata["test_groups"])].copy()
    test["fresh_aliases_training_nucleus"] = test.fresh_label.isin(train_labels)
    test["multiple_manual_nuclei_to_one_fresh_nucleus"] = test.fresh_label.isin(duplicate_labels)
    test["original_center_prediction"] = model.predict(test[feature_names].to_numpy(float))
    with threadpool_limits(limits=2):
        reference = image_metadata.get("reference_dapi")
        dapi_stack = image_metadata.get("dapi_stack")
        rows = []
        for row in test[test.matched].itertuples():
            cycle, index = int(row.cycle), int(row.fresh_index)
            if not 1 <= cycle <= len(sbs_stack):
                raise ValueError(f"Annotation cycle {cycle} is outside the supplied image")
            cycle_dapi = dapi_stack[cycle-1] if dapi_stack is not None else dapi_image
            result = analyze_nucleus_cycle(sbs_stack[cycle-1, 1],
                sbs_stack[cycle-1, 0], cycle_dapi,
                centers[index], nuclear_diameter=metadata["provenance"]["nuclear_diameter"],
                reference_dapi=reference, neighbor_centers_yx=np.delete(centers,index,axis=0))
            rows.append([result["base_row"][feature] for feature in feature_names])
        probability = model.predict_proba(np.asarray(rows,float))
        test.loc[test.matched,"fresh_center_prediction"] = model.classes_[probability.argmax(axis=1)]
        for index, label in enumerate(model.classes_):
            test.loc[test.matched,f"fresh_p_{label}"] = probability[:,index]
        del sbs_stack, dapi_image, image_metadata, dapi_stack, reference
    matched = test[test.matched]
    independent = matched[~matched.fresh_aliases_training_nucleus]
    unambiguous = independent[~independent.multiple_manual_nuclei_to_one_fresh_nucleus]
    report = {"model": "evaluation_base_caller.joblib", "total_fresh_nuclei": len(centers),
        "heldout_rows": len(test), "heldout_nuclei": test.group.nunique(),
        "match_distance_limit_px": args.max_distance,
        "matched_rows": len(matched), "unmatched_rows": int((~test.matched).sum()),
        "matched_nuclei": matched.group.nunique(),
        "training_alias_rows_excluded_from_independent_metrics": int(matched.fresh_aliases_training_nucleus.sum()),
        "duplicate_fresh_nuclei_across_manual_groups": len(duplicate_labels),
        "heldout_rows_mapping_to_duplicate_fresh_nuclei": int(matched.multiple_manual_nuclei_to_one_fresh_nucleus.sum()),
        "centroid_distance_median_px": float(matched.centroid_distance_px.median()),
        "centroid_distance_p95_px": float(matched.centroid_distance_px.quantile(.95)),
        "original_centers_all_heldout": metrics(test,"original_center_prediction"),
        "original_centers_same_independent_rows": metrics(independent,"original_center_prediction"),
        "fresh_centers_independent": metrics(independent,"fresh_center_prediction"),
        "fresh_centers_independent_distance_lt10": metrics(independent[independent.centroid_distance_px<10],"fresh_center_prediction"),
        "fresh_centers_independent_unambiguous": metrics(unambiguous,"fresh_center_prediction"),
    }
    test[[c for c in test.columns if c in ["cell","cycle","x","y","group","annotated_base","fresh_label",
        "centroid_distance_px","matched","fresh_aliases_training_nucleus","multiple_manual_nuclei_to_one_fresh_nucleus",
        "original_center_prediction","fresh_center_prediction","fresh_p_A","fresh_p_C","fresh_p_G","fresh_p_T"]]].to_csv(
            args.model_dir / "fresh_centroid_holdout_predictions.csv",index=False)
    (args.model_dir / "fresh_centroid_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
