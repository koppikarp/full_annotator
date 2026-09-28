"""Calibrated annotation models with nucleus-grouped holdout and calibration."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, classification_report, confusion_matrix,
    roc_auc_score, log_loss)
from popcrop.calibration.ground_truth_adapter import derive_blob_presence_labels, annotation_groups
logger = logging.getLogger(__name__)

BLOB_DEFAULT_FEATURES = [
    "median_contrast",
    "peak_contrast",
    "upper_quantile_contrast",
    "integrated_excess_intensity",
    "spatial_smoothness",
    "solidity",
    "eccentricity",
    "perimeter_to_area_ratio",
    "persistence_cycles",
    "green_detected",
    "red_detected",
    "green_area",
    "red_area",
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

BASE_DEFAULT_FEATURES = [
    "blob_area",
    "green_median",
    "green_trimmed_mean",
    "green_p25",
    "green_p75",
    "green_p90",
    "green_p95",
    "green_mean",
    "green_std",
    "green_excess_median",
    "green_integrated_excess",
    "red_median",
    "red_trimmed_mean",
    "red_p25",
    "red_p75",
    "red_p90",
    "red_p95",
    "red_mean",
    "red_std",
    "red_excess_median",
    "red_integrated_excess",
    "rg_ratio_median",
    "rg_ratio_trimmed",
    "dist_to_center_norm",
    "ownership_ratio",
    "dapi_under_blob_mean",
]


def load_trained_model(model_path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    """Loads a serialized joblib pipeline along with its metadata JSON if present."""
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found: {m_path}")
    
    pipeline = joblib.load(m_path)
    meta_path = m_path.with_suffix(".json")
    if not meta_path.exists():
        meta_path = m_path.parent / (m_path.stem + "_metadata.json")
    
    metadata = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata from {meta_path}: {e}")
            
    return pipeline, metadata


def _read_table(data):
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",")
    return data.copy()


def grouped_holdout(y, groups, test_size=0.2, random_state=42):
    """Choose a group-disjoint split with every class represented on both sides.

    Selection inspects labels only, never predictions or feature performance.
    """
    classes = set(np.unique(y))
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if set(y[train]) == classes and set(y[test]) == classes:
            return train, test
    raise ValueError("Cannot construct a nucleus-disjoint holdout with every class; add annotated nuclei.")


def grouped_calibration_folds(y, groups, random_state=42):
    """Prevent cycles of one nucleus leaking between fitting and calibration."""
    classes = set(np.unique(y))
    count = min(len(np.unique(groups[y == c])) for c in classes)
    for n_splits in range(min(3, count), 1, -1):
        for seed in range(random_state, random_state + 10):
            splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                          .split(np.zeros(len(y)), y, groups))
            if all(set(y[a]) == classes and set(y[b]) == classes for a, b in splits):
                return splits
    raise ValueError("Insufficient independent nuclei per class for grouped probability calibration.")


class _AnnotationTrainer:
    binary = False

    def _table(self, data):
        df = _read_table(data)
        if self.binary and self.target_col == "is_real_blob" and (
            "call_status" in df or "blob_call_status" in df or "green_called_correctly" in df
        ):
            # Never reinterpret correctness as presence. Derive before regenerating features.
            df = derive_blob_presence_labels(df)
            df[self.target_col] = df["any_signal_present"]
        if self.target_col not in df:
            raise ValueError(f"Target column {self.target_col!r} not found")
        if not self.binary:
            df[self.target_col] = df[self.target_col].astype(str).str.strip().str.upper()
            df = df[df[self.target_col].isin(["A", "C", "G", "T"])].copy()
        return df.reset_index(drop=True)

    def prepare_data(self, data):
        df = self._table(data)
        if self.binary:
            y = df[self.target_col].astype(str).str.strip().str.lower().map({
                "true": 1, "1": 1, "1.0": 1, "false": 0, "0": 0, "0.0": 0})
            if y.isna().any():
                raise ValueError("Blob presence labels must be explicit booleans, with no missing labels")
            y = y.to_numpy(dtype=int)
        else:
            y = df[self.target_col].to_numpy()
        missing = [f for f in self.features if f not in df]
        if missing and self.strict_features:
            raise ValueError(f"Missing model features: {missing}")
        if missing:
            logger.warning("Legacy training input lacks features; imputing zeros: %s", missing)
        X = df.reindex(columns=self.features, fill_value=0.0).astype(float)
        if self.strict_features and not np.isfinite(X.to_numpy()).all():
            raise ValueError("Model features contain missing or non-finite values")
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0), y, list(self.features)

    def _fit(self, X, y, groups):
        folds = grouped_calibration_folds(y, groups, self.random_state)
        if self.classifier_type == "gradient_boosting":
            estimator = HistGradientBoostingClassifier(random_state=self.random_state,
                max_iter=150, min_samples_leaf=max(2, min(12, len(y) // 20)), l2_regularization=1.0)
        elif self.classifier_type == "random_forest":
            estimator = RandomForestClassifier(n_estimators=160, random_state=self.random_state,
                class_weight="balanced", n_jobs=2, min_samples_leaf=2)
        elif self.classifier_type == "logistic_regression":
            estimator = LogisticRegression(C=1.0, random_state=self.random_state,
                max_iter=1000, class_weight="balanced")
        else:
            raise ValueError(f"Unknown classifier type {self.classifier_type!r}")
        # Scaling belongs inside each calibration fold.
        scaled = Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
        fitted = Pipeline([("classifier", CalibratedClassifierCV(scaled, cv=folds, method="sigmoid"))])
        fitted.fit(X, y)
        return fitted

    def train(self, data, output_dir=None, model_name=None, *, provenance=None):
        df = self._table(data)
        X, y, used_features = self.prepare_data(df)
        if len(np.unique(y)) < 2:
            raise ValueError("At least two target classes are required")
        groups = annotation_groups(df)
        train, test = grouped_holdout(y, groups, self.test_size, self.random_state)
        heldout_model = self._fit(X.iloc[train].to_numpy(), y[train], groups[train])
        probabilities = heldout_model.predict_proba(X.iloc[test].to_numpy())
        pred = ((probabilities[:, list(heldout_model.classes_).index(1)] >= self.prob_threshold).astype(int)
                if self.binary else heldout_model.classes_[probabilities.argmax(axis=1)])
        average = "weighted" if self.binary else "macro"
        prec, recall, f1, _ = precision_recall_fscore_support(y[test], pred, average=average, zero_division=0)
        classes, counts = np.unique(y, return_counts=True)
        metrics = {
            "train_samples": len(train), "test_samples": len(test),
            "train_nuclei": len(np.unique(groups[train])), "test_nuclei": len(np.unique(groups[test])),
            "group_overlap": len(set(groups[train]) & set(groups[test])),
            "accuracy": float(accuracy_score(y[test], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            f"{average}_precision": float(prec), f"{average}_recall": float(recall),
            f"{average}_f1": float(f1),
            "classes": [str(v) for v in classes],
            "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
            "confusion_matrix": confusion_matrix(y[test], pred, labels=classes).tolist(),
            "classification_report": classification_report(y[test], pred, labels=classes,
                output_dict=True, zero_division=0),
            "log_loss": float(log_loss(y[test], probabilities, labels=heldout_model.classes_)),
        }
        if self.binary:
            metrics["prob_threshold"] = self.prob_threshold
            metrics["roc_auc"] = float(roc_auc_score(y[test], probabilities[:, list(heldout_model.classes_).index(1)]))
        self.holdout_predictions = df.iloc[test][[c for c in ("cell", "cycle", "x", "y", self.target_col) if c in df]].copy()
        self.holdout_predictions["predicted"] = pred
        self.holdout_predictions["group"] = groups[test]
        for index, label in enumerate(heldout_model.classes_):
            self.holdout_predictions[f"p_{label}"] = probabilities[:, index]
        self.holdout_model = heldout_model
        self.holdout_train_groups = set(groups[train])
        self.holdout_test_groups = set(groups[test])
        # Final deployed model uses all annotations, with disjoint grouped calibration.
        self.pipeline = self._fit(X.to_numpy(), y, groups)
        self.metadata = {
            "model_type": "blob_classifier" if self.binary else "base_caller",
            "classifier_type": self.classifier_type, "target_col": self.target_col,
            "feature_names": used_features, "classes": [str(v) for v in self.pipeline.classes_],
            "test_size": self.test_size, "random_state": self.random_state,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation": "nucleus_group_holdout", "calibration": "stratified_nucleus_group_sigmoid",
            "final_refit_samples": len(y), "final_refit_nuclei": len(np.unique(groups)),
            "metrics": metrics, "provenance": provenance or {},
        }
        if self.binary:
            self.metadata["prob_threshold"] = self.prob_threshold
        if output_dir is not None:
            self.save(output_dir, model_name)
        return metrics

    def save(self, output_dir, model_name=None):
        if self.pipeline is None:
            raise ValueError("Cannot save an unfitted model")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = model_name or ("blob_classifier" if self.binary else "base_caller")
        model_path, meta_path = out / f"{name}.joblib", out / f"{name}_metadata.json"
        joblib.dump(self.pipeline, model_path)
        meta_path.write_text(json.dumps(self.metadata, indent=2))
        return model_path, meta_path


class BlobModelTrainer(_AnnotationTrainer):
    binary = True

    def __init__(self, features=None, classifier_type="gradient_boosting", target_col="is_real_blob",
                 test_size=0.2, random_state=42, prob_threshold=0.70, strict_features=False):
        self.features = list(features or BLOB_DEFAULT_FEATURES)
        self.classifier_type, self.target_col = classifier_type, target_col
        self.test_size, self.random_state = float(test_size), int(random_state)
        self.prob_threshold, self.strict_features = float(prob_threshold), bool(strict_features)
        self.pipeline, self.metadata = None, {}


class BaseModelTrainer(_AnnotationTrainer):
    def __init__(self, features=None, classifier_type="gradient_boosting", target_col="annotated_base",
                 test_size=0.2, random_state=42, strict_features=False):
        self.features = list(features or BASE_DEFAULT_FEATURES)
        self.classifier_type, self.target_col = classifier_type, target_col
        self.test_size, self.random_state = float(test_size), int(random_state)
        self.strict_features = bool(strict_features)
        self.pipeline, self.metadata = None, {}
