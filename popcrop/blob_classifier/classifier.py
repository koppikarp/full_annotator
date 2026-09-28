"""Calibrated real-blob vs artifact classifier with machine-learning and analytical fallback modes."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
import json
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from popcrop.blob_classifier.features import FEATURE_NAMES, BLOB_FEATURE_NAMES
from popcrop.models.trainer import load_trained_model

logger = logging.getLogger(__name__)


class BlobClassifier:
    """Classifies candidate components as real fluorescent blobs vs imaging artifacts."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        prob_threshold: float = 0.70,
        classifier_type: str = "logistic_regression",
        feature_names: Optional[List[str]] = None
    ):
        self.prob_threshold = float(prob_threshold)
        self.classifier_type = classifier_type
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = list(feature_names) if feature_names else list(FEATURE_NAMES)

        if model_path is not None and Path(model_path).exists():
            self.load(model_path)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> BlobClassifier:
        """
        Fits a calibrated classifier on object-level features.
        y is binary: 1 for real blob, 0 for artifact / negative control.
        """
        logger.info(f"Training calibrated {self.classifier_type} on {len(y)} samples.")
        if isinstance(X, pd.DataFrame):
            if feature_names is not None:
                self.feature_names = list(feature_names)
            else:
                # Use columns present in BLOB_FEATURE_NAMES or FEATURE_NAMES, or all numeric columns
                common = [c for c in BLOB_FEATURE_NAMES if c in X.columns]
                self.feature_names = common if common else [c for c in FEATURE_NAMES if c in X.columns]
                if not self.feature_names:
                    self.feature_names = list(X.columns)

            X_mat = X[self.feature_names].values
        else:
            X_mat = X

        if self.classifier_type == "gradient_boosting":
            base_estimator = HistGradientBoostingClassifier(random_state=42, max_iter=100)
            calibrated = CalibratedClassifierCV(base_estimator, cv=3)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", calibrated)
            ])
        else:
            base_estimator = LogisticRegression(penalty="l2", C=1.0, random_state=42, max_iter=500)
            calibrated = CalibratedClassifierCV(base_estimator, cv=3)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", calibrated)
            ])

        pipeline.fit(X_mat, y)
        self.pipeline = pipeline
        logger.info("Classifier training and calibration complete.")
        return self

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Returns P(real_blob) for every row in df_features.
        If no trained pipeline exists, uses calibrated analytical heuristic.
        """
        if df_features.empty:
            return np.array([], dtype=np.float32)

        if self.pipeline is not None:
            # Build feature matrix safely, imputing 0.0 for any missing features
            feats = self.feature_names or FEATURE_NAMES
            X_df = pd.DataFrame(index=df_features.index)
            for f in feats:
                if f in df_features.columns:
                    X_df[f] = df_features[f].astype(float)
                else:
                    X_df[f] = 0.0
            probs = self.pipeline.predict_proba(X_df.values)[:, 1]
            return probs.astype(np.float32)

        return self._analytical_heuristic(df_features)

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """Returns binary predictions (1 for real blob, 0 for artifact) using prob_threshold."""
        probs = self.predict_proba(df_features)
        return (probs >= self.prob_threshold).astype(int)

    def _analytical_heuristic(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Calibrated analytical score based on SNR, temporal persistence, contrast,
        and morphology when no trained model is supplied.
        """
        snr = df_features["median_contrast"].values.astype(np.float32)
        upper_snr = df_features["upper_quantile_contrast"].values.astype(np.float32)
        persistence = df_features["persistence_cycles"].values.astype(np.float32)
        solidity = df_features["solidity"].values.astype(np.float32)
        area = df_features["area"].values.astype(np.float32)
        sat_frac = df_features["saturated_pixel_fraction"].values.astype(np.float32)
        dist_border = df_features["distance_from_nuclear_boundary"].values.astype(np.float32)

        # 1. SNR term: smooth transition around z=2.2
        snr_term = 1.0 / (1.0 + np.exp(-1.0 * (snr - 2.2)))

        # 2. Persistence term: rewards blobs appearing across multiple cycles
        persist_term = np.clip(persistence / 2.0, 0.0, 1.0)

        # 3. Shape and Area term
        area_term = np.clip(area / 8.0, 0.0, 1.0)
        solidity_term = np.clip(solidity, 0.4, 1.0)

        # 4. Border term: mild effect (0.85 to 1.0) since T7 blobs can occupy much of nucleus
        border_term = np.clip(0.85 + 0.15 * (dist_border / 2.0), 0.85, 1.0)

        # 5. Saturation penalty
        sat_penalty = 1.0 - 0.4 * sat_frac

        score = (
            0.50 * snr_term
            + 0.25 * persist_term
            + 0.15 * (upper_snr / (upper_snr + 2.5))
            + 0.10 * (0.5 * area_term + 0.5 * solidity_term)
        ) * border_term * sat_penalty

        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        meta_path = path.with_suffix(".json")
        meta = {
            "feature_names": self.feature_names,
            "prob_threshold": self.prob_threshold,
            "classifier_type": self.classifier_type
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved blob classifier to {path}")

    def load(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier model file not found: {path}")
        data = joblib.load(path)
        if isinstance(data, dict) and "pipeline" in data:
            self.pipeline = data["pipeline"]
            self.prob_threshold = float(data.get("prob_threshold", self.prob_threshold))
        else:
            self.pipeline = data
            meta_path = path.with_suffix(".json")
            if not meta_path.exists():
                meta_path = path.parent / (path.stem + "_metadata.json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    if "feature_names" in self.metadata:
                        self.feature_names = self.metadata["feature_names"]
                    if "prob_threshold" in self.metadata:
                        self.prob_threshold = float(self.metadata["prob_threshold"])
                except Exception as e:
                    logger.warning(f"Could not load metadata from {meta_path}: {e}")
        logger.info(f"Loaded blob classifier from {path} with prob_threshold={self.prob_threshold}")
