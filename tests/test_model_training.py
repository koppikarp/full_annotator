"""Unit tests for model training scripts and serialization."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from popcrop.models.trainer import (
    BlobModelTrainer,
    BaseModelTrainer,
    load_trained_model,
    BLOB_DEFAULT_FEATURES,
    BASE_DEFAULT_FEATURES,
)
from popcrop.blob_classifier import BlobClassifier
from popcrop.color_calling import ModelBasedBaseCaller, CoreAwareColorCaller


@pytest.fixture
def synthetic_blob_ground_truth():
    """Generates synthetic blob ground truth matching annotated_blob_ground_truth.csv."""
    np.random.seed(42)
    n = 60
    is_real = np.array([1] * 30 + [0] * 30)

    data = {
        "cell": np.arange(1, n + 1),
        "cycle": [1] * n,
        "tile": [14] * n,
        "well": ["B2"] * n,
        "x": np.random.uniform(100, 2000, n),
        "y": np.random.uniform(100, 2000, n),
        "is_real_blob": [bool(x) for x in is_real],
        "area": np.where(is_real == 1, np.random.uniform(100, 500, n), np.random.uniform(5, 30, n)),
        "fraction_nucleus_occupied": np.where(is_real == 1, np.random.uniform(0.1, 0.4, n), np.random.uniform(0.01, 0.05, n)),
        "median_contrast": np.where(is_real == 1, np.random.uniform(8.0, 25.0, n), np.random.uniform(0.5, 2.0, n)),
        "peak_contrast": np.where(is_real == 1, np.random.uniform(15.0, 40.0, n), np.random.uniform(1.0, 3.5, n)),
        "upper_quantile_contrast": np.where(is_real == 1, np.random.uniform(10.0, 30.0, n), np.random.uniform(0.8, 2.5, n)),
        "integrated_excess_intensity": np.where(is_real == 1, np.random.uniform(2000, 10000, n), np.random.uniform(20, 200, n)),
        "spatial_smoothness": np.random.uniform(0.4, 0.9, n),
        "solidity": np.where(is_real == 1, np.random.uniform(0.75, 0.98, n), np.random.uniform(0.3, 0.6, n)),
        "eccentricity": np.random.uniform(0.2, 0.8, n),
        "perimeter_to_area_ratio": np.where(is_real == 1, np.random.uniform(0.05, 0.2, n), np.random.uniform(0.5, 1.2, n)),
        "persistence_cycles": np.where(is_real == 1, np.random.uniform(2.0, 8.0, n), np.random.uniform(0.0, 1.0, n)),
        "distance_from_nuclear_boundary": np.random.uniform(2.0, 15.0, n),
        "saturated_pixel_fraction": np.zeros(n),
        "component_count": [1] * n,
        "green_detected": [bool(x) for x in (is_real == 1)],
        "red_detected": [bool(x) for x in (is_real == 1)],
        "green_area": np.where(is_real == 1, np.random.uniform(50, 250, n), 0),
        "red_area": np.where(is_real == 1, np.random.uniform(50, 250, n), 0),
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_base_ground_truth():
    """Generates synthetic base ground truth matching annotated_base_ground_truth.csv."""
    np.random.seed(42)
    bases = ["A", "C", "G", "T"] * 20  # 80 samples, 20 each
    n = len(bases)

    rows = []
    for i, b in enumerate(bases):
        if b == "A":  # Green dominant, Red low
            g_med = np.random.uniform(4000, 8000)
            r_med = np.random.uniform(100, 400)
        elif b == "T":  # Red dominant, Green low
            g_med = np.random.uniform(100, 400)
            r_med = np.random.uniform(3000, 7000)
        elif b == "C":  # Both high
            g_med = np.random.uniform(3500, 7000)
            r_med = np.random.uniform(2500, 6000)
        else:  # G: Dark / neither
            g_med = np.random.uniform(80, 250)
            r_med = np.random.uniform(80, 250)

        rows.append({
            "cell": i + 1,
            "cycle": 1,
            "tile": 14,
            "well": "B2",
            "x": 500.0 + i,
            "y": 500.0 + i,
            "annotated_base": b,
            "blob_area": int(np.random.uniform(200, 800)),
            "green_median": g_med,
            "green_trimmed_mean": g_med * 1.1,
            "green_p25": g_med * 0.8,
            "green_p75": g_med * 1.2,
            "green_p90": g_med * 1.4,
            "green_p95": g_med * 1.5,
            "green_mean": g_med * 1.05,
            "green_std": g_med * 0.25,
            "green_excess_median": max(0.0, g_med - 150.0),
            "green_integrated_excess": max(0.0, g_med - 150.0) * 400.0,
            "red_median": r_med,
            "red_trimmed_mean": r_med * 1.1,
            "red_p25": r_med * 0.8,
            "red_p75": r_med * 1.2,
            "red_p90": r_med * 1.4,
            "red_p95": r_med * 1.5,
            "red_mean": r_med * 1.05,
            "red_std": r_med * 0.25,
            "red_excess_median": max(0.0, r_med - 150.0),
            "red_integrated_excess": max(0.0, r_med - 150.0) * 400.0,
            "rg_ratio_median": round(r_med / max(1.0, g_med), 3),
            "rg_ratio_trimmed": round((r_med * 1.1) / max(1.0, g_med * 1.1), 3),
        })
    return pd.DataFrame(rows)


def test_blob_model_trainer(synthetic_blob_ground_truth):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = BlobModelTrainer(test_size=0.25, random_state=42)
        metrics = trainer.train(
            data=synthetic_blob_ground_truth,
            output_dir=tmpdir,
            model_name="test_blob_model"
        )

        assert metrics["train_samples"] == 45
        assert metrics["test_samples"] == 15
        assert metrics["accuracy"] >= 0.85
        assert metrics["weighted_f1"] >= 0.85
        assert "roc_auc" in metrics

        joblib_path = os.path.join(tmpdir, "test_blob_model.joblib")
        json_path = os.path.join(tmpdir, "test_blob_model_metadata.json")
        assert os.path.exists(joblib_path)
        assert os.path.exists(json_path)

        # Load into BlobClassifier and predict
        clf = BlobClassifier(model_path=joblib_path)
        probs = clf.predict_proba(synthetic_blob_ground_truth.iloc[:5])
        assert len(probs) == 5
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        # First 5 are real blobs
        assert np.all(probs > 0.7)


def test_base_model_trainer(synthetic_base_ground_truth):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = BaseModelTrainer(test_size=0.25, random_state=42)
        metrics = trainer.train(
            data=synthetic_base_ground_truth,
            output_dir=tmpdir,
            model_name="test_base_model"
        )

        assert metrics["train_samples"] == 60
        assert metrics["test_samples"] == 20
        assert metrics["accuracy"] >= 0.90
        assert metrics["macro_f1"] >= 0.90
        assert len(metrics["classes"]) == 4

        joblib_path = os.path.join(tmpdir, "test_base_model.joblib")
        assert os.path.exists(joblib_path)

        # Load into ModelBasedBaseCaller
        model_caller = ModelBasedBaseCaller(model_path=joblib_path)
        probs = model_caller.predict_cycle_probabilities(synthetic_base_ground_truth.iloc[:4])
        assert len(probs) == 4
        for p in probs:
            assert "green" in p and "red" in p and "yellow" in p and "dark" in p
            # Should sum to approximately 1.0
            assert abs(sum(p.values()) - 1.0) < 1e-2

        # Check mapping: Sample 0 is Base A -> green should be dominant
        assert probs[0]["green"] > probs[0]["red"]
        # Sample 1 is Base C -> yellow should be dominant
        assert probs[1]["yellow"] > probs[1]["dark"]
        # Sample 2 is Base G -> dark should be dominant
        assert probs[2]["dark"] > probs[2]["green"]
        # Sample 3 is Base T -> red should be dominant
        assert probs[3]["red"] > probs[3]["green"]


def test_grouped_holdout_and_calibration_exclude_all_cycles_of_test_nuclei():
    from popcrop.models.trainer import grouped_holdout, grouped_calibration_folds
    groups = np.repeat(np.arange(40), 12)
    y = np.tile(np.array(["A", "C", "G", "T"] * 3), 40)
    train, test = grouped_holdout(y, groups)
    assert not set(groups[train]) & set(groups[test])
    for fit, calibration in grouped_calibration_folds(y[train], groups[train]):
        assert not set(groups[train][fit]) & set(groups[train][calibration])
        assert set(y[train][fit]) == set(y[train][calibration]) == {"A", "C", "G", "T"}


def test_trainer_derives_presence_and_rejects_unknown_labels():
    df = pd.DataFrame({"call_status": ["both", "neither", "only_green"],
        "green_detected": [False, False, True], "red_detected": [False, False, True],
        "is_real_blob": [True, False, True], "median_contrast": [0., 0., 5.]})
    _, y, _ = BlobModelTrainer(features=["median_contrast"]).prepare_data(df)
    assert y.tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="Missing model features"):
        BaseModelTrainer(features=["absent"], strict_features=True).prepare_data(
            pd.DataFrame({"annotated_base": ["A", "N"]}))
