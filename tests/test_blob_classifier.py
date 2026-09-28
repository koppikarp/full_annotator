"""Unit tests for blob classifier and feature extraction."""

import numpy as np
import pandas as pd
import pytest

from popcrop.blob_classifier import extract_component_features, BlobClassifier, FEATURE_NAMES


def test_feature_extraction():
    T, C, H, W = 2, 2, 40, 40
    stack = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    bg = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    noise = np.ones((T, C, H, W), dtype=np.float32) * 5.0
    sat = np.zeros((T, C, H, W), dtype=bool)

    nuclear_mask = np.zeros((H, W), dtype=np.int32)
    nuclear_mask[5:35, 5:35] = 1

    blob_mask = np.zeros((H, W), dtype=np.int32)
    blob_mask[15:25, 15:25] = 1  # 10x10 = 100 pixels

    top_k = np.zeros((H, W), dtype=np.float32)
    top_k[15:25, 15:25] = 12.0
    persist = np.zeros((H, W), dtype=np.int32)
    persist[15:25, 15:25] = 2

    df_feats = extract_component_features(
        sbs_stack=stack,
        background_maps=bg,
        noise_maps=noise,
        saturation_masks=sat,
        blob_mask=blob_mask,
        nuclear_mask=nuclear_mask,
        top_k_map=top_k,
        persistence_map=persist
    )

    assert len(df_feats) == 1
    assert df_feats.iloc[0]["area"] == 100.0
    assert df_feats.iloc[0]["median_contrast"] == 12.0
    assert df_feats.iloc[0]["persistence_cycles"] == 2.0


def test_classifier_fit_and_heuristic():
    clf = BlobClassifier()

    # Create synthetic feature matrix
    np.random.seed(42)
    df_pos = pd.DataFrame({
        "area": np.random.uniform(50, 200, 20),
        "fraction_nucleus_occupied": np.random.uniform(0.1, 0.4, 20),
        "integrated_excess_intensity": np.random.uniform(1000, 5000, 20),
        "median_contrast": np.random.uniform(8.0, 20.0, 20),
        "upper_quantile_contrast": np.random.uniform(10.0, 25.0, 20),
        "peak_contrast": np.random.uniform(15.0, 30.0, 20),
        "edge_to_background_contrast": np.random.uniform(3.0, 8.0, 20),
        "spatial_smoothness": np.random.uniform(0.5, 0.9, 20),
        "solidity": np.random.uniform(0.8, 0.98, 20),
        "eccentricity": np.random.uniform(0.2, 0.7, 20),
        "perimeter_to_area_ratio": np.random.uniform(0.1, 0.3, 20),
        "persistence_cycles": np.random.uniform(3.0, 6.0, 20),
        "distance_from_nuclear_boundary": np.random.uniform(3.0, 10.0, 20),
        "saturated_pixel_fraction": np.zeros(20),
        "component_count": np.ones(20),
    })

    df_neg = pd.DataFrame({
        "area": np.random.uniform(2, 10, 20),
        "fraction_nucleus_occupied": np.random.uniform(0.01, 0.05, 20),
        "integrated_excess_intensity": np.random.uniform(10, 50, 20),
        "median_contrast": np.random.uniform(0.5, 1.5, 20),
        "upper_quantile_contrast": np.random.uniform(1.0, 2.0, 20),
        "peak_contrast": np.random.uniform(1.5, 3.0, 20),
        "edge_to_background_contrast": np.random.uniform(0.8, 1.2, 20),
        "spatial_smoothness": np.random.uniform(0.1, 0.4, 20),
        "solidity": np.random.uniform(0.3, 0.6, 20),
        "eccentricity": np.random.uniform(0.8, 0.99, 20),
        "perimeter_to_area_ratio": np.random.uniform(0.8, 1.5, 20),
        "persistence_cycles": np.random.uniform(0.0, 1.0, 20),
        "distance_from_nuclear_boundary": np.random.uniform(0.0, 1.0, 20),
        "saturated_pixel_fraction": np.zeros(20),
        "component_count": np.ones(20),
    })

    # Test heuristic on pos and neg
    p_pos = clf.predict_proba(df_pos)
    p_neg = clf.predict_proba(df_neg)
    assert np.mean(p_pos) > 0.70
    assert np.mean(p_neg) < 0.30

    # Train model
    X = pd.concat([df_pos, df_neg], ignore_index=True)
    y = np.array([1]*20 + [0]*20)
    clf.fit(X, y)

    p_trained_pos = clf.predict_proba(df_pos)
    p_trained_neg = clf.predict_proba(df_neg)
    assert np.mean(p_trained_pos) > 0.80
    assert np.mean(p_trained_neg) < 0.20
