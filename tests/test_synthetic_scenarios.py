"""Comprehensive synthetic benchmark tests demonstrating all 7 required OPS scenarios."""

import numpy as np
import pandas as pd
import pytest

from popcrop.config import PipelineConfig
from popcrop.pipeline import OpsPipeline
from popcrop.io.codebook import Codebook
from popcrop.blob_detection import (
    compute_zscore_and_energy,
    aggregate_cross_cycle_signal,
    segment_extended_blobs,
)
from popcrop.color_calling import CoreAwareColorCaller
from popcrop.decoding import BarcodeDecoder


def create_base_environment(H=64, W=64, T=4, C=2):
    """Helper to initialize synthetic image stack, masks, and backgrounds."""
    sbs_stack = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    bg_maps = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    noise_maps = np.ones((T, C, H, W), dtype=np.float32) * 10.0
    nuclear_mask = np.zeros((H, W), dtype=np.int32)
    y, x = np.ogrid[:H, :W]
    nuclear_mask[(y - 32)**2 + (x - 32)**2 <= 20**2] = 1
    return sbs_stack, bg_maps, noise_maps, nuclear_mask


def test_scenario_1_broad_uniform_bright_plateau():
    """
    Scenario 1: A broad uniform bright plateau is detected despite lacking a sharp local maximum.
    Traditional point-spot detection (Laplacian of Gaussian local maxima) fails here because
    gradient is zero across the interior. Popcrop hysteresis segmenter must detect the whole plateau.
    """
    sbs_stack, bg_maps, noise_maps, nuclear_mask = create_base_environment()
    T, C, H, W = sbs_stack.shape

    for t in range(T):
        sbs_stack[t, 0, 24:40, 24:40] = 400.0

    z_scores, energies = compute_zscore_and_energy(sbs_stack, bg_maps, noise_maps)
    top_k, persist, metric = aggregate_cross_cycle_signal(energies, top_k=2)

    blob_mask, prob_map = segment_extended_blobs(
        metric, nuclear_mask, hysteresis_high=4.0, hysteresis_low=2.0, min_blob_area=10
    )

    assert blob_mask.max() >= 1
    detected_area = np.sum(blob_mask == 1)
    assert detected_area == 256
    assert np.all(prob_map[24:40, 24:40] > 0.95)


def test_scenario_2_off_center_blob():
    """
    Scenario 2: An off-center blob is detected.
    The blob is located near the nuclear periphery, away from the nucleus centroid.
    The pipeline must not assume central location.
    """
    sbs_stack, bg_maps, noise_maps, nuclear_mask = create_base_environment()
    T, C, H, W = sbs_stack.shape

    # Nucleus center is (32, 32). Place blob near periphery at (44, 44)
    for t in range(T):
        sbs_stack[t, 0, 42:48, 42:48] = 400.0

    z_scores, energies = compute_zscore_and_energy(sbs_stack, bg_maps, noise_maps)
    top_k, persist, metric = aggregate_cross_cycle_signal(energies, top_k=2)

    blob_mask, prob_map = segment_extended_blobs(
        metric, nuclear_mask, hysteresis_high=4.0, hysteresis_low=2.0, min_blob_area=5
    )

    assert blob_mask.max() >= 1
    coords = np.argwhere(blob_mask == 1)
    cy, cx = np.mean(coords, axis=0)
    assert cy > 40.0 and cx > 40.0


def test_scenario_3_fragmented_blob_retained_as_multiple_components():
    """
    Scenario 3: A fragmented blob can be retained as multiple related components.
    Two separate foci exist inside the same nucleus.
    """
    sbs_stack, bg_maps, noise_maps, nuclear_mask = create_base_environment()
    T, C, H, W = sbs_stack.shape

    # Component 1 at (24, 24)
    for t in range(T):
        sbs_stack[t, 0, 22:27, 22:27] = 450.0

    # Component 2 at (38, 38)
    for t in range(T):
        sbs_stack[t, 0, 36:41, 36:41] = 450.0

    z_scores, energies = compute_zscore_and_energy(sbs_stack, bg_maps, noise_maps)
    top_k, persist, metric = aggregate_cross_cycle_signal(energies, top_k=2)

    blob_mask, prob_map = segment_extended_blobs(
        metric, nuclear_mask, hysteresis_high=4.0, hysteresis_low=2.0, min_blob_area=5
    )

    assert blob_mask.max() == 2
    comp1_area = np.sum(blob_mask == 1)
    comp2_area = np.sum(blob_mask == 2)
    assert comp1_area == 25
    assert comp2_area == 25


def test_scenario_4_yellow_core_with_stronger_green_rim():
    """
    Scenario 4: A yellow core with a stronger green rim is called yellow.
    Critical rule: "A blob with a spatially coherent yellow center and a stronger or
    larger green rim must be called yellow. A green halo must not override a convincing yellow core."
    """
    H, W = 40, 40
    T, C = 1, 2
    sbs_stack = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    bg = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    noise = np.ones((T, C, H, W), dtype=np.float32) * 10.0

    # Create 14x14 blob at center (13:27, 13:27)
    blob_mask = np.zeros((H, W), dtype=bool)
    blob_mask[13:27, 13:27] = True

    # 1. Core (inner 6x6: 17:23, 17:23):
    # Co-localized Red and Green (Yellow)
    sbs_stack[0, 0, 17:23, 17:23] = 500.0

    # 2. Rim (outer periphery of 14x14):
    # Stronger Green halo: Green = 1500 (3x brighter!), Red = 100
    sbs_stack[0, 1, 13:27, 13:27] = 1500.0
    sbs_stack[0, 1, 17:23, 17:23] = 600.0  # core green

    z_scores = np.maximum(0.0, (sbs_stack - bg) / noise)
    caller = CoreAwareColorCaller(min_connected_yellow_core=4)

    probs = caller.call_blob_cycle_probabilities(
        sbs_stack=sbs_stack,
        background_maps=bg,
        noise_maps=noise,
        z_scores=z_scores,
        component_mask=blob_mask,
        p_real_blob=1.0
    )

    # Must be called yellow, NOT green!
    assert probs[0]["yellow"] > probs[0]["green"]
    assert probs[0]["yellow"] > 0.85


def test_scenario_5_spatially_separate_red_and_green_not_yellow():
    """
    Scenario 5: Spatially separate red and green components are NOT called yellow.
    Critical rule: "Spatially separate red and green regions must not automatically
    be called yellow; return mixed or ambiguous when appropriate."
    """
    H, W = 50, 50
    T, C = 1, 2
    sbs_stack = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    bg = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    noise = np.ones((T, C, H, W), dtype=np.float32) * 10.0

    # Single connected blob component spanning 10:35, 20:30
    blob_mask = np.zeros((H, W), dtype=bool)
    blob_mask[10:35, 20:30] = True

    # Red signal concentrated in top half (10:18)
    sbs_stack[0, 0, 10:18, 20:30] = 500.0
    # Green signal concentrated in bottom half (27:35) -> separated by > 9 pixels!
    sbs_stack[0, 1, 27:35, 20:30] = 500.0

    z_scores = np.maximum(0.0, (sbs_stack - bg) / noise)
    caller = CoreAwareColorCaller(max_spatial_separation=4.0, min_colocalization_overlap=0.25)

    probs = caller.call_blob_cycle_probabilities(
        sbs_stack=sbs_stack,
        background_maps=bg,
        noise_maps=noise,
        z_scores=z_scores,
        component_mask=blob_mask,
        p_real_blob=1.0
    )

    # MUST NOT be called yellow!
    assert probs[0]["yellow"] < 0.05
    # Should be flagged as ambiguous
    assert probs[0]["ambiguous"] > 0.70


def test_scenario_6_background_noise_rejected():
    """
    Scenario 6: Background noise is rejected.
    A nucleus containing only random background noise yields P(real_blob) ~ 0,
    and no false barcode is assigned.
    """
    sbs_stack, bg_maps, noise_maps, nuclear_mask = create_base_environment()
    T, C, H, W = sbs_stack.shape

    np.random.seed(42)
    sbs_stack += np.random.normal(0, 10.0, (T, C, H, W))

    df_cb = pd.DataFrame({"barcode": ["ACGT", "TTTT"], "gene": ["G1", "G2"]})
    codebook = Codebook(df_cb)

    pipeline = OpsPipeline()
    df_nuclei, df_components, masks = pipeline.run_on_arrays(
        sbs_stack=sbs_stack,
        nuclear_mask=nuclear_mask,
        codebook=codebook
    )

    assert len(df_nuclei) == 1
    row = df_nuclei.iloc[0]
    assert row["p_real_blob"] < 0.30
    assert row["decoding_status"] == "no_signal"
    assert row["assigned_barcode"] == "UNASSIGNED"


def test_scenario_7_saturated_data_produce_qc_warning():
    """
    Scenario 7: Saturated data produce a QC warning rather than unjustified confidence.
    """
    sbs_stack, bg_maps, noise_maps, nuclear_mask = create_base_environment()
    T, C, H, W = sbs_stack.shape

    # Create saturated blob at center (value 65535 >= saturation threshold)
    for t in range(T):
        sbs_stack[t, 0, 28:36, 28:36] = 65535.0
        sbs_stack[t, 1, 28:36, 28:36] = 65535.0

    df_cb = pd.DataFrame({"barcode": ["CCCC", "AAAA"], "gene": ["C_GENE", "A_GENE"]})
    codebook = Codebook(df_cb)

    cfg = PipelineConfig()
    cfg.preprocessing.saturation_threshold = 65520.0
    pipeline = OpsPipeline(config=cfg)

    df_nuclei, df_components, masks = pipeline.run_on_arrays(
        sbs_stack=sbs_stack,
        nuclear_mask=nuclear_mask,
        codebook=codebook
    )

    assert len(df_nuclei) == 1
    row = df_nuclei.iloc[0]
    assert "SATURATED" in row["qc_flags"]
    assert row["saturated_pixel_fraction"] > 0.0
