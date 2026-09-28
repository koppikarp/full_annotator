"""Automated tests for spot_engine, multi-tier distributional featurization, and Flask API."""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

from popcrop.annotation.spot_engine import (
    segment_channel_spot,
    segment_cycle_spots,
    segment_crop_blob,
    extract_blob_contours,
    extract_basecaller_distribution_features,
    extract_blob_validation_features,
)
from popcrop.annotation import app as app_module


@pytest.fixture
def sample_crop():
    """Creates a synthetic crop with a known bright circular plateau."""
    np.random.seed(42)
    H, W = 100, 100
    green = np.random.normal(100, 15, (H, W)).astype(np.float32)
    red = np.random.normal(100, 15, (H, W)).astype(np.float32)

    # Add a bright circular plateau in center (radius 15)
    cy, cx = 50, 50
    y_grid, x_grid = np.ogrid[:H, :W]
    spot_mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= (14 ** 2)

    # Green plateau ~3000, Red plateau ~1200
    green[spot_mask] += 2900.0
    red[spot_mask] += 1100.0

    nuc_mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= (30 ** 2)
    return green, red, spot_mask, nuc_mask


def test_spot_segmentation(sample_crop):
    green, red, spot_mask, nuc_mask = sample_crop
    blob_mask, energy, stats = segment_crop_blob(
        green_plane=green,
        red_plane=red,
        nuc_mask=nuc_mask,
        high_thresh=3.5,
        low_thresh=1.8
    )

    assert stats["blob_detected"] is True
    assert stats["blob_area"] > 50
    # Overlap with true spot should be high
    iou = np.sum(blob_mask & spot_mask) / np.sum(blob_mask | spot_mask)
    assert iou > 0.85

    # Check contour extraction
    contours = extract_blob_contours(blob_mask, nuc_mask)
    assert len(contours["blob"]) >= 1
    assert len(contours["nuc"]) >= 1
    assert len(contours["blob"][0]) >= 3


def test_independent_per_channel_spot_detection():
    """Verify independent per-channel spot detection across OPS base rules:
    Base A: Green only
    Base T: Red only
    Base C: Both
    Base G: Neither (Dark)
    """
    H, W = 100, 100
    cy, cx = 50, 50
    y_grid, x_grid = np.ogrid[:H, :W]
    nuc_mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= (35 ** 2)

    spot_green = ((y_grid - 48) ** 2 + (x_grid - 48) ** 2) <= (10 ** 2)
    spot_red = ((y_grid - 53) ** 2 + (x_grid - 53) ** 2) <= (12 ** 2)

    # 1. Base A (Green only)
    g_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    r_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    g_plane[spot_green] += 3000.0  # Green bright spot

    res_a = segment_cycle_spots(g_plane, r_plane, nuc_mask)
    assert res_a["green_detected"] is True
    assert res_a["green_area"] > 50
    assert res_a["red_detected"] is False
    assert res_a["red_area"] == 0
    assert np.sum(res_a["red_mask"]) == 0

    c_a = extract_blob_contours(res_a["composite_mask"], nuc_mask, res_a["green_mask"], res_a["red_mask"])
    assert len(c_a["green_blob"]) >= 1
    assert len(c_a["red_blob"]) == 0

    # 2. Base T (Red only)
    g_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    r_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    r_plane[spot_red] += 3000.0  # Red bright spot

    res_t = segment_cycle_spots(g_plane, r_plane, nuc_mask)
    assert res_t["green_detected"] is False
    assert res_t["green_area"] == 0
    assert res_t["red_detected"] is True
    assert res_t["red_area"] > 50

    c_t = extract_blob_contours(res_t["composite_mask"], nuc_mask, res_t["green_mask"], res_t["red_mask"])
    assert len(c_t["green_blob"]) == 0
    assert len(c_t["red_blob"]) >= 1

    # 3. Base C (Both Green and Red spots present, slightly offset)
    g_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    r_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    g_plane[spot_green] += 3000.0
    r_plane[spot_red] += 2500.0

    res_c = segment_cycle_spots(g_plane, r_plane, nuc_mask)
    assert res_c["green_detected"] is True
    assert res_c["red_detected"] is True
    # Independent masks should not be identical because spots are slightly offset
    assert not np.array_equal(res_c["green_mask"], res_c["red_mask"])

    c_c = extract_blob_contours(res_c["composite_mask"], nuc_mask, res_c["green_mask"], res_c["red_mask"])
    assert len(c_c["green_blob"]) >= 1
    assert len(c_c["red_blob"]) >= 1

    # 4. Base G (Neither: pure background noise)
    g_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)
    r_plane = np.random.normal(100, 10, (H, W)).astype(np.float32)

    res_g = segment_cycle_spots(g_plane, r_plane, nuc_mask)
    assert res_g["green_detected"] is False
    assert res_g["red_detected"] is False
    assert res_g["composite_detected"] is False


def test_basecaller_distribution_features(sample_crop):
    green, red, spot_mask, nuc_mask = sample_crop
    dist = extract_basecaller_distribution_features(green, red, spot_mask, nuc_mask)

    assert dist["has_blob"] is True
    assert dist["blob_area"] == int(np.sum(spot_mask))

    # Tier 1 Quantiles
    g = dist["green"]
    assert 2800 < g["median"] < 3200
    assert g["min"] <= g["p10"] <= g["p25"] <= g["median"] <= g["p75"] <= g["p90"] <= g["p95"] <= g["max"]
    assert g["excess_median"] > 2500  # Elevated above nuclear bg

    # Trimmed Mean (P50 - P90)
    assert 2800 < g["trimmed_mean"] < 3200
    assert g["upper_core_median"] == g["p75"]

    # Red channel
    r = dist["red"]
    assert 1000 < r["median"] < 1400
    assert 1000 < r["trimmed_mean"] < 1400

    # Ratios
    assert 0.3 < dist["rg_ratio_median"] < 0.5
    assert 0.3 < dist["rg_ratio_trimmed"] < 0.5

    # Tier 2 Histograms & JSON serializability
    assert len(g["histogram"]["counts"]) == 32
    json_str = json.dumps(g["histogram"])
    assert len(json_str) > 0


def test_blob_validation_features(sample_crop):
    green, red, spot_mask, nuc_mask = sample_crop
    _, energy, _ = segment_crop_blob(green, red, nuc_mask)
    feat = extract_blob_validation_features(green, red, spot_mask, energy, nuc_mask)

    assert feat["has_blob"] is True
    assert feat["area"] > 50
    assert feat["median_contrast"] > 5.0
    assert feat["solidity"] > 0.7
    assert feat["spatial_smoothness"] > 0.0
    assert feat["p_real_blob"] > 0.80  # True blob should have high P(real)


def test_flask_api_endpoints():
    app = app_module.app
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Config endpoint
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.get_json()
    assert "available_cycles" in data
    assert "base_ground_truth_file" in data
    assert "blob_ground_truth_file" in data

    with tempfile.TemporaryDirectory() as tmpdir:
        # Divert ground truth paths to temp directory
        orig_base = app_module.BASE_GROUND_TRUTH_PATH
        orig_blob = app_module.BLOB_GROUND_TRUTH_PATH
        app_module.BASE_GROUND_TRUTH_PATH = os.path.join(tmpdir, "test_base_gt.csv")
        app_module.BLOB_GROUND_TRUTH_PATH = os.path.join(tmpdir, "test_blob_gt.csv")

        try:
            # 2. Annotate Base Call
            base_payload = {
                "cell_id": 42,
                "base": "C",
                "cycle": 1,
                "tile": 14,
                "well": "B2",
                "x": 120.5,
                "y": 240.2,
                "blob_area": 150,
                "green_median": 3150.0,
                "green_trimmed_mean": 3175.0,
                "green_p25": 2900.0,
                "green_p75": 3300.0,
                "green_p90": 3450.0,
                "green_p95": 3500.0,
                "red_median": 1250.0,
                "red_trimmed_mean": 1270.0,
                "rg_ratio_trimmed": 0.40,
                "green_distribution": {"counts": [1, 5, 20], "bin_edges": [0, 1000, 2000, 3000]},
                "red_distribution": {"counts": [2, 10, 15], "bin_edges": [0, 500, 1000, 1500]}
            }
            res_b = client.post("/api/annotate_base", json=base_payload)
            assert res_b.status_code == 200
            assert res_b.get_json()["success"] is True

            # Verify saved base CSV
            df_base = pd.read_csv(app_module.BASE_GROUND_TRUTH_PATH)
            assert len(df_base) == 1
            assert df_base["annotated_base"].iloc[0] == "C"
            assert df_base["green_trimmed_mean"].iloc[0] == 3175.0
            assert "green_distribution_json" in df_base.columns

            # 3. Annotate Blob Signal (True)
            blob_payload = {
                "cell_id": 42,
                "is_real_blob": True,
                "cycle": 1,
                "tile": 14,
                "well": "B2",
                "x": 120.5,
                "y": 240.2,
                "area": 150,
                "fraction_nucleus_occupied": 0.35,
                "median_contrast": 8.5,
                "peak_contrast": 12.0,
                "spatial_smoothness": 0.45,
                "solidity": 0.92,
                "p_real_blob": 0.94
            }
            res_bl = client.post("/api/annotate_blob", json=blob_payload)
            assert res_bl.status_code == 200
            assert res_bl.get_json()["success"] is True

            # Verify saved blob CSV
            df_blob = pd.read_csv(app_module.BLOB_GROUND_TRUTH_PATH)
            assert len(df_blob) == 1
            assert bool(df_blob["is_real_blob"].iloc[0]) is True
            assert df_blob["p_real_blob"].iloc[0] == 0.94

            # 4. Fetch Ground Truth Lists
            res_gt_b = client.get("/api/ground_truth_base")
            assert res_gt_b.status_code == 200
            assert res_gt_b.get_json()["total"] == 1

            res_gt_bl = client.get("/api/ground_truth_blob")
            assert res_gt_bl.status_code == 200
            assert res_gt_bl.get_json()["total"] == 1

        finally:
            app_module.BASE_GROUND_TRUTH_PATH = orig_base
            app_module.BLOB_GROUND_TRUTH_PATH = orig_blob


def test_per_channel_per_cycle_lut_persistence():
    app = app_module.app
    app.config["TESTING"] = True
    client = app.test_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_lut_path = app_module.LUT_PROFILE_PATH
        app_module.LUT_PROFILE_PATH = os.path.join(tmpdir, "test_saved_lut_profiles.json")

        try:
            # 1. Save Cycle 1 per-channel settings
            c1_payload = {
                "cycle": 1,
                "settings": {
                    "greenMin": 1000,
                    "greenMax": 4000,
                    "greenLut": "viridis",
                    "greenMode": "pseudocolor",
                    "redMin": 300,
                    "redMax": 1200,
                    "redLut": "fire",
                    "redMode": "pseudocolor",
                    "dapiMin": 50,
                    "dapiMax": 200,
                    "dapiLut": "cyan"
                },
                "cycle0_settings": {
                    "cycle0FarredMin": 2500,
                    "cycle0FarredMax": 15000,
                    "cycle0DapiMin": 80,
                    "cycle0DapiMax": 210,
                    "cycle0OverlayBlue": True
                }
            }
            res1 = client.post("/api/lut_profile", json=c1_payload)
            assert res1.status_code == 200

            # 2. Save Cycle 2 per-channel settings with distinct values
            c2_payload = {
                "cycle": 2,
                "settings": {
                    "greenMin": 1500,
                    "greenMax": 5500,
                    "greenLut": "green",
                    "greenMode": "greyscale",
                    "redMin": 500,
                    "redMax": 1800,
                    "redLut": "plasma",
                    "redMode": "pseudocolor",
                    "dapiMin": 100,
                    "dapiMax": 300,
                    "dapiLut": "dapi"
                }
            }
            res2 = client.post("/api/lut_profile", json=c2_payload)
            assert res2.status_code == 200

            # 3. GET LUT profile and verify cycle 1 & cycle 2 have independent per-channel configurations
            res_get = client.get("/api/lut_profile")
            assert res_get.status_code == 200
            profile = res_get.get_json()["profile"]

            cycles = profile["cycles"]
            assert "1" in cycles and "2" in cycles

            # Cycle 1 checks
            assert cycles["1"]["greenMin"] == 1000
            assert cycles["1"]["greenLut"] == "viridis"
            assert cycles["1"]["redLut"] == "fire"
            assert cycles["1"]["dapiLut"] == "cyan"

            # Cycle 2 checks
            assert cycles["2"]["greenMin"] == 1500
            assert cycles["2"]["greenLut"] == "green"
            assert cycles["2"]["redLut"] == "plasma"
            assert cycles["2"]["dapiLut"] == "dapi"

            # Cycle 0 checks
            c0 = profile["cycle0"]
            assert c0["cycle0FarredMin"] == 2500
            assert c0["cycle0DapiMin"] == 80

            # 4. Verify full FOV rendering accepts per-channel colormaps and limits
            res_fov = client.get("/api/full_fov_images?cycle=1&green_min=1000&green_max=4000&green_lut=viridis&red_lut=fire&dapi_lut=cyan&c0_dapi_min=80&c0_dapi_max=210")
            assert res_fov.status_code == 200
            data_fov = res_fov.get_json()
            assert "green_fov" in data_fov
            assert "red_fov" in data_fov
            assert "cycle0_fov" in data_fov

            # 5. Verify cell crop rendering accepts per-channel colormaps and limits
            res_crop = client.get("/api/crop/1?cycle=1&green_min=1000&green_max=4000&green_lut=viridis&red_lut=fire&dapi_lut=cyan&c0_dapi_min=80&c0_dapi_max=210")
            assert res_crop.status_code == 200
            data_crop = res_crop.get_json()
            assert "crops" in data_crop
            assert "green" in data_crop["crops"]
            assert "red" in data_crop["crops"]
            assert "cycle0" in data_crop["crops"]
            assert "contours" in data_crop
            assert "green_blob" in data_crop["contours"]
            assert "red_blob" in data_crop["contours"]
            assert "detection" in data_crop
            assert "green_detected" in data_crop["detection"]
            assert "red_detected" in data_crop["detection"]
            cache_header = res_crop.headers.get("Cache-Control", "")
            assert "no-cache" in cache_header and "no-store" in cache_header

        finally:
            app_module.LUT_PROFILE_PATH = orig_lut_path


def test_tight_spot_boundaries_real_cells():
    """
    Regression test for tight spot boundary delineation and negative channel rejection
    on real microscope data (Cells 3, 4, 5):
      - Cell 3: Both Green and Red detected with slightly relaxed tight boundaries (< 1400 px, NOT 7000 px).
      - Cell 4: Green detected tightly (~460 px, NOT 4300 px); Red dark/noise strictly rejected (0 px).
      - Cell 5: Green tiny punctate speckle (~59 px, NOT 5090 px); Red central blob (< 1400 px, NOT 6626 px).
    """
    tiff_path = app_module.get_tiff_path()
    if not tiff_path or not os.path.exists(tiff_path) or "Test_RGB.tif" not in os.path.basename(tiff_path):
        pytest.skip("Test_RGB.tif not available in test environment")

    with app_module.app.test_client() as client:
        # 1. Cell 4: Base A (Green positive, Red negative)
        r4 = client.get("/api/crop/4?cycle=1")
        assert r4.status_code == 200
        d4 = r4.get_json()
        assert d4["detection"]["green_detected"] is True
        assert 100 < d4["detection"]["green_area"] < 600  # ~460 px, captures full shoulder
        assert d4["detection"]["red_detected"] is False
        assert d4["detection"]["red_area"] == 0
        assert len(d4["contours"]["red_blob"]) == 0

        # 2. Cell 3: Base C (Both channels positive)
        r3 = client.get("/api/crop/3?cycle=1")
        assert r3.status_code == 200
        d3 = r3.get_json()
        assert d3["detection"]["green_detected"] is True
        assert 150 < d3["detection"]["green_area"] < 1400  # bounded by true organic nucleus
        assert d3["detection"]["red_detected"] is True
        assert 300 < d3["detection"]["red_area"] < 1400

        # 3. Cell 5: Base C (Green tiny punctate speckle, Red central blob)
        r5 = client.get("/api/crop/5?cycle=1")
        assert r5.status_code == 200
        d5 = r5.get_json()
        assert d5["detection"]["green_detected"] is True
        assert 10 < d5["detection"]["green_area"] < 120  # ~59 px tiny speckle, NOT 5090 px
        assert d5["detection"]["red_detected"] is True
        assert 500 < d5["detection"]["red_area"] < 1500  # ~1279 px, NOT 6626 px


def test_blob_validator_four_channel_accuracy_options():
    """
    Tests the 4 blob validation options:
      1. 'both': Both called correctly
      2. 'only_green': Only green called correctly
      3. 'only_red': Only red called correctly
      4. 'neither': Neither called correctly
    """
    app = app_module.app
    app.config["TESTING"] = True
    client = app.test_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_blob = app_module.BLOB_GROUND_TRUTH_PATH
        app_module.BLOB_GROUND_TRUTH_PATH = os.path.join(tmpdir, "test_annotated_blob_ground_truth.csv")

        try:
            # 1. Post 'both'
            r_both = client.post("/api/annotate_blob", json={
                "cell_id": 1, "cycle": 1, "call_status": "both",
                "green_detected": True, "red_detected": True, "area": 800
            })
            assert r_both.status_code == 200
            assert r_both.get_json()["call_status"] == "both"
            assert r_both.get_json()["green_called_correctly"] is True
            assert r_both.get_json()["red_called_correctly"] is True

            # 2. Post 'only_green'
            r_g = client.post("/api/annotate_blob", json={
                "cell_id": 2, "cycle": 1, "call_status": "only_green",
                "green_detected": True, "red_detected": True, "area": 400
            })
            assert r_g.status_code == 200
            assert r_g.get_json()["call_status"] == "only_green"
            assert r_g.get_json()["green_called_correctly"] is True
            assert r_g.get_json()["red_called_correctly"] is False

            # 3. Post 'only_red'
            r_r = client.post("/api/annotate_blob", json={
                "cell_id": 3, "cycle": 1, "call_status": "only_red",
                "green_detected": True, "red_detected": True, "area": 500
            })
            assert r_r.status_code == 200
            assert r_r.get_json()["call_status"] == "only_red"
            assert r_r.get_json()["green_called_correctly"] is False
            assert r_r.get_json()["red_called_correctly"] is True

            # 4. Post 'neither'
            r_n = client.post("/api/annotate_blob", json={
                "cell_id": 4, "cycle": 1, "call_status": "neither",
                "green_detected": False, "red_detected": False, "area": 0
            })
            assert r_n.status_code == 200
            assert r_n.get_json()["call_status"] == "neither"
            assert r_n.get_json()["green_called_correctly"] is False
            assert r_n.get_json()["red_called_correctly"] is False

            # Verify saved CSV
            df = pd.read_csv(app_module.BLOB_GROUND_TRUTH_PATH)
            assert len(df) == 4
            assert list(df["call_status"]) == ["both", "only_green", "only_red", "neither"]

            # Verify /api/cells reflects the 4 statuses
            r_cells = client.get("/api/cells?cycle=1")
            assert r_cells.status_code == 200
            cells_data = r_cells.get_json()["cells"]
            c_map = {c["cell_id"]: c.get("blob_call_status") for c in cells_data}
            assert c_map[1] == "both"
            assert c_map[2] == "only_green"
            assert c_map[3] == "only_red"
            assert c_map[4] == "neither"

        finally:
            app_module.BLOB_GROUND_TRUTH_PATH = orig_blob


def test_segment_nucleus_crop_watershed():
    """Tests classical adaptive watershed nuclear segmentation on synthetic DAPI."""
    H, W = 100, 100
    dapi = np.zeros((H, W), dtype=np.float32)
    # Target oval nucleus centered at (50, 50)
    y_grid, x_grid = np.ogrid[:H, :W]
    target_oval = (((y_grid - 50) / 16.0) ** 2 + ((x_grid - 50) / 10.0) ** 2) <= 1.0
    dapi[target_oval] = 5000.0

    # Touching neighbor nucleus at (50, 80)
    neighbor_oval = (((y_grid - 50) / 14.0) ** 2 + ((x_grid - 80) / 12.0) ** 2) <= 1.0
    dapi[neighbor_oval] = 4800.0

    # Segment target at (50, 50)
    mask = app_module.segment_nucleus_crop_watershed(dapi, cy_rel=50, cx_rel=50, expected_radius=15.0)
    assert mask.dtype == bool
    assert mask.shape == (H, W)
    # Target center must be covered
    assert bool(mask[50, 50]) is True
    # Touching neighbor must be excluded!
    assert bool(mask[50, 80]) is False
    assert int(np.sum(mask & neighbor_oval)) == 0

    # Check contour extraction
    cnts = extract_blob_contours(nuc_mask=mask)
    assert len(cnts["nuc"]) == 1
    # Contour points should be non-empty
    assert len(cnts["nuc"][0]) >= 10


def test_get_cell_nuclear_mask_caching():
    """Tests that get_cell_nuclear_mask caches results per (fingerprint, cell_id, diameter)."""
    app_module.NUC_CROP_CACHE.clear()
    H, W = 100, 100
    dapi_full = np.zeros((200, 200), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:200, :200]
    dapi_full[((y_grid - 100) ** 2 + (x_grid - 100) ** 2) <= (20 ** 2)] = 6000.0

    m1 = app_module.get_cell_nuclear_mask(
        dapi_full=dapi_full,
        y_c=100, x_c=100,
        y_min=50, y_max=150,
        x_min=50, x_max=150,
        cell_id=999,
        fingerprint="test_fp",
        expected_diameter=40.0
    )
    assert np.sum(m1) > 0
    cache_key = "test_fp_999_40"
    assert cache_key in app_module.NUC_CROP_CACHE
    # Second call returns exact cached mask
    m2 = app_module.get_cell_nuclear_mask(
        dapi_full=dapi_full,
        y_c=100, x_c=100,
        y_min=50, y_max=150,
        x_min=50, x_max=150,
        cell_id=999,
        fingerprint="test_fp",
        expected_diameter=40.0
    )
    assert m1 is m2


def test_api_crop_organic_nuclear_contour():
    """Tests that /api/crop endpoint returns organic nuclear contours."""
    client = app_module.app.test_client()
    res = client.get("/api/crop/1?cycle=1")
    assert res.status_code == 200
    data = res.get_json()
    assert "contours" in data
    contours = data["contours"]
    assert "nuc" in contours
    assert len(contours["nuc"]) >= 1
    poly = contours["nuc"][0]
    assert len(poly) >= 20


def test_crop_analysis_cache_and_lut_decoupling():
    """Tests that pure LUT changes reuse cached spot analysis without re-segmenting."""
    app_module.CROP_ANALYSIS_CACHE.clear()
    client = app_module.app.test_client()

    # Call 1: Default LUT values
    res1 = client.get("/api/crop/1?cycle=1&green_min=0&green_max=65535&red_min=0&red_max=65535")
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert len(app_module.CROP_ANALYSIS_CACHE) == 1

    # Call 2: Modified LUT values (pure display change)
    res2 = client.get("/api/crop/1?cycle=1&green_min=1000&green_max=15000&red_min=500&red_max=8000&lut_only=1")
    assert res2.status_code == 200
    data2 = res2.get_json()

    # Cache should still have 1 entry (cache hit!)
    assert len(app_module.CROP_ANALYSIS_CACHE) == 1

    # Analytical features and contours should be exactly identical
    assert data1["base_features"] == data2["base_features"]
    assert data1["blob_features"] == data2["blob_features"]
    assert data1["contours"] == data2["contours"]
    assert data1["detection"] == data2["detection"]

    # But rendered crop images should be updated with new normalization
    assert data1["crops"]["green"] != data2["crops"]["green"]
    assert data1["crops"]["red"] != data2["crops"]["red"]

    # Call 3: Changing segmentation threshold parameters creates a new entry
    res3 = client.get("/api/crop/1?cycle=1&hysteresis_high=5.0")
    assert res3.status_code == 200
    assert len(app_module.CROP_ANALYSIS_CACHE) == 2



