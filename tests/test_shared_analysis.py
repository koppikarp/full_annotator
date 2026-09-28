"""Behavioral checks for the raw-image contract shared by UI, training and batch."""
import json

import numpy as np
import pandas as pd
import pytest

from popcrop.annotation import app as ui
from popcrop.annotation.analysis import analyze_nucleus_cycle
from popcrop.annotation.spot_engine import segment_cycle_spots, extract_basecaller_distribution_features


@pytest.fixture
def raw_planes():
    rng = np.random.default_rng(24)
    yy, xx = np.mgrid[:128, :128]
    nucleus = (yy - 63.2) ** 2 + (xx - 65.1) ** 2 < 22 ** 2
    spot = (yy - 61) ** 2 + (xx - 65) ** 2 < 8 ** 2
    green = (rng.normal(1200, 20, (128, 128)) + 6500 * spot).astype(np.uint16)
    red = (rng.normal(600, 15, (128, 128)) + 2200 * spot).astype(np.uint16)
    dapi = (200 + 6500 * nucleus).astype(np.uint16)
    return green, red, dapi


@pytest.fixture
def client(raw_planes, monkeypatch, tmp_path):
    green, red, dapi = raw_planes
    stack = np.stack([red, dapi, dapi, green, red, red])
    monkeypatch.setattr(ui, "get_tiff_path", lambda: "synthetic_raw.tif")
    monkeypatch.setattr(ui, "load_tiff_image", lambda path: stack)
    monkeypatch.setattr(ui, "get_current_tiff_fingerprint", lambda: "image_a")
    monkeypatch.setattr(ui, "load_cell_coordinates", lambda: pd.DataFrame([
        {"cell": 1, "i": 62.0, "j": 63.0}, {"cell": 2, "i": 110., "j": 110.}]))
    monkeypatch.setattr(ui, "BASE_GROUND_TRUTH_PATH", str(tmp_path / "base.csv"))
    monkeypatch.setattr(ui, "BLOB_GROUND_TRUTH_PATH", str(tmp_path / "blob.csv"))
    ui.CROP_ANALYSIS_CACHE.clear()
    yield ui.app.test_client()
    ui.CROP_ANALYSIS_CACHE.clear()


def test_helper_matches_detector_and_distribution(raw_planes):
    result = analyze_nucleus_cycle(*raw_planes, (62., 63.), neighbor_centers_yx=[[110, 110]])
    direct = segment_cycle_spots(result["g_crop"], result["r_crop"], result["nuc_mask"],
                                 **result["provenance"]["detector_params"])
    expected = extract_basecaller_distribution_features(result["g_crop"], result["r_crop"],
        direct["composite_mask"], result["nuc_mask"], direct["green_mask"], direct["red_mask"])
    assert np.array_equal(result["composite_mask"], direct["composite_mask"])
    assert all(result["base_features"][key] == value for key, value in expected.items())
    assert result["provenance"]["nuclear_area"] == int(result["nuc_mask"].sum())
    assert result["base_row"]["green_samples"] == expected["green"]["samples"]


def test_ui_equals_batch_and_lut_is_display_only(raw_planes, client):
    first = client.get("/api/crop/1?cycle=1&nuclear_diameter=55").get_json()
    second = client.get("/api/crop/1?cycle=1&nuclear_diameter=55&green_min=2500&green_max=4000&red_lut=fire").get_json()
    batch = analyze_nucleus_cycle(*raw_planes, (62., 63.), nuclear_diameter=55,
                                 reference_dapi=raw_planes[2], neighbor_centers_yx=[[110, 110]], include_contours=True)
    for field in ("base_features", "blob_features", "detection", "contours"):
        assert first[field] == second[field] == batch[field]
    assert first["analysis_token"] == second["analysis_token"]
    assert first["crops"]["green"] != second["crops"]["green"]
    assert len(ui.CROP_ANALYSIS_CACHE) == 1


def test_exact_settings_and_features_saved_from_server(client):
    shown = client.get("/api/crop/1?cycle=1&nuclear_diameter=55.03&hysteresis_high=3.21").get_json()
    payload = {"cell_id": 1, "cycle": 1, "analysis_token": shown["analysis_token"],
               "base": "C", "green_median": -100, "x": -10, "y": -10}
    response = client.post("/api/annotate_base", json=payload)
    assert response.status_code == 200
    row = pd.read_csv(ui.BASE_GROUND_TRUTH_PATH).iloc[0]
    provenance = json.loads(row.analysis_provenance_json)
    assert row.nuclear_diameter == 55.03
    assert provenance["detector_params"]["high_thresh"] == 3.21
    assert row.green_median == shown["base_features"]["green"]["median"]
    assert row.x == 63.0 and row.y == 62.0
    assert json.loads(row.green_samples_json) == shown["base_features"]["green"]["samples"]
    assert json.loads(row.green_distribution_json) == shown["base_features"]["green"]["histogram"]
    payload["cell_id"] = 2
    assert client.post("/api/annotate_base", json=payload).status_code == 409


@pytest.mark.parametrize("green_detected", [False, True])
@pytest.mark.parametrize("red_detected", [False, True])
@pytest.mark.parametrize("status", ["both", "only_green", "only_red", "neither"])
def test_correctness_is_not_presence(client, green_detected, red_detected, status):
    response = client.post("/api/annotate_blob", json={"cell_id": 1, "cycle": 1,
        "call_status": status, "green_detected": green_detected, "red_detected": red_detected})
    assert response.status_code == 200
    row = pd.read_csv(ui.BLOB_GROUND_TRUTH_PATH).iloc[0]
    expected_g = green_detected == (status in ("both", "only_green"))
    expected_r = red_detected == (status in ("both", "only_red"))
    assert bool(row.green_true_presence) == expected_g
    assert bool(row.red_true_presence) == expected_r
    assert bool(row.is_real_blob) == (expected_g or expected_r)
    assert client.get("/api/cells?cycle=1").get_json()["cells"][0]["is_real_blob"] == (expected_g or expected_r)


def test_diameter_changes_analysis_and_key(client):
    small = client.get("/api/crop/1?nuclear_diameter=55").get_json()
    big = client.get("/api/crop/1?nuclear_diameter=75").get_json()
    assert small["analysis_token"] != big["analysis_token"]
    assert small["analysis_provenance"]["nuclear_area"] < big["analysis_provenance"]["nuclear_area"]
    assert small["blob_features"]["fraction_nucleus_occupied"] > big["blob_features"]["fraction_nucleus_occupied"]


def test_same_ids_from_different_images_survive(client, monkeypatch):
    shown = client.get("/api/crop/1").get_json()
    payload = {"cell_id": 1, "base": "C", "analysis_token": shown["analysis_token"]}
    assert client.post("/api/annotate_base", json=payload).status_code == 200
    monkeypatch.setattr(ui, "get_current_tiff_fingerprint", lambda: "image_b")
    assert client.post("/api/annotate_base", json=payload).status_code == 409
    shown = client.get("/api/crop/1").get_json()
    payload["analysis_token"] = shown["analysis_token"]
    assert client.post("/api/annotate_base", json=payload).status_code == 200
    assert len(pd.read_csv(ui.BASE_GROUND_TRUTH_PATH)) == 2


def test_missing_neighbors_is_valid_json(raw_planes, client, monkeypatch):
    monkeypatch.setattr(ui, "load_cell_coordinates", lambda: pd.DataFrame([{"cell": 1, "i": 62., "j": 63.}]))
    response = client.get("/api/crop/1")
    assert response.status_code == 200
    assert "NaN" not in response.text
    assert response.get_json()["base_features"]["ownership_ratio"] is None


def test_invalid_settings_fail(client):
    assert client.get("/api/crop/1?nuclear_diameter=nan").status_code == 400
    assert client.get("/api/crop/1?radius_scale=0").status_code == 400
    assert client.get("/api/crop/1?max_area_fraction=1.5").status_code == 400
