"""Unit tests for ground truth adapter and calibration derivation."""

import pandas as pd
import numpy as np
import pytest

from popcrop.calibration import GroundTruthAdapter, PipelineCalibrator


def test_ground_truth_adapter_and_calibrator(tmp_path):
    # Create synthetic point-value ground truth
    df = pd.DataFrame({
        "annotated_base": ["A"]*10 + ["T"]*10 + ["C"]*10 + ["G"]*10,
        # A is green only, slight red bleed (10%)
        # T is red only, slight green bleed (5%)
        # C is dual positive, Green:Red ratio = 2.0
        # G is dark
        "red_intensity": [10.0]*10 + [100.0]*10 + [50.0]*10 + [5.0]*10,
        "green_intensity": [100.0]*10 + [5.0]*10 + [100.0]*10 + [5.0]*10,
    })
    csv_file = tmp_path / "ground_truth.csv"
    df.to_csv(csv_file, index=False)

    adapter = GroundTruthAdapter().load_point_values(csv_file)
    stats = adapter.compute_channel_statistics()
    assert "A" in stats
    assert "T" in stats
    assert "C" in stats
    assert "G" in stats

    calibrator = PipelineCalibrator()
    res = calibrator.calibrate_from_ground_truth(adapter)

    # Bleed Green into Red = 10 / 100 = 0.10
    assert res["unmixing_matrix"][0][1] == pytest.approx(0.10, abs=0.01)
    # Bleed Red into Green = 5 / 100 = 0.05
    assert res["unmixing_matrix"][1][0] == pytest.approx(0.05, abs=0.01)
    # Yellow ratio = 100 / 50 = 2.0
    assert res["expected_green_red_ratio_yellow"] == pytest.approx(2.0, abs=0.01)


def test_correctness_annotations_recover_all_presence_combinations():
    from itertools import product
    from popcrop.calibration.ground_truth_adapter import derive_blob_presence_labels
    records = []
    for gd, rd, gc, rc in product((False, True), repeat=4):
        status = {(True, True): "both", (True, False): "only_green",
                  (False, True): "only_red", (False, False): "neither"}[gc, rc]
        records.append(dict(green_detected=gd, red_detected=rd,
            green_called_correctly=gc, red_called_correctly=rc, call_status=status,
            is_real_blob=(gc or rc)))
    original = pd.DataFrame(records)
    unchanged = original.copy(deep=True)
    derived = derive_blob_presence_labels(original)
    for row in derived.itertuples():
        assert row.green_signal_present == (row.green_detected == row.green_called_correctly)
        assert row.red_signal_present == (row.red_detected == row.red_called_correctly)
        assert row.any_signal_present == (row.green_signal_present or row.red_signal_present)
        expected = {(False, False): "G", (True, False): "A", (False, True): "T", (True, True): "C"}
        assert row.derived_base == expected[row.green_signal_present, row.red_signal_present]
    pd.testing.assert_frame_equal(original, unchanged)


def test_correctness_without_historical_detection_fails():
    from popcrop.calibration.ground_truth_adapter import derive_blob_presence_labels
    with pytest.raises(ValueError, match="missing green_detected"):
        derive_blob_presence_labels(pd.DataFrame({"call_status": ["both"]}))


def test_group_identity_normalizes_csv_numeric_ids_and_missing_cells():
    from popcrop.calibration.ground_truth_adapter import annotation_groups
    a = pd.DataFrame({"cell": [459], "tile": [14], "well": ["B2"]})
    b = pd.DataFrame({"cell": [459.0], "tile": [14.0], "well": ["B2"]})
    assert annotation_groups(a)[0] == annotation_groups(b)[0]
    positions = pd.DataFrame({"cell": [np.nan]*3, "x": [5., 5., 8.], "y": [9., 9., 9.]})
    keys = annotation_groups(positions)
    assert keys[0] == keys[1] and keys[0] != keys[2]
