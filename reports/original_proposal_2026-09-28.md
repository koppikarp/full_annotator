# Intranuclear Spot Detection & Full Annotator Parity Implementation Plan

## Goal Description
Upgrade the optical pooled screening pipeline to achieve 100% mathematical and algorithmic parity with the manual basecaller (version 2.1) annotator. Replace global slide-level hysteresis thresholding with **intranuclear local baseline spot detection**, retrain the blob classifier on intrinsic photometric features (eliminating the skewed `fraction_nucleus_occupied` metric and repairing the training labels), vectorize the 15,766-barcode log-likelihood decoder, and process `wellb2_002_002_RGB.tif` to detect all true amplicons with 0–1 Hamming distance confidence.

---

## User Review Required

> [!IMPORTANT]
> **No Re-annotation Needed**: All 1,776 cycle annotations across 1,544 unique cells already contain the raw, uncorrupted photometric features (`max_peak_excess`, `fold_change`, `bg`, `noise`, etc.). We do not need to re-annotate cells.
> 
> **Eliminating `fraction_nucleus_occupied`**: As identified, `fraction_nucleus_occupied` fluctuates based on nuclear diameter settings (50 vs 90 px) and cell size, rather than physical amplicon biology. The retrained model will use intrinsic photometric metrics (`max_peak_excess`, `max_fold_change`, `contrast`, `solidity`, and `persistence`).

---

## Proposed Changes

### Component 1: Ground Truth Dataset & Blob Classifier Retraining

#### [MODIFY] [annotated_blob_ground_truth.csv](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/annotated_blob_ground_truth.csv)
- Correct label bug where `is_real_blob` was previously set to `True` only for `both` (dual-positive), which caused genuine `only_green` and `only_red` spots to be mislabeled as negative controls. Set `is_real_blob = (call_status != 'neither')`.
- Remove the single corrupted outlier row (cell 4179, cycle 3) where `median_contrast` was 1,304,035 counts.

#### [MODIFY] [popcrop/models/trainer.py](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/popcrop/models/trainer.py)
- Update default blob training feature list to use intrinsic features:
  `['max_peak_excess', 'max_fold_change', 'peak_contrast', 'median_contrast', 'upper_quantile_contrast', 'persistence_cycles', 'solidity', 'eccentricity', 'spatial_smoothness']`.
- Explicitly omit `fraction_nucleus_occupied` and `area` so nuclear diameter differences cannot distort the model.
- Retrain and serialize `models/blob_classifier.joblib` with probability calibration.

---

### Component 2: Intranuclear Local Baseline Spot Detection

#### [NEW] [popcrop/blob_detection/intranuclear.py](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/popcrop/blob_detection/intranuclear.py)
- Implement `detect_intranuclear_spots_batch`:
  - For each segmented nucleus (across all 5,378 nuclei), extract the bounding box ROI.
  - Compute the local intranuclear baseline and MAD noise strictly from the lower 40% of pixels inside `nuc_mask` (matching `compute_robust_bg_and_noise` in [`spot_engine.py`](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/popcrop/annotation/spot_engine.py#L13-L46)).
  - Apply the exact dual-gated photometric criteria from the annotator:
    $$\text{peak\_excess} = \text{peak\_val} - \text{bg}_{\text{nuc}} \ge \text{min\_excess}$$
    $$\text{peak\_snr} = \frac{\text{peak\_excess}}{\text{noise}_{\text{nuc}}} \ge \text{min\_snr}$$
    $$\text{fold} = \frac{\text{peak\_val}}{\max(1.0, \text{bg}_{\text{nuc}})} \ge \text{min\_fold}$$
  - For cells with detected spots, threshold at $\text{th} = \text{bg}_{\text{nuc}} + \alpha \times \text{peak\_excess}$ ($\alpha = 0.28$) and isolate the primary amplicon component.
  - Return a consolidated `blob_mask` containing true 50–350 px amplicon puncta.

#### [MODIFY] [popcrop/pipeline.py](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/popcrop/pipeline.py)
- Wire `detect_intranuclear_spots_batch` into Step 2 of `OpsPipeline`.
- Eliminate the arbitrary whole-nucleus rejection filter (`c_area / n_area > 0.95`).
- Ensure precomputed distribution features are passed cleanly to `self.model_caller.predict_cycle_probabilities`.

---

### Component 3: Ultra-Fast Vectorized Barcode Decoder

#### [MODIFY] [popcrop/decoding/decoder.py](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/popcrop/decoding/decoder.py)
- Pre-compile the 15,766 library barcodes into a 2D integer NumPy lookup matrix: `codebook_matrix` of shape `(15766, 12)`.
- Replace the Python loop `for b_idx in range(...)` with vectorized NumPy array indexing:
  $$\text{scores} = \sum_{t=0}^{11} \text{log\_prob\_matrix}[t, \text{codebook\_matrix}[:, t]]$$
- Speeds up single-component decoding from 53 ms down to **0.3 ms** (>150x faster), allowing all candidate cells in the well to decode in under 2 seconds.

---

### Component 4: Production Run Execution & Summary

#### [MODIFY] [scripts/run_production_wellb2.py](file:///Users/koppikarps/Development/shamchal_view_popcrop_revision/scripts/run_production_wellb2.py)
- Run the full pipeline with the retrained classifier and intranuclear spot detector.
- Export updated `Plate1_B2_14_nuclei.parquet`, `Plate1_B2_14_components.parquet`, `gene_abundance_summary.csv`, and `run_summary_report.md`.
- Print a detailed Hamming distance / mismatch distribution report (0 mismatches, 1 mismatch, etc.).

---

## Estimated Runtime

| Step | Operation | Estimated Duration |
| :--- | :--- | :---: |
| 1 | Model Retraining & Calibration | ~15 seconds |
| 2 | TIFF Loading & DAPI Nuclear Segmentation | ~1.5 minutes |
| 3 | Intranuclear Spot Detection (5,378 nuclei) | ~1.5 - 2.5 minutes |
| 4 | Multi-Cycle Distribution Feature Extraction | ~1.5 minutes |
| 5 | Vectorized Barcode Decoding (all candidate spots) | ~3 seconds |
| 6 | Parquet & CSV Export + Diagnostic Overlays | ~45 seconds |
| **Total** | **Full End-to-End Pipeline Execution** | **~6 – 8 minutes** |

---

## Verification Plan

### Automated Tests
1. **Unit Test for Intranuclear Detector**: Run `pytest tests/test_blob_detection.py` to ensure local baseline detection correctly segments amplicons without whole-nucleus bleeding.
2. **Unit Test for Vectorized Decoder**: Run `pytest tests/test_decoding.py` to verify that the vectorized NumPy decoder produces identical scores, margins, and calls as the reference decoder.
3. **Model Validation**: Check test-set accuracy, ROC-AUC, and precision-recall of the retrained blob classifier on the holdout split.

### Production Run Verification
1. Inspect the total number of candidate spots detected (expecting ~1,200–2,000 real amplicons across the 5,378 nuclei).
2. Verify that 0-mismatch and 1-mismatch barcode calls represent the vast majority of confident calls.
3. Verify output files in `popcrop_output/wellb2_002_002/`:
   - `gene_abundance_summary.csv`
   - `Plate1_B2_14_nuclei.parquet`
   - `Plate1_B2_14_diagnostics.png`
