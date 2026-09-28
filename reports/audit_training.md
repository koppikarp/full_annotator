# Annotation labels, provenance, and model validation

Historical blob labels judge whether each detector call is correct. True green presence is `green_detected == green_called_correctly`, and red follows the same rule. “Both correct” can therefore mean two correctly absent signals. `call_status != neither` is not a presence label. The adapter preserves original columns and derives per-channel presence, any-channel presence, and A/C/G/T. Missing detector states and contradictory correctness fields fail explicitly.

The 381 historical blob rows contain 172 true positives and 209 negatives. Stored `is_real_blob` disagrees with recovered presence in 154 rows. Six recovered positives have no detected masks, historically and after fixed-55 feature regeneration. Their old blob-only photometrics collapse to zeros; base fallback distributions retain raw nuclear information. Zero contours alone cannot establish absence of biological signal.

The 1,776 explicit base rows contain 16 N judgements, excluded from fitting and evaluation. The remaining 1,760 rows comprise A:417, C:304, G:794, T:245 from 1,534 nuclei. Eight explicit rows overlap blob-correctness rows; one conflicts (cell 3031, cycle 1: explicit G, recovered A). The production base model consequently uses explicit labels only. Recovered bases supply a separate validation set rather than silently mixing contradictory judgements.

Historical area divided by rounded occupied fraction estimates the historical sampling ROI area, not biological nuclear area. It gives median 2,375.19 px and an equivalent diameter of 54.99 px among nonzero rows. Fractions are rounded; circles may be clipped; zero-area rows cannot recover diameter. The claimed 193 areas above 300 px, 80 above 600 px, and maximum 1,280 px are verified. This does not repair mixed historical feature geometry: the spatial backfill independently hard-coded 90 px, and photometric backfill used app defaults.

`scripts/retrain_manual_parity.py` derives labels from immutable source detector states before regenerating all rows from raw TIFF pixels through the shared `analyze_nucleus_cycle`. The configurable sampling diameter is 55 px. Training uses the same cycle-specific DAPI anchoring, cycle-zero fallback, channel mapping, detector settings, background definitions, masks, and distributions as the updated UI and batch pipeline. Saved annotation coordinates are authoritative. Neighbor context comes from the complete saved profile whose image SHA256 matches the TIFF.

Derived Parquet tables retain all scalar measurements, histograms, representative intensity samples, source labels, and per-row analysis provenance. The base model consumes 43 scalar features and the blob model 32. Previously omitted quantiles, background statistics, channel areas, morphology and spatial features are restored. Histograms and representative samples remain available for inspection and future model development. Neither LUTs, illumination correction, global background subtraction, nor unmixing transform the raw measurements.

The compressed TIFF is decoded to a temporary disk mapping. Local crops and masks are processed and discarded per row, and the temporary mapping is removed at exit. Regeneration and training took approximately 10 seconds for 2,157 rows with two compute threads. Derived data plus production models occupy about 8 MB; the separate evaluation model adds approximately 4 MB. Before/after SHA256 checks confirm byte-identical source CSVs. Original model files under `models/` remain unchanged.

All cycles of each nucleus stay together in the holdout and the three inner probability-calibration folds. Groups include image, well, tile, and normalized cell identity; missing IDs use coordinates. Scaling happens inside each calibration fold. Split selection considers class coverage only, never predictions. Production models are refit on all usable labels after evaluation. A separate evaluation-only model retains the untouched holdout experiment.

With seed 42, explicit-base evaluation uses 335 cycles from 307 heldout nuclei and 1,425 fitting cycles from 1,227 nuclei. Accuracy is 94.93%, balanced accuracy 93.75%, macro F1 94.21%, and log loss 0.1733. Confusion rows in A/C/G/T order are `[76,1,5,0]`, `[0,44,2,0]`, `[2,1,157,1]`, `[0,3,2,41]`. There is no group overlap. Blob evaluation uses 77 heldout and 304 fitting nuclei. At threshold 0.70: 39 true negatives, 35 true positives, three false negatives, zero false positives; accuracy 96.10%, balanced accuracy 96.05%, ROC AUC 0.9845. These are same-image heldout estimates, not whole-barcode accuracy or evidence of generalization across acquisition conditions.

The explicit-base evaluation model also predicts 323 correctness-derived rows from unseen nuclei at 92.88% accuracy. This differently sampled set is reported separately. Annotation coverage is highly cycle-skewed: 1,139 source base rows are cycle 1; cycles 6, 7, 8, 10, 11 and 12 have nine rows each. Further independent multicycle annotations would support stronger estimates of read-level and cross-image performance.

`scripts/validate_fresh_centroids.py` applies the heldout model at fresh automated DAPI centers with fresh neighbor context. All 335 heldout rows map to 307 fresh nuclei within 27.5 px. There are no unmatched rows, duplicate manual-to-fresh assignments, or fresh-nucleus aliases between fit and test groups. Median center displacement is 0.0405 px and the 95th percentile 0.0615 px; eight rows exceed 10 px. Fresh centers yield 94.33% accuracy and 93.05% balanced accuracy versus 94.93% accuracy at saved coordinates. The 327 rows within 10 px score 94.80%. This tests coordinate and neighbor-context transfer independently of whole-read decoding.

Run from the project directory:

```sh
.venv/bin/python scripts/retrain_manual_parity.py --nuclear-diameter 55
.venv/bin/python scripts/validate_fresh_centroids.py --mask /path/to/nuclear-labels.npy
.venv/bin/python -m pytest tests/test_model_training.py tests/test_ground_truth_adapter.py
```

Machine-readable evidence is under `models/manual_parity/`: `training_audit.json`, both production metadata JSON files, holdout prediction CSVs, `evaluation_base_caller_metadata.json` with explicit fit/test groups, and `fresh_centroid_validation.json`. Generic legacy training scripts now default to `models/custom_training` and have corrected labels and grouped validation, but production raw-domain models should be regenerated with `retrain_manual_parity.py` instead of fitting historical mixed-geometry feature columns.
