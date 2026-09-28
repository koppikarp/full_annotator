# POPCROP: raw annotator-compatible nucleus analysis

The default automated path and the manual annotator now share `popcrop/annotation/analysis.py`. They measure the original red/green pixels, independently detect each channel in each cycle, and use the same 55 px circular sampling ROI with cycle-specific DAPI centering. The DAPI nuclear segmentation remains separate: it finds/counts nuclei, while the sampling circle defines the features used by the trained models.

The current assay uses A=green, T=red, C=both, G=neither. N annotations are excluded from training, and N-containing reads are excluded from library assignment. One barcode is reported per nucleus. `barcode_new` is the production library column following the assay owner's clarification; nine sequences in that column identify two possible genes, so they cannot produce unique gene assignments.

## Setup

Clone the private repository using an authenticated GitHub CLI, then create the validated Python 3.12 environment:

```sh
gh auth login
gh repo clone koppikarp/full_annotator
cd full_annotator
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.lock
```

Code, trained models, annotation ground truths, library and compact validation results are included in Git. The large reference TIFF is stored losslessly compressed in the private [reference-data-v1 release](https://github.com/koppikarp/full_annotator/releases/tag/reference-data-v1). Cloning does not download it. Restore it before reproducing the reference analysis or retraining:

```sh
python3 scripts/fetch_reference_image.py
```

The helper uses authenticated `gh` to download the release asset, verifies the archive and reconstructed TIFF against the recorded SHA-256 checksums, and writes `wellb2_002_002_RGB.tif` at the project root. It uses bounded memory, preserves existing files, and reuses an identical verified image. Allow about 4.3 GB of free space during download and restoration; the temporary compressed download is removed afterward. Python's standard library is sufficient for this step.

Alternatively, download `wellb2_002_002_RGB.tif.gz` from the release page in a browser and restore it without GitHub CLI:

```sh
python3 scripts/fetch_reference_image.py --archive /path/to/wellb2_002_002_RGB.tif.gz
```

The reference image is optional when analyzing your own compatible images with the included models.

## Run

From the project directory:

```sh
.venv/bin/python scripts/run_production_wellb2.py
```

The runner defaults to `wellb2_002_002_RGB.tif`, `data/library/v4.3_library_barcodes.csv`, column `barcode_new`, diameter 55, and results under `popcrop_output`. Each invocation creates a unique run directory and runs fresh DAPI segmentation. Existing outputs are preserved.

For another image of the same RGB cycle layout and acquisition settings:

```sh
.venv/bin/python scripts/run_production_wellb2.py \
  --image /absolute/path/wellb2_002_002_RGB.tif \
  --plate Plate1 --well B2 --tile 002_002 \
  --experiment-date 2026-09-28 --out /absolute/path/results
```

`--out` is a **parent directory** that can be shared by all images and parallel HPC jobs. Every run atomically reserves a new child directory. Every exported filename includes the image stem, plate/well/tile, optional experiment date, UTC run timestamp, and a random run suffix. Rerunning an image creates a separate result set. An example directory name is:

```text
wellb2_002_002_RGB__plate-Plate1__well-B2__tile-002_002__date-2026-09-28__20260928T143012123456Z_a1b2c3d4e5f6
```

Files inside share this prefix followed by suffixes such as `_nucleus_barcode_calls.csv` or `_cycles.parquet`.

`--experiment-date` (alias `--date`) is the optional acquisition/experiment date in `YYYY-MM-DD` format. It is never inferred from filesystem timestamps. The actual UTC run timestamp is always included separately. For new images, unknown plate/well identities remain `unspecified` and tile defaults to the image stem; supply explicit identifiers or a YAML configuration for experiment grouping. Full original filenames and paths are retained even when long or unusual names are shortened/sanitized in output paths.

For combining runs, every nucleus, component, cycle, barcode CSV and gene-abundance row includes `run_id`, `source_image`, `source_path`, `plate`, `well`, `tile`, `experiment_date`, `run_started_utc`, and `position_idx`. Nucleus-level and cycle rows also have `nucleus_uid`, combining run ID with local nucleus ID. Join by `nucleus_uid` (and `cycle` where appropriate); `nucleus_id` alone repeats between images. Select one run per source image/position when combining reruns, so the same field is not counted twice.

Each run has a `*_run_manifest.json` containing all exported paths. Aggregate only manifests with `status: complete`. Failed/interrupted runs retain `status: started`; completion is recorded only after all exports, diagnostics and runtime metadata finish. The terminal prints the actual run directory.

For other settings, use the generic CLI and a YAML configuration based on `configs/default_config.yaml`:

```sh
.venv/bin/python -m popcrop.cli run --image /absolute/path/image.tif --config configs/default_config.yaml --codebook data/library/v4.3_library_barcodes.csv --out /absolute/path/results
```

The tested image is a 16-bit RGB TIFF with 13 pages: phenotype/cycle 0, then 12 SBS cycles, each with red, green, blue/DAPI. Pixel values are not LUT-adjusted, offset-subtracted, flattened, unmixed, or globally background-subtracted. TIFF pages are memory mapped where possible. This file is an exported RGB stack; without acquisition provenance, the software cannot establish whether an earlier export applied any processing. Other acquisition/channel layouts must be verified before interpreting results. The new path rejects stacks with a channel count other than the expected red/green pair instead of silently choosing channels.

## Outputs

`*_nucleus_barcode_calls.csv` is a convenient flat view for inspection. `*_nuclei.parquet` has every segmented nucleus, its centroid and biological mask area, detector-positive cycles, signal-positive cycles, raw observed sequence, reverse complement, likelihood-selected reference, nearest reference(s), Hamming distances, gene candidates, assignment status, and QC flags. Use `assigned_barcode` and `assigned_gene` for accepted assignments. `best_barcode` is a ranking candidate and can be present even when assignment is rejected.

`*_cycles.parquet` has every nucleus/cycle, including detector-negative cycles. It preserves all scalar distribution measurements, 32-bin histograms, representative samples, per-channel detection statistics, trained probabilities, sampling settings and coordinates, and exact local channel masks as flattened pixel indices. JSON-valued columns retain nested data; crop dimensions and bounds in `analysis_provenance` reconstruct the masks. Full original pixels remain in the source TIFF. The trained base model uses all scalar shared features; histograms and samples remain available for auditing and future models.

`*_components.parquet` is a compatibility table containing one candidate read per nucleus with a detector proposal or classified signal. It does not count individual physical amplicons. `*_masks.npz` preserves the exact full-resolution nuclear and diagnostic union masks and floating-point score map. TIFF masks are compressed for Fiji/ImageJ; the TIFF probability display is quantized to 16 bits, while the NPZ retains the float map. Overlapping union footprints use the first nucleus label, and the display score uses the maximum across overlapping nuclei; exact scientific masks are in the cycle table.

`*_run_summary.json` records counts, Hamming distributions, run identity, configuration, input metadata, model/library hashes, runtime and peak process memory. `*_run_summary_report.md`, `*_gene_abundance_summary.csv`, and the diagnostic image provide compact views. Gene abundance includes only uniquely assigned genes, excluding ambiguous duplicate-library references.

A detector proposal is not proof of real signal. `has_signal` means at least one cycle has base-model P(A)+P(C)+P(T) at least 0.70. The corrected blob classifier also scores every cycle and remains available in the cycle output; it does not veto a nucleus before base calling. Its legacy feature vector collapses when no contour is detected, which makes that veto unsafe. A negative signal result does not establish that a cell is untransduced.

Library assignment requires a valid complete read, selected-reference Hamming distance at most 2, log-likelihood margin at least 0.8, and library-relative likelihood support at least 0.60. The nearest Hamming reference and selected likelihood reference can differ. Relative likelihood support is not a validated probability of a correct gene assignment. These thresholds are explicit configuration values, not selected to force high match rates.

## Rebuild models and verify

```sh
.venv/bin/python scripts/retrain_manual_parity.py
.venv/bin/python scripts/validate_manual_reads.py
.venv/bin/python scripts/validate_fresh_centroids.py
.venv/bin/python -m pytest -q
```

Source annotation CSVs and original models are never rewritten by retraining. Correctness labels are converted to presence using the original detector bits before new features are extracted. Models, derived features, grouped holdout predictions, and metadata are saved in `models/manual_parity`. All cycles belonging to a nucleus remain together in both calibration and evaluation. The deployed models are refitted on the annotated data after heldout evaluation; deployed-model agreement on those same annotations is not an independent test.

The diameter is configurable through `--diameter`. A model trained at a different diameter or detector setting is rejected to prevent hidden training/inference mismatch. Retrain with the corresponding diameter using `scripts/retrain_manual_parity.py --nuclear-diameter ...`, then run with that diameter. Training needs verified image/annotation coordinates, not a guessed radius inferred from rounded historic fractions. For additional acquisitions with different photometry or optics, representative annotations and independent validation are needed before claiming the same accuracy.

Python 3.12 and the exact validated scientific dependencies are recorded in `requirements-analysis.lock`; use the setup commands above on a new machine.

The manual annotator is launched with `.venv/bin/python run_app.py`. Future annotations save exact raw-analysis provenance and derive true channel presence from the four correctness buttons. LUTs remain display-only.

The old automated implementation remains available only through `analysis_mode: legacy` for controlled comparisons. The current default configuration uses the new raw shared path. The obsolete application folder, ZIP, legacy model bundles and source-overwriting backfill scripts were removed during cleanup; run the top-level application and current scripts.

## Reproducible workspace

There is one standalone raw reference image, `wellb2_002_002_RGB.tif`. Restore it from the private release using the setup instructions above. `data/reference_image.json` records the release identity and exact archive/raw sizes and checksums. The restored image is a regular file, so copying the complete folder to an HPC does not depend on a removed application directory or an external symlink. Git ignores this large local TIFF. The top-level annotation CSVs and coordinate/LUT profiles remain byte-identical. The profiles also mention older images that were already absent from the supplied workspace.

`data/library/` contains the reference library and unique experimental design originals. `data/annotation_history/` preserves the two distinct annotation snapshots extracted from the old ZIP. Those snapshots are historical evidence and must not be automatically merged into current training data.

`models/manual_parity/` retains the deployed models, grouped evaluation model, metadata, regenerated features and holdout predictions required to repeat validation without retraining. `scripts/retrain_manual_parity.py` rebuilds these artifacts from the original image and current annotations. `scripts/validate_fresh_centroids.py` generates fresh DAPI segmentation by default, or accepts `--mask` with a `.npy`, `.tif`, `.tiff` or production `.npz` mask.

`reports/reference_run/` retains the compact validated nucleus table and run summaries as a comparison baseline. `reports/legacy_reference/` retains only the compressed masks and tables needed to reproduce the historical failure audit with `scripts/audit_legacy_detection.py`. Large generated image/cycle exports, duplicate TIFFs, scratch plots/test models, caches and the machine-specific virtual environment were removed. The small audit documents and original proposed plan retain the reasoning behind the changes; their historical paths describe the pre-cleanup workspace.

For a fresh machine, create the environment and restore the reference image above, run the tests, then run `scripts/run_production_wellb2.py` to regenerate the full reference analysis. Retraining is optional when using the retained models. To reconstruct the models and their validation too, run the three training/validation commands above. Output directories are created automatically.

`reports/workspace_cleanup.json` records moved and removed artifacts. `reports/reproducibility_manifest.json` records SHA-256 checksums for the retained scientific inputs, model artifacts, code and validation references.

See `reports/investigation.md`, `audit_manual.md`, `reports/audit_training.md`, and `reports/audit_decoding.md` for the detailed investigation and remaining validation limits.
