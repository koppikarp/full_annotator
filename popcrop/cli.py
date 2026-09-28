"""Command-line interface (CLI) for popcrop."""

from __future__ import annotations
import argparse
import sys
import logging
from pathlib import Path

from popcrop.config import PipelineConfig
from popcrop.pipeline import OpsPipeline
from popcrop.calibration.ground_truth_adapter import GroundTruthAdapter
from popcrop.calibration.calibrator import PipelineCalibrator
from popcrop.blob_classifier.classifier import BlobClassifier
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("popcrop.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="popcrop",
        description="POPCROP: Production Pipeline for Extended Fluorescent Signal OPS Base-Calling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. RUN COMMAND
    p_run = subparsers.add_parser("run", help="Run end-to-end processing on TIFF or Nikon ND2 image")
    p_run.add_argument(
        "--image", "--tiff", "--nd2",
        dest="image",
        required=True,
        help="Path to multicycle, multichannel linear TIFF or Nikon ND2 file(s)"
    )
    p_run.add_argument("--config", default=None, help="Path to YAML configuration file")
    p_run.add_argument("--mask", default=None, help="Path to precomputed nuclear mask (.tif or .npy)")
    p_run.add_argument("--codebook", default=None, help="Path to barcode codebook (.csv or .tsv)")
    p_run.add_argument("--barcode-column", default=None, help="Explicit reference sequence column (this assay uses barcode_new)")
    p_run.add_argument("--position", type=int, default=0, help="Tile/position index if multi-position ND2")
    p_run.add_argument("--out", default=None, help="Results parent directory; every run creates a unique child")

    # 2. CALIBRATE COMMAND
    p_calib = subparsers.add_parser("calibrate", help="Calibrate unmixing and channel ratios from ground truth")
    p_calib.add_argument("--ground-truth", required=True, help="Path to annotated point-values or control TSV/CSV")
    p_calib.add_argument("--out", default="calibration.json", help="Path to save calibrated parameters JSON")

    # 3. TRAIN COMMAND
    p_train = subparsers.add_parser("train", help="Train calibrated P(real_blob) classifier")
    p_train.add_argument("--features", required=True, help="Path to component features CSV/Parquet")
    p_train.add_argument("--labels", required=True, help="Path to binary label column / file")
    p_train.add_argument("--out", default="blob_classifier.joblib", help="Output joblib model path")

    # 4. EVALUATE COMMAND
    p_eval = subparsers.add_parser("evaluate", help="Evaluate barcode calling accuracy against known truth")
    p_eval.add_argument("--predictions", required=True, help="Path to nuclei results CSV/Parquet")
    p_eval.add_argument("--ground-truth", required=True, help="Path to true barcode assignments")

    # 5. ANNOTATE COMMAND
    p_anno = subparsers.add_parser("annotate", help="Launch interactive spot-aware manual base caller and signal validator web app")
    p_anno.add_argument("--image", "--tiff", dest="image", default=None, help="Path to multichannel TIFF or ND2 image")
    p_anno.add_argument("--tsv", default=None, help="Path to cell coordinates TSV/CSV file")
    p_anno.add_argument("--port", type=int, default=5000, help="Port to run web server on")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        logger.info(f"Running pipeline on {args.image}")
        config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
        if args.codebook:
            config.decoding.codebook_path = args.codebook
        if args.barcode_column:
            config.decoding.barcode_column = args.barcode_column
        pipeline = OpsPipeline(config=config)
        pipeline.run_on_files(
            image_path=args.image,
            mask_path=args.mask,
            codebook_path=args.codebook,
            output_dir=args.out,
            position_idx=args.position
        )
        logger.info("Pipeline execution finished successfully.")

    elif args.command == "calibrate":
        logger.info(f"Calibrating from {args.ground_truth}")
        adapter = GroundTruthAdapter().load_point_values(args.ground_truth)
        calibrator = PipelineCalibrator()
        calibrator.calibrate_from_ground_truth(adapter)
        calibrator.save(args.out)
        logger.info(f"Saved calibration to {args.out}")

    elif args.command == "train":
        logger.info(f"Training classifier on {args.features}")
        df_feat = pd.read_parquet(args.features) if args.features.endswith(".parquet") else pd.read_csv(args.features)
        y = pd.read_csv(args.labels).values.ravel()
        clf = BlobClassifier()
        clf.fit(df_feat, y)
        clf.save(args.out)
        logger.info(f"Saved trained classifier to {args.out}")

    elif args.command == "evaluate":
        logger.info("Evaluating predictions...")
        df_pred = pd.read_parquet(args.predictions) if args.predictions.endswith(".parquet") else pd.read_csv(args.predictions)
        df_true = pd.read_csv(args.ground_truth)
        merged = pd.merge(df_pred, df_true, on="nucleus_id", suffixes=("_pred", "_true"))
        if "best_barcode_true" in merged.columns and "best_barcode_pred" in merged.columns:
            exact = (merged["best_barcode_pred"] == merged["best_barcode_true"]).mean()
            logger.info(f"Exact Barcode Assignment Accuracy: {exact * 100:.2f}% (N={len(merged)})")

    elif args.command == "annotate":
        from popcrop.annotation import app as app_module
        if args.image:
            app_module.CUSTOM_TIFF_PATH = str(Path(args.image).resolve())
        if args.tsv:
            app_module.ACTIVE_TSV_PATH = str(Path(args.tsv).resolve())
        logger.info(f"Starting POPCROP Manual Base Caller on http://localhost:{args.port}")
        app_module.app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
