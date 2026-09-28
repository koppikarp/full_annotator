"""Writers for exporting analysis tables, masks, and diagnostic records."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import tifffile
import logging
import json
from .run_identity import IDENTITY_COLUMNS

logger = logging.getLogger(__name__)


def export_results(
    output_dir: str | Path,
    df_nuclei: pd.DataFrame,
    df_components: Optional[pd.DataFrame] = None,
    masks_dict: Optional[Dict[str, np.ndarray]] = None,
    export_format: str = "parquet",
    prefix: str = "results"
) -> Dict[str, str]:
    """
    Exports results tables and multidimensional masks to disk.
    
    Returns:
      dict mapping output type to file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. Export Nuclei Table
    if export_format == "parquet":
        nuclei_path = out_dir / f"{prefix}_nuclei.parquet"
        df_nuclei.to_parquet(nuclei_path, index=False, compression="zstd")
    else:
        nuclei_path = out_dir / f"{prefix}_nuclei.csv"
        df_nuclei.to_csv(nuclei_path, index=False)
    saved_paths["nuclei"] = str(nuclei_path)
    if "assigned_barcode" in df_nuclei:
        csv_path = out_dir / f"{prefix}_nucleus_barcode_calls.csv"
        write_barcode_csv(df_nuclei, csv_path)
        saved_paths["barcode_calls_csv"] = str(csv_path)
    logger.info(f"Saved nuclei table with {len(df_nuclei)} rows to {nuclei_path}")

    # 2. Export Components Table
    if df_components is not None and not df_components.empty:
        if export_format == "parquet":
            comp_path = out_dir / f"{prefix}_components.parquet"
            df_components.to_parquet(comp_path, index=False, compression="zstd")
        else:
            comp_path = out_dir / f"{prefix}_components.csv"
            df_components.to_csv(comp_path, index=False)
        saved_paths["components"] = str(comp_path)
        logger.info(f"Saved components table with {len(df_components)} rows to {comp_path}")

    # 3. Export Masks and Probability Maps
    if masks_dict:
        npz_path = out_dir / f"{prefix}_masks.npz"
        np.savez_compressed(npz_path, **masks_dict)
        saved_paths["masks_npz"] = str(npz_path)
        
        # Also export blob_mask and blob_prob_map as TIFF for easy Fiji/ImageJ visualization
        if "blob_mask" in masks_dict:
            tif_mask_path = out_dir / f"{prefix}_blob_mask.tiff"
            tifffile.imwrite(tif_mask_path, masks_dict["blob_mask"].astype(np.int32), compression="deflate")
            saved_paths["blob_mask_tif"] = str(tif_mask_path)
            
        if "blob_prob_map" in masks_dict:
            tif_prob_path = out_dir / f"{prefix}_blob_prob.tiff"
            tifffile.imwrite(tif_prob_path, (masks_dict["blob_prob_map"] * 65535).astype(np.uint16), compression="deflate")
            saved_paths["blob_prob_tif"] = str(tif_prob_path)

        logger.info(f"Saved masks and probability maps to {npz_path}")

    return saved_paths


def write_barcode_csv(df_nuclei, path):
    """Convenient nucleus-level view; the complete scientific tables remain parquet."""
    columns = [c for c in (
        *IDENTITY_COLUMNS, "nucleus_uid",
        "nucleus_id", "centroid_y", "centroid_x", "nucleus_area", "has_detectable_spot",
        "has_signal", "detected_cycles", "signal_cycles", "called_sequence", "called_library_barcode",
        "assigned_barcode", "assigned_gene", "best_barcode", "gene_candidates",
        "hamming_distance", "nearest_hamming_distance", "nearest_barcode_count",
        "nearest_barcodes", "nearest_genes", "score_margin", "posterior_confidence",
        "decoding_status", "qc_flags") if c in df_nuclei]
    frame = df_nuclei[columns].copy()
    for c in ("gene_candidates", "nearest_barcodes", "nearest_genes"):
        if c in frame:
            frame[c] = frame[c].map(lambda values: json.dumps(list(values)))
    frame.to_csv(path, index=False)
