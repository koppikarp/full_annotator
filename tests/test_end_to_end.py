"""End-to-end integration tests for OpsPipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from popcrop.config import PipelineConfig
from popcrop.pipeline import OpsPipeline
from popcrop.io import Codebook


def test_full_pipeline_end_to_end(tmp_path):
    H, W = 100, 100
    T, C = 4, 2

    # Create 3 nuclei:
    # Nucleus 1: Barcode "ACGT" (Green, Yellow, Dark, Red)
    # Nucleus 2: Barcode "TGCA" (Red, Dark, Yellow, Green)
    # Nucleus 3: Empty (uninfected / no blob)
    nuclear_mask = np.zeros((H, W), dtype=np.int32)
    y, x = np.ogrid[:H, :W]
    nuclear_mask[(y - 25)**2 + (x - 25)**2 <= 12**2] = 1
    nuclear_mask[(y - 75)**2 + (x - 25)**2 <= 12**2] = 2
    nuclear_mask[(y - 50)**2 + (x - 75)**2 <= 12**2] = 3

    dapi_image = (nuclear_mask > 0).astype(np.float32) * 5000.0

    sbs_stack = np.ones((T, C, H, W), dtype=np.float32) * 100.0
    np.random.seed(42)
    sbs_stack += np.random.normal(0, 3.0, (T, C, H, W))

    # Nucleus 1 signals: "ACGT"
    # t=0: A (Green)
    sbs_stack[0, 1, 20:30, 20:30] = 500.0
    # t=1: C (Yellow: Red + Green)
    sbs_stack[1, 0, 22:28, 22:28] = 400.0
    sbs_stack[1, 1, 20:30, 20:30] = 600.0
    # t=2: G (Dark: none)
    # t=3: T (Red)
    sbs_stack[3, 0, 20:30, 20:30] = 500.0

    # Nucleus 2 signals: "TGCA"
    # t=0: T (Red)
    sbs_stack[0, 0, 70:80, 20:30] = 500.0
    # t=1: G (Dark)
    # t=2: C (Yellow)
    sbs_stack[2, 0, 72:78, 22:28] = 400.0
    sbs_stack[2, 1, 70:80, 20:30] = 600.0
    # t=3: A (Green)
    sbs_stack[3, 1, 70:80, 20:30] = 500.0

    # Reference Codebook
    df_cb = pd.DataFrame({
        "barcode": ["ACGT", "TGCA", "AAAA", "TTTT"],
        "gene": ["GENE_1", "GENE_2", "GENE_3", "GENE_4"]
    })
    codebook = Codebook(df_cb)

    cfg = PipelineConfig(analysis_mode="legacy")
    cfg.preprocessing.registration_method = "none"
    cfg.blob_classifier.model_path = None
    cfg.color_calling.calibrated_color_model_path = None
    cfg.output.output_dir = str(tmp_path / "output")
    pipeline = OpsPipeline(config=cfg)

    df_nuclei, df_components, masks = pipeline.run_on_arrays(
        sbs_stack=sbs_stack,
        dapi_image=dapi_image,
        nuclear_mask=nuclear_mask,
        codebook=codebook
    )

    # Verify results
    assert len(df_nuclei) == 3

    # Nucleus 1 verification
    n1 = df_nuclei[df_nuclei["nucleus_id"] == 1].iloc[0]
    assert n1["p_real_blob"] > 0.70
    assert n1["best_barcode"] == "ACGT"
    assert n1["best_gene"] == "GENE_1"
    assert n1["decoding_status"] == "matched"
    assert n1["called_sequence"] == "ACGT"

    # Nucleus 2 verification
    n2 = df_nuclei[df_nuclei["nucleus_id"] == 2].iloc[0]
    assert n2["p_real_blob"] > 0.70
    assert n2["best_barcode"] == "TGCA"
    assert n2["best_gene"] == "GENE_2"
    assert n2["decoding_status"] == "matched"
    assert n2["called_sequence"] == "TGCA"

    # Nucleus 3 verification (empty)
    n3 = df_nuclei[df_nuclei["nucleus_id"] == 3].iloc[0]
    assert n3["p_real_blob"] < 0.30
    assert n3["decoding_status"] in ["no_blob", "rejected_artifact"]
    assert n3["best_barcode"] == "UNASSIGNED"
