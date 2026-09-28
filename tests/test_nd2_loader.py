"""Unit tests for the Nikon ND2 file loader and format dispatcher."""

from pathlib import Path
import numpy as np
import pytest

from popcrop.io.nd2_loader import load_raw_nd2
from popcrop.io.image_loader import load_raw_image


SAMPLE_ND2 = Path(
    "/Users/koppikarps/Development/popcell_260623/brieflow/tests/small_test_analysis/small_test_data/phenotype/real_images/P001_Pheno_20x_Wells-A1_Points-005__Channel_AF750,Cy3,GFP,DAPI.nd2"
)


@pytest.mark.skipif(not SAMPLE_ND2.exists(), reason="Sample ND2 file not found on local path")
def test_load_real_nd2_file():
    sbs_stack, dapi, meta = load_raw_nd2(SAMPLE_ND2, has_dapi_cycle=False)
    assert sbs_stack.ndim == 4
    # Expected shape: (1 cycle, 4 channels, 2400, 2400)
    assert sbs_stack.shape[0] == 1
    assert sbs_stack.shape[1] == 4
    assert sbs_stack.shape[2] == 2400
    assert sbs_stack.shape[3] == 2400
    assert meta["format"] == "nd2"
    assert "pixel_size_um" in meta["nd2_meta"]


@pytest.mark.skipif(not SAMPLE_ND2.exists(), reason="Sample ND2 file not found on local path")
def test_universal_image_loader_nd2_dispatch():
    sbs_stack, dapi, meta = load_raw_image(SAMPLE_ND2, has_dapi_cycle=False)
    assert sbs_stack.ndim == 4
    assert sbs_stack.shape[1] == 4
    assert meta["format"] == "nd2"
