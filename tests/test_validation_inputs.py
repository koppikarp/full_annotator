"""Validation can be reproduced from the source image or ordinary run masks."""
import numpy as np
import pytest
import tifffile

from scripts import validate_fresh_centroids as validation


@pytest.fixture
def image_views(monkeypatch):
    stack = np.zeros((12, 2, 8, 9), dtype=np.uint16)
    dapi = np.ones((8, 9), dtype=np.uint16)
    metadata = {"dapi_stack": np.zeros((12, 8, 9), dtype=np.uint16)}
    monkeypatch.setattr(validation, "load_raw_image", lambda path: (stack, dapi, metadata))
    return stack, dapi, metadata


def test_omitted_mask_uses_active_dapi_segmentation(monkeypatch, image_views):
    expected = np.ones((8, 9), dtype=np.int32)
    calls = []

    def segment(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(validation, "load_or_generate_nuclear_mask", segment)
    stack, dapi, metadata, mask = validation.load_validation_inputs("source.tif")
    assert calls[0]["mask_path"] is None
    assert calls[0]["dapi_image"] is image_views[1]
    assert stack is image_views[0] and dapi is image_views[1] and metadata is image_views[2]
    assert mask is expected


@pytest.mark.parametrize("suffix", [".npy", ".tif", ".tiff", ".npz"])
def test_loads_saved_nuclear_labels_without_segmenting(tmp_path, image_views, monkeypatch, suffix):
    expected = np.zeros((8, 9), dtype=np.int32)
    expected[2:6, 3:7] = 7
    path = tmp_path / ("nuclei" + suffix)
    if suffix == ".npy":
        np.save(path, expected)
    elif suffix == ".npz":
        np.savez_compressed(path, nuclear_mask=expected, blob_mask=np.zeros_like(expected))
    else:
        tifffile.imwrite(path, expected)

    def unexpected_segmentation(*args, **kwargs):
        raise AssertionError("An existing mask must not regenerate segmentation")

    monkeypatch.setattr("popcrop.io.masks.segment_dapi_classical", unexpected_segmentation)
    _, _, _, actual = validation.load_validation_inputs("source.tif", path)
    np.testing.assert_array_equal(actual, expected)


def test_rejects_nonproduction_archive(tmp_path, image_views):
    path = tmp_path / "wrong.npz"
    np.savez(path, blob_mask=np.zeros((8, 9), dtype=np.int32))
    with pytest.raises(ValueError, match="nuclear_mask"):
        validation.load_validation_inputs("source.tif", path)


def test_rejects_mask_from_different_image_shape(tmp_path, image_views):
    path = tmp_path / "wrong.npy"
    np.save(path, np.zeros((4, 5), dtype=np.int32))
    with pytest.raises(ValueError, match="spatial shape"):
        validation.load_validation_inputs("source.tif", path)
