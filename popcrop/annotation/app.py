"""POPCROP Spot/Blob-Aware Manual Base Caller & Signal Validator Web Application.

Provides two distinct ground truth annotation workflows:
  1. Base Caller (/): Annotates base calls within segmented blobs using multi-tier
     distributional features (Tier 1 quantiles, Trimmed Mean, Tier 2 empirical distributions).
  2. Blob Signal Validator (/blob_validator): Annotates whether candidate blobs correspond
     to authentic signal or background noise/artifacts (True/False) with 10+ object features.
"""

from __future__ import annotations
import os
import io
import re
import glob
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import base64
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, send_file
from scipy import ndimage as ndi

from popcrop.annotation.analysis import analyze_nucleus_cycle, DEFAULT_DETECTOR_PARAMS

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Base paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
APP_DIR = MODULE_DIR
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
DATA_DIR = os.path.join(APP_DIR, "sample_data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Ground truth output files (Two separate files as requested)
BASE_GROUND_TRUTH_PATH = os.path.join(ROOT_DIR, "annotated_base_ground_truth.csv")
BLOB_GROUND_TRUTH_PATH = os.path.join(ROOT_DIR, "annotated_blob_ground_truth.csv")

# Legacy / sample data paths
CUSTOM_TIFF_PATH: Optional[str] = None
ACTIVE_TSV_PATH: Optional[str] = None
DEFAULT_TILE = 14
DEFAULT_CYCLE = 1
DEFAULT_NUCLEI_DIAMETER = 55.0
DEFAULT_SEQUENCE_CYCLE_COUNT = 12

HAS_CYCLE_ZERO_REFERENCE = True
CYCLE_ZERO_PAGE_COUNT = 2
CYCLE_ZERO_CHANNEL_PAGE_MAP = {3: 0, 0: 1}  # 647 / far-red = 0, DAPI = 1

# Caches
IMAGE_CACHE: Dict[str, np.ndarray] = {}
TSV_CACHE: Optional[Dict[str, Any]] = None
TIFF_FINGERPRINT_CACHE: Dict[str, str] = {}
ACTIVE_TIFF_FINGERPRINT: Optional[str] = None
ACTIVE_TIFF_FINGERPRINT_PATH: Optional[str] = None
NUC_CROP_CACHE: Dict[str, np.ndarray] = {}
CROP_ANALYSIS_CACHE: Dict[Any, Dict[str, Any]] = {}
CELLPOSE_SAM_MODEL: Any = None
CELLPOSE_SAM_LOCK = threading.Lock()

# Profiles (check ROOT_DIR / annotation_app_1.8 as fallback)
LUT_PROFILE_PATH = os.path.join(ROOT_DIR, "saved_lut_profiles.json")
if not os.path.exists(LUT_PROFILE_PATH) and os.path.exists(os.path.join(ROOT_DIR, "annotation_app_1.8", "saved_lut_profiles.json")):
    LUT_PROFILE_PATH = os.path.join(ROOT_DIR, "annotation_app_1.8", "saved_lut_profiles.json")

LUT_PROFILE_LOCK = threading.RLock()
COORD_PROFILE_PATH = os.path.join(ROOT_DIR, "saved_coordinate_profiles.json")
if not os.path.exists(COORD_PROFILE_PATH) and os.path.exists(os.path.join(ROOT_DIR, "annotation_app_1.8", "saved_coordinate_profiles.json")):
    COORD_PROFILE_PATH = os.path.join(ROOT_DIR, "annotation_app_1.8", "saved_coordinate_profiles.json")
COORD_PROFILE_LOCK = threading.RLock()
ANNOTATION_LOCK = threading.RLock()


# --- Content-Addressed Profile Helpers ---

def _file_sha256(file_path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = os.path.abspath(file_path)
    if path in TIFF_FINGERPRINT_CACHE:
        return TIFF_FINGERPRINT_CACHE[path]
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    digest = hasher.hexdigest()
    TIFF_FINGERPRINT_CACHE[path] = digest
    return digest


def get_current_tiff_fingerprint() -> Optional[str]:
    global ACTIVE_TIFF_FINGERPRINT, ACTIVE_TIFF_FINGERPRINT_PATH
    path = get_tiff_path()
    if not path or not os.path.exists(path):
        return None
    abs_path = os.path.abspath(path)
    if not ACTIVE_TIFF_FINGERPRINT or ACTIVE_TIFF_FINGERPRINT_PATH != abs_path:
        ACTIVE_TIFF_FINGERPRINT = _file_sha256(abs_path)
        ACTIVE_TIFF_FINGERPRINT_PATH = abs_path
    return ACTIVE_TIFF_FINGERPRINT


def _read_lut_database() -> Dict[str, Any]:
    with LUT_PROFILE_LOCK:
        if not os.path.exists(LUT_PROFILE_PATH):
            return {"version": 1, "images": {}}
        try:
            with open(LUT_PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 1, "images": {}}


def _write_lut_database(db: Dict[str, Any]) -> None:
    with LUT_PROFILE_LOCK:
        tmp_path = LUT_PROFILE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, LUT_PROFILE_PATH)


def _read_coord_database() -> Dict[str, Any]:
    with COORD_PROFILE_LOCK:
        if not os.path.exists(COORD_PROFILE_PATH):
            return {"version": 2, "images": {}}
        try:
            with open(COORD_PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 2, "images": {}}


def _write_coord_database(db: Dict[str, Any]) -> None:
    with COORD_PROFILE_LOCK:
        tmp_path = COORD_PROFILE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, COORD_PROFILE_PATH)


# --- TIFF Inspection & Frame Extraction ---

def describe_tiff_array(img: Any) -> Dict[str, Any]:
    if img is None:
        return {"layout": "none", "shape": None, "has_cycle_zero": False, "sequencing_cycles": 0}
    arr = np.squeeze(np.asarray(img))
    shape = tuple(int(v) for v in arr.shape)
    info = {"layout": "unknown", "shape": list(shape), "has_cycle_zero": False, "sequencing_cycles": 0}

    # RGB composite stack (frames, Y, X, 3/4)
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        frames = int(arr.shape[0])
        has_c0 = frames > 1
        info.update({
            "layout": "rgb_composite_stack",
            "has_cycle_zero": has_c0,
            "sequencing_cycles": max(1, frames - 1 if has_c0 else frames),
            "height": int(arr.shape[1]), "width": int(arr.shape[2])
        })
        return info

    # Raw grayscale page stack (pages, Y, X)
    if arr.ndim == 3:
        pages = int(arr.shape[0])
        has_c0 = pages >= 2 and (pages - 2) % 4 == 0
        cycles = (pages - 2) // 4 if has_c0 else (pages // 4 if pages % 4 == 0 else max(1, pages))
        info.update({
            "layout": "raw_pages",
            "has_cycle_zero": has_c0,
            "sequencing_cycles": int(cycles),
            "height": int(arr.shape[1]), "width": int(arr.shape[2])
        })
        return info

    # Raw 4D array (cycles, channels, Y, X)
    if arr.ndim == 4:
        frames = int(arr.shape[0])
        has_c0 = frames > 1
        info.update({
            "layout": "raw_cycle_channel",
            "has_cycle_zero": has_c0,
            "sequencing_cycles": max(1, frames - 1 if has_c0 else frames),
            "height": int(arr.shape[-2]), "width": int(arr.shape[-1])
        })
        return info

    return info


def extract_channel_frame(img: Any, cycle: int = 1, channel_idx: int = 0) -> np.ndarray:
    """
    Returns a 2D plane for requested channel_idx (0=DAPI, 1=Green, 2=Red, 3=Far-red/647).
    Supports RGB composite stacks, 4D stacks, and raw grayscale pages.
    """
    if img is None:
        return np.zeros((2048, 2048), dtype=np.uint16)
    arr = np.squeeze(np.asarray(img))
    meta = describe_tiff_array(arr)
    has_c0 = bool(meta.get("has_cycle_zero", False))

    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        # RGB composite page
        frame_idx = int(cycle) if has_c0 else int(cycle) - 1
        frame_idx = max(0, min(arr.shape[0] - 1, frame_idx))
        page = arr[frame_idx]
        if int(cycle) == 0:
            if channel_idx == 0:
                return page[:, :, 2]  # DAPI Blue
            if channel_idx == 3:
                return page[:, :, 0]  # Far-red 647 (Red channel in C0)
            return np.zeros(page.shape[:2], dtype=page.dtype)
        if channel_idx == 0:
            return page[:, :, 2]  # Blue DAPI
        if channel_idx == 1:
            return page[:, :, 1]  # Green GFP
        if channel_idx in (2, 3):
            return page[:, :, 0]  # Red Cy3
        return page[:, :, 0]

    if arr.ndim == 4:
        # (cycles, channels, Y, X)
        frame_idx = int(cycle) if has_c0 else int(cycle) - 1
        frame_idx = max(0, min(arr.shape[0] - 1, frame_idx))
        ch_idx = max(0, min(arr.shape[1] - 1, channel_idx))
        return arr[frame_idx, ch_idx, :, :]

    if arr.ndim == 3:
        pages = int(arr.shape[0])
        if cycle == 0:
            p_idx = CYCLE_ZERO_CHANNEL_PAGE_MAP.get(channel_idx, 0)
            return arr[max(0, min(pages - 1, p_idx)), :, :]
        offset = 2 if has_c0 else 0
        p_idx = offset + (int(cycle) - 1) * 4 + int(channel_idx)
        return arr[max(0, min(pages - 1, p_idx)), :, :]

    if arr.ndim == 2:
        return arr
    return np.zeros((2048, 2048), dtype=np.uint16)


def get_tiff_path(tile: Optional[int] = None, cycle: int = 1) -> Optional[str]:
    if CUSTOM_TIFF_PATH and os.path.exists(CUSTOM_TIFF_PATH):
        return CUSTOM_TIFF_PATH

    candidates = [
        os.path.join(ROOT_DIR, "wellb2_002_002_RGB.tif"),
        os.path.join(ROOT_DIR, "annotation_app_2.0", "wellb2_002_002_RGB.tif"),
        os.path.join(ROOT_DIR, "Test_RGB.tif"),
        os.path.join(MODULE_DIR, "sample_data", "13 cycles flexible LUT.tif"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    all_tiffs = glob.glob(os.path.join(ROOT_DIR, "*.tif*")) + glob.glob(os.path.join(DATA_DIR, "*.tif*")) + glob.glob(os.path.join(UPLOAD_DIR, "*.tif*"))
    return all_tiffs[0] if all_tiffs else None


def load_tiff_image(tiff_path: Optional[str]) -> Optional[np.ndarray]:
    if not tiff_path or not os.path.exists(tiff_path):
        return None
    if tiff_path not in IMAGE_CACHE:
        try:
            try:
                IMAGE_CACHE[tiff_path] = tifffile.memmap(tiff_path, mode="r")
            except (ValueError, OSError):
                IMAGE_CACHE[tiff_path] = tifffile.imread(tiff_path)
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            return None
    return IMAGE_CACHE[tiff_path]


# --- Cell Coordinates & DAPI Detection ---

def detect_nuclei_simple_dapi(dapi_plane: np.ndarray, expected_diameter: float = 90.0) -> List[Dict[str, Any]]:
    """Fast classical nucleus detector for generating cell coordinates from DAPI."""
    from skimage.filters import gaussian, threshold_otsu
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    dapi = gaussian(dapi_plane.astype(np.float32), sigma=2.0)
    try:
        th = threshold_otsu(dapi)
    except ValueError:
        th = float(np.mean(dapi))
    binary = dapi > th

    dist = ndi.distance_transform_edt(binary)
    min_dist = max(8, int(expected_diameter * 0.4))
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary)
    if len(coords) == 0:
        return []

    markers = np.zeros(dist.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labeled = watershed(-dist, markers, mask=binary)

    from skimage.measure import regionprops
    props = regionprops(labeled)
    cells = []
    for idx, p in enumerate(props, start=1):
        if p.area >= 50:
            cells.append({
                "cell": idx,
                "read": idx,
                "tile": 14,
                "well": "B2",
                "cycle": 1,
                "i": round(float(p.centroid[0]), 1),
                "j": round(float(p.centroid[1]), 1),
                "area": int(p.area)
            })
    return cells



def load_cell_coordinates(tile: Optional[int] = None) -> pd.DataFrame:
    """Loads cell coordinates from profile, TSV, or auto-detects from DAPI."""
    global TSV_CACHE
    fp = get_current_tiff_fingerprint()

    # 1. Try saved coordinate profile
    if fp:
        db = _read_coord_database()
        if fp in db.get("images", {}):
            data = db["images"][fp].get("cells", [])
            if data:
                return pd.DataFrame(data)

    # 2. Try active or default TSV
    tsv_candidates = [
        ACTIVE_TSV_PATH,
        os.path.join(os.path.dirname(MODULE_DIR), "annotation_app_1.8", "sample_data", "P-1_W-B2_T-14__bases.tsv"),
        os.path.join(DATA_DIR, "P-1_W-B2_T-14__bases.tsv"),
    ]
    for p in tsv_candidates:
        if p and os.path.exists(p):
            sep = "\t" if p.endswith(".tsv") else ","
            df = pd.read_csv(p, sep=sep)
            if "i" in df.columns and "j" in df.columns:
                if "cell" not in df.columns:
                    df["cell"] = np.arange(1, len(df) + 1)
                return df

    # 3. Auto-detect from DAPI plane
    tiff_path = get_tiff_path()
    img = load_tiff_image(tiff_path)
    if img is not None:
        meta = describe_tiff_array(img)
        dapi_cycle = 1 if meta.get("has_cycle_zero", False) and meta.get("sequencing_cycles", 0) >= 1 else 0
        dapi = extract_channel_frame(img, cycle=dapi_cycle, channel_idx=0)
        detected = detect_nuclei_simple_dapi(dapi, expected_diameter=DEFAULT_NUCLEI_DIAMETER)
        if detected:
            df_det = pd.DataFrame(detected)
            if fp:
                db = _read_coord_database()
                db.setdefault("images", {})[fp] = {"cells": detected, "filename": os.path.basename(tiff_path or "")}
                _write_coord_database(db)
            return df_det

    # Fallback dummy single coordinate
    return pd.DataFrame([{"cell": 1, "read": 1, "tile": 14, "well": "B2", "cycle": 1, "i": 500.0, "j": 500.0}])


# --- Cellpose-SAM (CPSAM) & DAPI Segmentation ---

def get_cellpose_sam_model():
    """Lazily loads the Cellpose-SAM (CPSAM) model singleton with PyTorch MPS or CPU acceleration."""
    global CELLPOSE_SAM_MODEL
    if CELLPOSE_SAM_MODEL is not None:
        return CELLPOSE_SAM_MODEL
    with CELLPOSE_SAM_LOCK:
        if CELLPOSE_SAM_MODEL is None:
            try:
                import torch
                from cellpose import models
                cpsam_path = os.path.expanduser("~/.cellpose/models/cpsam")
                use_gpu = torch.backends.mps.is_available()
                if os.path.exists(cpsam_path):
                    CELLPOSE_SAM_MODEL = models.CellposeModel(gpu=use_gpu, pretrained_model=cpsam_path)
                else:
                    CELLPOSE_SAM_MODEL = models.CellposeModel(gpu=use_gpu, model_type="cpsam")
            except Exception as e:
                print(f"Warning: Failed to load Cellpose-SAM model: {e}")
                CELLPOSE_SAM_MODEL = None
    return CELLPOSE_SAM_MODEL


def segment_nucleus_crop_watershed(
    dapi_crop: np.ndarray,
    cy_rel: int,
    cx_rel: int,
    expected_radius: float = 22.5
) -> np.ndarray:
    """Classical adaptive Otsu-watershed fallback for nuclear segmentation in local DAPI crops."""
    from skimage.filters import gaussian, threshold_otsu
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects

    H, W = dapi_crop.shape
    cy_rel = int(max(0, min(H - 1, round(cy_rel))))
    cx_rel = int(max(0, min(W - 1, round(cx_rel))))

    if dapi_crop is None or dapi_crop.size == 0 or dapi_crop.max() <= dapi_crop.min():
        y_g, x_g = np.ogrid[:H, :W]
        return ((y_g - cy_rel) ** 2 + (x_g - cx_rel) ** 2) <= (expected_radius ** 2)

    dapi_f = dapi_crop.astype(np.float32)
    smoothed = gaussian(dapi_f, sigma=2.0, preserve_range=True)

    try:
        th = float(threshold_otsu(smoothed))
    except Exception:
        th = float(np.mean(smoothed))

    min_val = float(smoothed.min())
    binary = smoothed > max(min_val + 1.0, min_val + 0.75 * (th - min_val))

    c_val = float(smoothed[cy_rel, cx_rel])
    if not binary[cy_rel, cx_rel] and c_val > (min_val + 0.20 * (th - min_val)):
        alt_th = min_val + 0.20 * (c_val - min_val)
        binary = smoothed > alt_th

    binary = remove_small_objects(binary, min_size=30)
    binary = ndi.binary_fill_holes(binary)

    if not np.any(binary):
        y_g, x_g = np.ogrid[:H, :W]
        return ((y_g - cy_rel) ** 2 + (x_g - cx_rel) ** 2) <= (expected_radius ** 2)

    dist = ndi.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=10, labels=binary)

    if len(coords) == 0:
        labeled, _ = ndi.label(binary)
        lbl = labeled[cy_rel, cx_rel]
        if lbl > 0:
            return ndi.binary_fill_holes(labeled == lbl)
        y_g, x_g = np.ogrid[:H, :W]
        return ((y_g - cy_rel) ** 2 + (x_g - cx_rel) ** 2) <= (expected_radius ** 2)

    markers = np.zeros(dist.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labeled = watershed(-dist, markers, mask=binary)

    target_lbl = labeled[cy_rel, cx_rel]
    if target_lbl == 0:
        dists = [np.hypot(r - cy_rel, c - cx_rel) for r, c in coords]
        target_lbl = int(np.argmin(dists)) + 1

    mask = (labeled == target_lbl)
    return ndi.binary_fill_holes(mask)


def segment_nucleus_cellpose_sam(
    dapi_full: np.ndarray,
    y_c: int,
    x_c: int,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    expected_diameter: float = 90.0
) -> np.ndarray:
    """Segments the target nucleus using Cellpose-SAM (CPSAM) on a contextual patch with watershed fallback."""
    dapi_crop = dapi_full[y_min:y_max, x_min:x_max]
    H_c, W_c = dapi_crop.shape
    cy_rel = y_c - y_min
    cx_rel = x_c - x_min

    model = get_cellpose_sam_model()
    if model is not None:
        try:
            H_full, W_full = dapi_full.shape
            patch_r = max(128, int(round(max(H_c, W_c) / 2.0)))
            py_min, py_max = max(0, y_c - patch_r), min(H_full, y_c + patch_r)
            px_min, px_max = max(0, x_c - patch_r), min(W_full, x_c + patch_r)
            patch = dapi_full[py_min:py_max, px_min:px_max]
            p_cy = y_c - py_min
            p_cx = x_c - px_min

            masks, flows, styles = model.eval(
                patch,
                diameter=expected_diameter
            )
            target_lbl = int(masks[p_cy, p_cx])
            if target_lbl == 0:
                y_lo, y_hi = max(0, p_cy - 15), min(patch.shape[0], p_cy + 16)
                x_lo, x_hi = max(0, p_cx - 15), min(patch.shape[1], p_cx + 16)
                sub = masks[y_lo:y_hi, x_lo:x_hi]
                nonzeros = sub[sub > 0]
                if len(nonzeros) > 0:
                    vals, counts = np.unique(nonzeros, return_counts=True)
                    target_lbl = int(vals[np.argmax(counts)])

            if target_lbl > 0:
                patch_mask = (masks == target_lbl)
                crop_mask = patch_mask[y_min - py_min : y_max - py_min, x_min - px_min : x_max - px_min]
                if np.sum(crop_mask) >= 50:
                    return ndi.binary_fill_holes(crop_mask)
        except Exception as err:
            print(f"Warning: Cellpose-SAM eval exception, falling back to watershed: {err}")

    return segment_nucleus_crop_watershed(dapi_crop, cy_rel, cx_rel, expected_radius=expected_diameter / 2.0)


def get_cell_nuclear_mask(
    dapi_full: np.ndarray,
    y_c: int,
    x_c: int,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    cell_id: int,
    fingerprint: Optional[str] = None,
    expected_diameter: float = 90.0
) -> np.ndarray:
    """Returns the cached or newly segmented organic nucleus mask for the given cell crop."""
    cache_key = f"{fingerprint or 'default'}_{cell_id}_{int(round(expected_diameter))}"
    if cache_key in NUC_CROP_CACHE:
        return NUC_CROP_CACHE[cache_key]

    nuc_mask = segment_nucleus_cellpose_sam(
        dapi_full=dapi_full,
        y_c=y_c,
        x_c=x_c,
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        expected_diameter=expected_diameter
    )
    NUC_CROP_CACHE[cache_key] = nuc_mask
    return nuc_mask


# --- Normalization & LUTs ---

def normalize_channel(plane: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if plane is None or plane.size == 0:
        return np.zeros((64, 64), dtype=np.uint8)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((plane.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def apply_lut(norm_uint8: np.ndarray, lut_name: str = "green", display_mode: str = "pseudocolor") -> np.ndarray:
    h, w = norm_uint8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if display_mode == "greyscale" or lut_name in ["gray", "grayscale"]:
        rgb[:, :, 0] = norm_uint8
        rgb[:, :, 1] = norm_uint8
        rgb[:, :, 2] = norm_uint8
        return rgb

    if lut_name in ["green", "default"]:
        rgb[:, :, 1] = norm_uint8
    elif lut_name == "red":
        rgb[:, :, 0] = norm_uint8
    elif lut_name in ["dapi", "blue"]:
        rgb[:, :, 2] = norm_uint8
    elif lut_name in ["farred", "magenta"]:
        rgb[:, :, 0] = norm_uint8
        rgb[:, :, 2] = norm_uint8
    elif lut_name == "cyan":
        rgb[:, :, 1] = norm_uint8
        rgb[:, :, 2] = norm_uint8
    elif lut_name == "fire":
        rgb[:, :, 0] = np.clip(norm_uint8 * 1.5, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((norm_uint8 - 64) * 2.0, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((norm_uint8 - 180) * 3.0, 0, 255).astype(np.uint8)
    elif lut_name == "viridis":
        v = norm_uint8.astype(np.float32) / 255.0
        rgb[:, :, 0] = np.clip((0.267 + 0.3 * v + 0.4 * (v ** 2)) * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((0.003 + 0.8 * v - 0.2 * (v ** 2)) * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((0.329 + 0.4 * v - 0.7 * (v ** 2)) * 255, 0, 255).astype(np.uint8)
    elif lut_name == "plasma":
        v = norm_uint8.astype(np.float32) / 255.0
        rgb[:, :, 0] = np.clip((0.058 + 1.2 * v - 0.3 * (v ** 2)) * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((0.015 + 0.1 * v + 0.8 * (v ** 2)) * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((0.550 + 0.7 * v - 1.2 * (v ** 2)) * 255, 0, 255).astype(np.uint8)
    else:
        rgb[:, :, 0] = norm_uint8
        rgb[:, :, 1] = norm_uint8
        rgb[:, :, 2] = norm_uint8
    return rgb


def array_to_base64_png(rgb_array: np.ndarray) -> str:
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def array_to_base64_jpeg(rgb_array: np.ndarray, quality: int = 85) -> str:
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Flask Endpoints ---

@app.route("/")
@app.route("/base_caller")
def index():
    return render_template("index.html", initial_mode="base_caller")


@app.route("/blob_validator")
def blob_validator():
    return render_template("index.html", initial_mode="blob_validator")


@app.route("/api/config")
def get_config():
    tiff_path = get_tiff_path()
    img = load_tiff_image(tiff_path)
    meta = describe_tiff_array(img)
    has_c0 = bool(meta.get("has_cycle_zero", False))
    n_cycles = int(meta.get("sequencing_cycles", DEFAULT_SEQUENCE_CYCLE_COUNT))

    return jsonify({
        "tiff_path": tiff_path,
        "tiff_filename": os.path.basename(tiff_path) if tiff_path else "None",
        "has_cycle_zero": has_c0,
        "sequencing_cycles": max(1, n_cycles),
        "available_cycles": list(range(1, max(1, n_cycles) + 1)),
        "default_cycle": DEFAULT_CYCLE,
        "default_tile": DEFAULT_TILE,
        "nuclear_diameter": DEFAULT_NUCLEI_DIAMETER,
        "base_ground_truth_file": os.path.basename(BASE_GROUND_TRUTH_PATH),
        "blob_ground_truth_file": os.path.basename(BLOB_GROUND_TRUTH_PATH),
        "image_width": int(meta.get("width") or 2048),
        "image_height": int(meta.get("height") or 2048),
    })


@app.route("/api/lut_profile", methods=["GET", "POST"])
def handle_lut_profile():
    fp = get_current_tiff_fingerprint()
    if not fp:
        return jsonify({"success": False, "error": "No TIFF loaded"}), 400

    db = _read_lut_database()
    if request.method == "POST":
        data = request.json or {}
        cycle_str = str(data.get("cycle", 1))
        cycle_settings = data.get("settings", {})
        c0_settings = data.get("cycle0_settings", {})

        img_entry = db.setdefault("images", {}).setdefault(fp, {
            "filename": os.path.basename(get_tiff_path() or ""),
            "settings": {"cycles": {}, "cycle0": {}}
        })
        img_entry.setdefault("settings", {}).setdefault("cycles", {})[cycle_str] = cycle_settings
        if c0_settings:
            img_entry.setdefault("settings", {})["cycle0"] = c0_settings

        _write_lut_database(db)
        return jsonify({"success": True, "cycle": cycle_str})

    # GET profile
    img_entry = db.get("images", {}).get(fp, {})
    return jsonify({
        "success": True,
        "profile": img_entry.get("settings", {"cycles": {}, "cycle0": {}})
    })


@app.route("/api/full_fov_images")
def get_full_fov_images():
    cycle = int(request.args.get("cycle", DEFAULT_CYCLE))
    tiff_path = get_tiff_path()
    img = load_tiff_image(tiff_path)

    g_min = float(request.args.get("green_min", 0))
    g_max = float(request.args.get("green_max", 65535))
    r_min = float(request.args.get("red_min", 0))
    r_max = float(request.args.get("red_max", 65535))
    b_min = float(request.args.get("dapi_min", 0))
    b_max = float(request.args.get("dapi_max", 65535))
    c0_fr_min = float(request.args.get("c0_farred_min", 0))
    c0_fr_max = float(request.args.get("c0_farred_max", 65535))
    c0_dapi_min = float(request.args.get("c0_dapi_min", b_min))
    c0_dapi_max = float(request.args.get("c0_dapi_max", b_max))

    g_mode = request.args.get("green_mode", "pseudocolor")
    r_mode = request.args.get("red_mode", "pseudocolor")
    g_lut = request.args.get("green_lut", "green")
    r_lut = request.args.get("red_lut", "red")
    b_lut = request.args.get("dapi_lut", "dapi")

    g_blue = request.args.get("green_blue", "false").lower() == "true"
    r_blue = request.args.get("red_blue", "false").lower() == "true"
    overlay_blue = request.args.get("overlay_blue", "true").lower() == "true"
    c0_overlay_blue = request.args.get("c0_overlay_blue", "true").lower() == "true"

    sub = 2
    dapi_p = extract_channel_frame(img, cycle=cycle, channel_idx=0)[::sub, ::sub]
    green_p = extract_channel_frame(img, cycle=cycle, channel_idx=1)[::sub, ::sub]
    red_p = extract_channel_frame(img, cycle=cycle, channel_idx=2)[::sub, ::sub]

    g_norm = normalize_channel(green_p, g_min, g_max)
    r_norm = normalize_channel(red_p, r_min, r_max)
    b_norm = normalize_channel(dapi_p, b_min, b_max)

    g_rgb = apply_lut(g_norm, lut_name=g_lut, display_mode=g_mode)
    if g_blue:
        g_rgb[:, :, 2] = np.maximum(g_rgb[:, :, 2], b_norm)

    r_rgb = apply_lut(r_norm, lut_name=r_lut, display_mode=r_mode)
    if r_blue:
        r_rgb[:, :, 2] = np.maximum(r_rgb[:, :, 2], b_norm)

    h, w = g_norm.shape
    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_rgb[:, :, 0] = r_norm
    overlay_rgb[:, :, 1] = g_norm
    if overlay_blue:
        overlay_rgb[:, :, 2] = b_norm

    # Cycle 0 FISH probe (Far-red / 647 = Magenta + DAPI = Blue)
    c0_fr = extract_channel_frame(img, cycle=0, channel_idx=3)[::sub, ::sub]
    c0_dapi = extract_channel_frame(img, cycle=0, channel_idx=0)[::sub, ::sub]
    c0_fr_norm = normalize_channel(c0_fr, c0_fr_min, c0_fr_max)
    c0_b_norm = normalize_channel(c0_dapi, c0_dapi_min, c0_dapi_max)

    cycle0_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    cycle0_rgb[:, :, 0] = c0_fr_norm  # Red component of magenta
    if c0_overlay_blue:
        cycle0_rgb[:, :, 2] = np.maximum(c0_fr_norm, c0_b_norm)  # Blue component
    else:
        cycle0_rgb[:, :, 2] = c0_fr_norm

    resp = jsonify({
        "green_fov": array_to_base64_jpeg(g_rgb),
        "red_fov": array_to_base64_jpeg(r_rgb),
        "overlay_fov": array_to_base64_jpeg(overlay_rgb),
        "cycle0_fov": array_to_base64_jpeg(cycle0_rgb)
    })
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/cells")
def get_cells():
    cycle = int(request.args.get("cycle", DEFAULT_CYCLE))
    df_coords = load_cell_coordinates()

    # Load existing annotations
    base_gt_dict = {}
    if os.path.exists(BASE_GROUND_TRUTH_PATH):
        try:
            b_df = _current_image_rows(pd.read_csv(BASE_GROUND_TRUTH_PATH))
            for _, r in b_df.iterrows():
                key = f"{int(r['cell'])}_{int(r.get('cycle', cycle))}"
                base_gt_dict[key] = str(r["annotated_base"])
        except Exception:
            pass

    blob_gt_dict = {}
    blob_presence_dict = {}
    if os.path.exists(BLOB_GROUND_TRUTH_PATH):
        try:
            bl_df = _current_image_rows(pd.read_csv(BLOB_GROUND_TRUTH_PATH))
            for _, r in bl_df.iterrows():
                key = f"{int(r['cell'])}_{int(r.get('cycle', cycle))}"
                status = str(r.get("call_status", r.get("blob_call_status", "")))
                if not status or status == "nan":
                    status = "both" if bool(r.get("is_real_blob")) else "neither"
                blob_gt_dict[key] = status
                detected = [_optional_bool(r.get(f"{c}_detected")) for c in ("green", "red")]
                if all(v is not None for v in detected) and status in {"both", "only_green", "only_red", "neither"}:
                    blob_presence_dict[key] = ((detected[0] == (status in ("both", "only_green"))) or
                                               (detected[1] == (status in ("both", "only_red"))))
                else:
                    blob_presence_dict[key] = _optional_bool(r.get("is_real_blob"))
        except Exception:
            pass

    cells_list = []
    for idx, row in df_coords.iterrows():
        c_id = int(row.get("cell", idx + 1))
        key = f"{c_id}_{cycle}"
        b_status = blob_gt_dict.get(key, None)
        cells_list.append({
            "index": idx,
            "cell_id": c_id,
            "y": float(row.get("i", row.get("y", 0.0))),
            "x": float(row.get("j", row.get("x", 0.0))),
            "tile": int(row.get("tile", DEFAULT_TILE)),
            "well": str(row.get("well", "B2")),
            "cycle": cycle,
            "annotated_base": base_gt_dict.get(key, None),
            "is_real_blob": blob_presence_dict.get(key),
            "blob_call_status": b_status
        })

    return jsonify({
        "cycle": cycle,
        "total_cells": len(cells_list),
        "annotated_base_count": sum(1 for c in cells_list if c["annotated_base"] is not None),
        "annotated_blob_count": sum(1 for c in cells_list if c["blob_call_status"] is not None),
        "cells": cells_list
    })


@app.route("/api/crop/<int:cell_id>")
def get_cell_crop(cell_id: int):
    cycle = int(request.args.get("cycle", DEFAULT_CYCLE))
    tiff_path = get_tiff_path()
    img = load_tiff_image(tiff_path)
    if img is None:
        return jsonify({"error": "No TIFF loaded"}), 404

    df_coords = load_cell_coordinates()
    cell_row = df_coords[df_coords["cell"] == cell_id]
    if cell_row.empty:
        return jsonify({"error": f"Cell {cell_id} not found"}), 404

    y_coord = float(cell_row["i"].values[0] if "i" in cell_row.columns else cell_row["y"].values[0])
    x_coord = float(cell_row["j"].values[0] if "j" in cell_row.columns else cell_row["x"].values[0])

    try:
        dia = float(request.args.get("nuclear_diameter", DEFAULT_NUCLEI_DIAMETER))
        radius_scale = float(request.args.get("radius_scale", 1.6))
        detector_params = dict(DEFAULT_DETECTOR_PARAMS)
        request_names = {"high_thresh": "hysteresis_high", "low_thresh": "hysteresis_low", "alpha": "spot_alpha"}
        for name, default in detector_params.items():
            detector_params[name] = float(request.args.get(request_names.get(name, name), default))
        detector_params["min_area"] = int(detector_params["min_area"])
        if "hysteresis_low" not in request.args:
            detector_params["low_thresh"] = max(1.0, detector_params["high_thresh"] * 0.5)
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "Invalid analysis settings"}), 400

    fp = get_current_tiff_fingerprint()
    y_column = "i" if "i" in df_coords else "y"
    x_column = "j" if "j" in df_coords else "x"
    neighbors = df_coords.loc[df_coords["cell"] != cell_id, [y_column, x_column]].to_numpy(dtype=float)
    coord_digest = hashlib.sha256(neighbors.tobytes()).hexdigest()
    # Exact values, including coordinates, enter the key. LUTs never enter analysis.
    analysis_key = hashlib.sha256(repr((fp, cell_id, cycle, y_coord, x_coord, dia,
        radius_scale, sorted(detector_params.items()), coord_digest)).encode()).hexdigest()
    if analysis_key not in CROP_ANALYSIS_CACHE:
        try:
            analysis = analyze_nucleus_cycle(
                extract_channel_frame(img, cycle=cycle, channel_idx=1),
                extract_channel_frame(img, cycle=cycle, channel_idx=2),
                extract_channel_frame(img, cycle=cycle, channel_idx=0),
                (y_coord, x_coord), nuclear_diameter=dia, radius_scale=radius_scale,
                reference_dapi=extract_channel_frame(img, cycle=0, channel_idx=0),
                detector_params=detector_params, neighbor_centers_yx=neighbors,
                include_contours=True,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        box = analysis["crop_box"]
        analysis["c0_fr_crop"] = extract_channel_frame(img, cycle=0, channel_idx=3)[
            box["y_min"]:box["y_max"], box["x_min"]:box["x_max"]]
        analysis["identity"] = {"cell_id": cell_id, "cycle": cycle, "tiff_fingerprint": fp}
        analysis["provenance"].update({"tiff_fingerprint": fp,
            "tiff_filename": os.path.basename(tiff_path),
            "source_dtype": str(np.asarray(img).dtype),
            "source_layout": describe_tiff_array(img)["layout"],
            "neighbor_coordinate_digest": coord_digest})
        CROP_ANALYSIS_CACHE[analysis_key] = analysis
        if len(CROP_ANALYSIS_CACHE) > 300:
            del CROP_ANALYSIS_CACHE[next(iter(CROP_ANALYSIS_CACHE))]
    analysis = CROP_ANALYSIS_CACHE[analysis_key]
    g_crop, r_crop, b_crop = (analysis[k] for k in ("g_crop", "r_crop", "b_crop"))
    c0_fr_crop, c0_b_crop = analysis["c0_fr_crop"], analysis["c0_b_crop"]
    crop_box, contours = analysis["crop_box"], analysis["contours"]
    detection_summary, segmentation_stats = analysis["detection_summary"], analysis["segmentation_stats"]
    base_features, blob_features = analysis["base_features"], analysis["blob_features"]
    H_c, W_c = g_crop.shape

    # LUT normalization & pseudocolor mapping (runs in ~1ms on local crop array)
    g_min = float(request.args.get("green_min", 0))
    g_max = float(request.args.get("green_max", 65535))
    r_min = float(request.args.get("red_min", 0))
    r_max = float(request.args.get("red_max", 65535))
    b_min = float(request.args.get("dapi_min", 0))
    b_max = float(request.args.get("dapi_max", 65535))
    c0_fr_min = float(request.args.get("c0_farred_min", 0))
    c0_fr_max = float(request.args.get("c0_farred_max", 65535))
    c0_dapi_min = float(request.args.get("c0_dapi_min", b_min))
    c0_dapi_max = float(request.args.get("c0_dapi_max", b_max))

    g_norm = normalize_channel(g_crop, g_min, g_max)
    r_norm = normalize_channel(r_crop, r_min, r_max)
    b_norm = normalize_channel(b_crop, b_min, b_max)
    c0_fr_norm = normalize_channel(c0_fr_crop, c0_fr_min, c0_fr_max)
    c0_b_norm = normalize_channel(c0_b_crop, c0_dapi_min, c0_dapi_max)

    g_lut = request.args.get("green_lut", "green")
    r_lut = request.args.get("red_lut", "red")
    g_mode = request.args.get("green_mode", "pseudocolor")
    r_mode = request.args.get("red_mode", "pseudocolor")

    g_blue = request.args.get("green_blue", "false").lower() == "true"
    r_blue = request.args.get("red_blue", "false").lower() == "true"
    overlay_blue = request.args.get("overlay_blue", "true").lower() == "true"
    c0_overlay_blue = request.args.get("c0_overlay_blue", "true").lower() == "true"

    g_rgb = apply_lut(g_norm, lut_name=g_lut, display_mode=g_mode)
    if g_blue:
        g_rgb[:, :, 2] = np.maximum(g_rgb[:, :, 2], b_norm)

    r_rgb = apply_lut(r_norm, lut_name=r_lut, display_mode=r_mode)
    if r_blue:
        r_rgb[:, :, 2] = np.maximum(r_rgb[:, :, 2], b_norm)

    overlay_rgb = np.zeros((H_c, W_c, 3), dtype=np.uint8)
    overlay_rgb[:, :, 0] = r_norm
    overlay_rgb[:, :, 1] = g_norm
    if overlay_blue:
        overlay_rgb[:, :, 2] = b_norm

    c0_rgb = np.zeros((H_c, W_c, 3), dtype=np.uint8)
    c0_rgb[:, :, 0] = c0_fr_norm
    if c0_overlay_blue:
        c0_rgb[:, :, 2] = np.maximum(c0_fr_norm, c0_b_norm)
    else:
        c0_rgb[:, :, 2] = c0_fr_norm

    resp = jsonify({
        "cell_id": cell_id,
        "cycle": cycle,
        "x": x_coord,
        "y": y_coord,
        "crop_box": crop_box,
        "analysis_token": analysis_key,
        "analysis_provenance": analysis["provenance"],
        "crops": {
            "green": array_to_base64_png(g_rgb),
            "red": array_to_base64_png(r_rgb),
            "composite": array_to_base64_png(overlay_rgb),
            "cycle0": array_to_base64_png(c0_rgb)
        },
        "contours": contours,
        "detection": detection_summary,
        "segmentation_stats": segmentation_stats,
        "base_features": _json_safe(base_features),
        "blob_features": _json_safe(blob_features)
    })
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _optional_bool(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1"}:
            return True
        if value.strip().lower() in {"false", "0"}:
            return False
        return None
    return bool(value)


def _current_image_rows(df):
    filename = os.path.basename(get_tiff_path() or "N/A")
    fingerprint = get_current_tiff_fingerprint()
    if "tiff_fingerprint" in df:
        specified = df["tiff_fingerprint"].notna() & df["tiff_fingerprint"].ne("")
        keep = specified & df["tiff_fingerprint"].eq(fingerprint)
        if "tiff_filename" in df:
            keep |= ~specified & df["tiff_filename"].eq(filename)
        return df.loc[keep]
    if "tiff_filename" in df:
        return df.loc[df["tiff_filename"].eq(filename)]
    return df


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _annotation_source(data, workflow):
    """Bind an annotation to the exact crop shown; old API clients are flagged."""
    token = data.get("analysis_token")
    if not token:
        return dict(data), {"analysis_version": "legacy_client_unverified",
            "tiff_fingerprint": get_current_tiff_fingerprint()}
    analysis = CROP_ANALYSIS_CACHE.get(token)
    expected = {"cell_id": int(data.get("cell_id", 0)),
                "cycle": int(data.get("cycle", DEFAULT_CYCLE)),
                "tiff_fingerprint": get_current_tiff_fingerprint()}
    if analysis is None or analysis["identity"] != expected:
        raise ValueError("Crop analysis expired or belongs to another cell/image; reload the crop before annotating")
    source = dict(data)
    source.update(analysis["base_row"] if workflow == "base" else analysis["blob_features"])
    source.update(analysis["detection_summary"])
    source["y"], source["x"] = analysis["provenance"]["center_input_yx"]
    return source, dict(analysis["provenance"])


def _provenance_columns(provenance):
    return {"analysis_version": provenance["analysis_version"],
        "tiff_fingerprint": provenance.get("tiff_fingerprint"),
        "nuclear_diameter": provenance.get("nuclear_diameter"),
        "nuclear_area": provenance.get("nuclear_area"),
        "radius_scale": provenance.get("radius_scale"),
        "analysis_provenance_json": json.dumps(_json_safe(provenance), sort_keys=True, allow_nan=False)}


def _save_annotation_row(path, row):
    # Do not overwrite rows from other images sharing the same numeric cell ID.
    with ANNOTATION_LOCK:
        if os.path.exists(path):
            df = pd.read_csv(path)  # A corrupt file must fail rather than silently erase labels.
            same = (df["cell"] == row["cell"]) & (df["cycle"] == row["cycle"])
            if "tiff_fingerprint" in df:
                same &= df["tiff_fingerprint"].fillna("").eq(row.get("tiff_fingerprint") or "")
            elif "tiff_filename" in df:
                same &= df["tiff_filename"].eq(row["tiff_filename"])
            for field in ("tile", "well"):
                if field in df:
                    same &= df[field].eq(row[field])
            df = pd.concat([df.loc[~same], pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        temp_path = path + ".tmp"
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    return len(df)


# --- Annotation Persistence (Two Separate Workflows) ---

@app.route("/api/annotate_base", methods=["POST"])
def annotate_base():
    """Saves ground truth for Base Caller (Tier 1 quantiles, trimmed mean, and Tier 2 JSON)."""
    try:
        data, provenance = _annotation_source(request.json or {}, "base")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    cell_id = int(data.get("cell_id", 0))
    base = str(data.get("base", "N")).upper()
    if base not in {"A", "C", "G", "T", "N"}:
        return jsonify({"error": "Base must be A, C, G, T, or N"}), 400
    cycle = int(data.get("cycle", DEFAULT_CYCLE))
    tile = int(data.get("tile", DEFAULT_TILE))
    well = str(data.get("well", "B2"))
    x = float(data.get("x", 0.0))
    y = float(data.get("y", 0.0))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tiff_name = os.path.basename(get_tiff_path() or "N/A")

    row = {
        "cell": cell_id,
        "cycle": cycle,
        "tile": tile,
        "well": well,
        "x": x,
        "y": y,
        "annotated_base": base,
        "blob_area": int(data.get("blob_area", 0)),
        # Green Tier 1 + Trimmed Mean
        "green_median": float(data.get("green_median", 0.0)),
        "green_trimmed_mean": float(data.get("green_trimmed_mean", 0.0)),
        "green_p25": float(data.get("green_p25", 0.0)),
        "green_p75": float(data.get("green_p75", 0.0)),
        "green_p90": float(data.get("green_p90", 0.0)),
        "green_p95": float(data.get("green_p95", 0.0)),
        "green_mean": float(data.get("green_mean", 0.0)),
        "green_std": float(data.get("green_std", 0.0)),
        "green_excess_median": float(data.get("green_excess_median", 0.0)),
        "green_integrated_excess": float(data.get("green_integrated_excess", 0.0)),
        # Red Tier 1 + Trimmed Mean
        "red_median": float(data.get("red_median", 0.0)),
        "red_trimmed_mean": float(data.get("red_trimmed_mean", 0.0)),
        "red_p25": float(data.get("red_p25", 0.0)),
        "red_p75": float(data.get("red_p75", 0.0)),
        "red_p90": float(data.get("red_p90", 0.0)),
        "red_p95": float(data.get("red_p95", 0.0)),
        "red_mean": float(data.get("red_mean", 0.0)),
        "red_std": float(data.get("red_std", 0.0)),
        "red_excess_median": float(data.get("red_excess_median", 0.0)),
        "red_integrated_excess": float(data.get("red_integrated_excess", 0.0)),
        # Distribution ratios
        "rg_ratio_median": float(data.get("rg_ratio_median", 0.0)),
        "rg_ratio_trimmed": float(data.get("rg_ratio_trimmed", 0.0)),
        # Tier 2 Empirical distribution (JSON strings)
        "green_distribution_json": json.dumps(data.get("green_distribution", {})),
        "red_distribution_json": json.dumps(data.get("red_distribution", {})),
        "timestamp": timestamp,
        "tiff_filename": tiff_name
    }

    row.update(_provenance_columns(provenance))
    # Preserve the complete shared extractor output, including previously dropped features.
    for key, value in data.items():
        if key.startswith(("green_", "red_")) and key not in {"green_distribution", "red_distribution"}:
            row[key + "_json" if isinstance(value, (dict, list)) else key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        elif key in {"has_blob", "blob_area", "dist_to_center_norm", "ownership_ratio", "dapi_under_blob_mean"}:
            row[key] = value
    total = _save_annotation_row(BASE_GROUND_TRUTH_PATH, row)
    return jsonify({"success": True, "cell_id": cell_id, "annotated_base": base, "total": total})


@app.route("/api/annotate_blob", methods=["POST"])
def annotate_blob():
    """Saves ground truth for Blob Signal Validator across the 4 channel accuracy options."""
    try:
        data, provenance = _annotation_source(request.json or {}, "blob")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    cell_id = int(data.get("cell_id", 0))
    call_status = str(data.get("call_status", "")).strip()
    legacy_binary = not call_status
    if legacy_binary:
        call_status = "both" if bool(data.get("is_real_blob", False)) else "neither"
    if call_status not in {"both", "only_green", "only_red", "neither"}:
        return jsonify({"error": "Unknown channel-correctness label"}), 400
    green_correct = call_status in ("both", "only_green")
    red_correct = call_status in ("both", "only_red")
    green_present = _optional_bool(data["green_detected"]) == green_correct if _optional_bool(data.get("green_detected")) is not None else None
    red_present = _optional_bool(data["red_detected"]) == red_correct if _optional_bool(data.get("red_detected")) is not None else None
    is_real = (green_present or red_present) if green_present is not None and red_present is not None else (
        bool(data.get("is_real_blob", False)) if legacy_binary else None)

    cycle = int(data.get("cycle", DEFAULT_CYCLE))
    tile = int(data.get("tile", DEFAULT_TILE))
    well = str(data.get("well", "B2"))
    x = float(data.get("x", 0.0))
    y = float(data.get("y", 0.0))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tiff_name = os.path.basename(get_tiff_path() or "N/A")

    row = {
        "cell": cell_id,
        "cycle": cycle,
        "tile": tile,
        "well": well,
        "x": x,
        "y": y,
        "call_status": call_status,
        "blob_call_status": call_status,
        "is_real_blob": is_real,
        "green_called_correctly": green_correct,
        "red_called_correctly": red_correct,
        "green_true_presence": green_present,
        "red_true_presence": red_present,
        "label_semantics": "channel_correctness" if not legacy_binary else "legacy_binary",
        "green_detected": bool(data.get("green_detected", False)),
        "red_detected": bool(data.get("red_detected", False)),
        "green_area": int(data.get("green_area", 0)),
        "red_area": int(data.get("red_area", 0)),
        "area": int(data.get("area", 0)),
        "fraction_nucleus_occupied": float(data.get("fraction_nucleus_occupied", 0.0)),
        "median_contrast": float(data.get("median_contrast", 0.0)),
        "peak_contrast": float(data.get("peak_contrast", 0.0)),
        "upper_quantile_contrast": float(data.get("upper_quantile_contrast", 0.0)),
        "integrated_excess_intensity": float(data.get("integrated_excess_intensity", 0.0)),
        "spatial_smoothness": float(data.get("spatial_smoothness", 0.0)),
        "solidity": float(data.get("solidity", 0.0)),
        "eccentricity": float(data.get("eccentricity", 0.0)),
        "perimeter_to_area_ratio": float(data.get("perimeter_to_area_ratio", 0.0)),
        "persistence_cycles": float(data.get("persistence_cycles", 1.0)),
        "distance_from_nuclear_boundary": float(data.get("distance_from_nuclear_boundary", 0.0)),
        "saturated_pixel_fraction": float(data.get("saturated_pixel_fraction", 0.0)),
        "peak_excess_green": float(data.get("peak_excess_green", 0.0)),
        "peak_excess_red": float(data.get("peak_excess_red", 0.0)),
        "max_peak_excess": float(data.get("max_peak_excess", 0.0)),
        "fold_change_green": float(data.get("fold_change_green", 1.0)),
        "fold_change_red": float(data.get("fold_change_red", 1.0)),
        "max_fold_change": float(data.get("max_fold_change", 1.0)),
        "bg_green": float(data.get("bg_green", 0.0)),
        "bg_red": float(data.get("bg_red", 0.0)),
        "noise_green": float(data.get("noise_green", 0.0)),
        "noise_red": float(data.get("noise_red", 0.0)),
        "dist_to_center_norm": float(data.get("dist_to_center_norm", 0.0)),
        "ownership_ratio": float(data.get("ownership_ratio", 0.0)),
        "dapi_under_blob_mean": float(data.get("dapi_under_blob_mean", 0.0)),
        "p_real_blob": float(data.get("p_real_blob", 0.0)),
        "timestamp": timestamp,
        "tiff_filename": tiff_name
    }

    row.update(_provenance_columns(provenance))
    if data.get("analysis_token"):
        row.update(CROP_ANALYSIS_CACHE[data["analysis_token"]]["blob_features"])
    total = _save_annotation_row(BLOB_GROUND_TRUTH_PATH, row)
    return jsonify({
        "success": True,
        "cell_id": cell_id,
        "call_status": call_status,
        "is_real_blob": is_real,
        "green_called_correctly": green_correct,
        "red_called_correctly": red_correct,
        "total": total
    })


@app.route("/api/ground_truth_base")
def get_ground_truth_base():
    if not os.path.exists(BASE_GROUND_TRUTH_PATH):
        return jsonify({"total": 0, "annotations": []})
    try:
        df = pd.read_csv(BASE_GROUND_TRUTH_PATH).fillna("")
        return jsonify({"total": len(df), "annotations": df.to_dict(orient="records")})
    except Exception:
        return jsonify({"total": 0, "annotations": []})


@app.route("/api/ground_truth_blob")
def get_ground_truth_blob():
    if not os.path.exists(BLOB_GROUND_TRUTH_PATH):
        return jsonify({"total": 0, "annotations": []})
    try:
        df = pd.read_csv(BLOB_GROUND_TRUTH_PATH).fillna("")
        return jsonify({"total": len(df), "annotations": df.to_dict(orient="records")})
    except Exception:
        return jsonify({"total": 0, "annotations": []})


@app.route("/api/clear_ground_truth", methods=["POST"])
def clear_ground_truth():
    target = (request.json or {}).get("target", "both")
    if target in ("base", "both") and os.path.exists(BASE_GROUND_TRUTH_PATH):
        os.remove(BASE_GROUND_TRUTH_PATH)
    if target in ("blob", "both") and os.path.exists(BLOB_GROUND_TRUTH_PATH):
        os.remove(BLOB_GROUND_TRUTH_PATH)
    return jsonify({"success": True, "target": target})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    global CUSTOM_TIFF_PATH, ACTIVE_TSV_PATH, ACTIVE_TIFF_FINGERPRINT, ACTIVE_TIFF_FINGERPRINT_PATH
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No filename"}), 400
    fname = secure_filename(f.filename)
    dst = os.path.join(UPLOAD_DIR, fname)
    f.save(dst)
    ext = os.path.splitext(fname)[1].lower()

    if ext in [".tif", ".tiff"]:
        CUSTOM_TIFF_PATH = dst
        TIFF_FINGERPRINT_CACHE.pop(os.path.abspath(dst), None)
        ACTIVE_TIFF_FINGERPRINT = None
        ACTIVE_TIFF_FINGERPRINT_PATH = None
        IMAGE_CACHE.clear()
        NUC_CROP_CACHE.clear()
        CROP_ANALYSIS_CACHE.clear()
        file_type = "tiff"
    elif ext in [".tsv", ".csv"]:
        ACTIVE_TSV_PATH = dst
        file_type = "tsv"
    else:
        return jsonify({"error": f"Unsupported extension {ext}"}), 400

    return jsonify({"success": True, "file_type": file_type, "filename": fname})


if __name__ == "__main__":
    print("Starting POPCROP Manual Base Caller & Signal Validator on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
