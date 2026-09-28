"""Collision-safe file-run identities and provenance for combining exports."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4


IDENTITY_COLUMNS = (
    "run_id", "source_image", "source_path", "plate", "well", "tile",
    "experiment_date", "run_started_utc", "position_idx",
)


def normalize_experiment_date(value):
    """Only explicit ISO dates are accepted; file modification time is irrelevant."""
    if value is None:
        return None
    value = str(value)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Experiment date must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("Experiment date must be a valid YYYY-MM-DD date") from exc


def _filename_part(value, limit):
    # Bound each part to leave space for suffixes on filesystems with NAME_MAX=255.
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-")[:limit] or "unspecified"


def allocate_run(output_root, image_paths, config, position_idx=0):
    """Reserve a new directory atomically, including when HPC jobs start together."""
    sources = list(image_paths) if isinstance(image_paths, (list, tuple)) else [image_paths]
    paths = [Path(source) for source in sources]
    if not paths:
        raise ValueError("At least one source image is required")
    experiment_date = normalize_experiment_date(config.experiment_date)
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    parts = [
        _filename_part(paths[0].stem, 60),
        "plate-" + _filename_part(config.plate_id, 20),
        "well-" + _filename_part(config.well_id, 16),
        "tile-" + _filename_part(config.tile_id, 28),
    ]
    if experiment_date:
        parts.append("date-" + experiment_date)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        run_id = timestamp + "_" + uuid4().hex[:12]
        prefix = "__".join([*parts, run_id])
        directory = root / prefix
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return directory, prefix, {
            "run_id": run_id,
            "source_image": paths[0].name,
            "source_path": str(paths[0].resolve()),
            "source_images": [p.name for p in paths],
            "source_paths": [str(p.resolve()) for p in paths],
            "plate": str(config.plate_id), "well": str(config.well_id),
            "tile": str(config.tile_id), "experiment_date": experiment_date,
            "run_started_utc": started.isoformat().replace("+00:00", "Z"),
            "position_idx": position_idx,
            "output_dir": str(directory), "filename_prefix": prefix,
        }
    raise FileExistsError("Could not reserve a unique run directory; no existing output was overwritten")


def add_run_columns(frame, metadata):
    """Attach identity in place, retaining a useful schema even for empty tables."""
    for name in IDENTITY_COLUMNS:
        frame[name] = metadata[name]
    if "nucleus_id" in frame:
        frame["nucleus_uid"] = metadata["run_id"] + ":" + frame.nucleus_id.astype(str)


def write_run_manifest(path, metadata, status, saved_files=None):
    """Only 'complete' manifests identify fully exported runs, including diagnostics."""
    record = {**metadata, "status": status, "saved_files": saved_files or {}}
    if status == "complete":
        record["run_completed_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
