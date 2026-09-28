#!/usr/bin/env python3
"""Restore the original reference TIFF from its checksum-verified release archive.

Only Python's standard library is required. Downloading a private release also
requires an authenticated GitHub CLI (``gh auth login``). A previously downloaded
archive can instead be supplied with ``--archive``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data/reference_image.json"
CHUNK_SIZE = 8 * 1024 * 1024


def load_metadata(path: Path = METADATA_PATH) -> dict:
    """Read the release identity and exact byte-level integrity requirements."""
    metadata = json.loads(Path(path).read_text())
    if not isinstance(metadata, dict):
        raise ValueError("Reference metadata must be a JSON object")
    for key in ("filename", "asset_name"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(f"Reference metadata {key} must be a plain filename")
    repository = metadata.get("repository")
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Reference metadata repository must have the form owner/repository")
    tag = metadata.get("tag")
    if not isinstance(tag, str) or not tag or tag.startswith("-"):
        raise ValueError("Reference metadata requires a valid release tag")
    for prefix in ("raw", "archive"):
        digest = metadata.get(f"{prefix}_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Reference metadata requires {prefix}_sha256")
        size = metadata.get(f"{prefix}_size")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Reference metadata requires a positive {prefix}_size")
    return metadata


def verify_file(path: Path, size: int, sha256: str, description: str) -> None:
    """Verify an existing file without reading it into memory."""
    if not path.is_file():
        raise ValueError(f"{description} is not a regular file: {path}")
    if path.stat().st_size != size:
        raise ValueError(f"{description} size does not match the recorded source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"{description} SHA-256 does not match the recorded source: {path}")


def restore_archive(archive: Path, output: Path, metadata: dict) -> Path:
    """Stream a verified gzip archive to an atomically published TIFF."""
    verify_file(archive, metadata["archive_size"], metadata["archive_sha256"], "Reference archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "wb") as destination, gzip.open(archive, "rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                total += len(chunk)
                if total > metadata["raw_size"]:
                    raise ValueError("Decompressed reference image exceeds its recorded size")
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if total != metadata["raw_size"] or digest.hexdigest() != metadata["raw_sha256"]:
            raise ValueError("Decompressed reference image size or SHA-256 does not match the recorded source")
        try:
            # A same-filesystem hard link publishes complete bytes atomically and
            # refuses to replace a path created concurrently by another job.
            os.link(temporary, output)
        except FileExistsError:
            verify_file(output, metadata["raw_size"], metadata["raw_sha256"], "Existing output (preserved)")
        return output
    finally:
        temporary.unlink(missing_ok=True)


def fetch_reference_image(metadata: dict, output: Path | None = None, archive: Path | None = None) -> Path:
    """Return an existing verified image or download and reconstruct it safely."""
    output = Path(output) if output is not None else ROOT / metadata["filename"]
    if output.exists() or output.is_symlink():
        verify_file(output, metadata["raw_size"], metadata["raw_sha256"], "Existing output (preserved)")
        return output
    if archive is not None:
        return restore_archive(Path(archive), output, metadata)
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError("GitHub CLI 'gh' is required for a private release download. Install it and run 'gh auth login', or supply --archive /path/to/the/downloaded.tif.gz.")
    with tempfile.TemporaryDirectory(prefix="popcrop-reference-download-") as directory:
        command = [gh, "release", "download", metadata["tag"], "--repo", metadata["repository"],
                   "--pattern", metadata["asset_name"], "--dir", directory]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"GitHub release download failed (exit {exc.returncode}). Check 'gh auth status' and access to {metadata['repository']} release {metadata['tag']}, or supply --archive.") from exc
        return restore_archive(Path(directory) / metadata["asset_name"], output, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Restore from a local .tif.gz archive without GitHub CLI")
    parser.add_argument("--output", type=Path, help="Destination TIFF; defaults to the reference filename at the project root")
    args = parser.parse_args()
    try:
        metadata = load_metadata()
        path = fetch_reference_image(metadata, output=args.output, archive=args.archive)
    except (OSError, ValueError, RuntimeError, EOFError) as exc:
        parser.exit(1, f"Reference image restore failed: {exc}\n")
    print(f"Verified reference image: {path}")


if __name__ == "__main__":
    main()
