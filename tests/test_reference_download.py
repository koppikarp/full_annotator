"""Small standard-library tests; no scientific environment or network required."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.fetch_reference_image import fetch_reference_image, load_metadata


class ReferenceImageDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = b"II*\x00reference-image-test-pixels\x00\xff" * 2048
        self.archive = self.root / "reference.tif.gz"
        self.archive.write_bytes(gzip.compress(self.raw, mtime=0))
        self.output = self.root / "restored" / "reference.tif"
        self.metadata = {
            "repository": "koppikarp/full_annotator", "tag": "reference-data-v1",
            "asset_name": self.archive.name, "filename": self.output.name,
            "raw_sha256": hashlib.sha256(self.raw).hexdigest(), "raw_size": len(self.raw),
            "archive_sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            "archive_size": self.archive.stat().st_size,
        }

    def test_local_restore_verifies_exact_bytes_and_removes_partial_files(self):
        with patch("scripts.fetch_reference_image.shutil.which", side_effect=AssertionError("Local restore must not need gh")):
            restored = fetch_reference_image(self.metadata, self.output, self.archive)
        self.assertEqual(restored, self.output)
        self.assertEqual(self.output.read_bytes(), self.raw)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])
        self.assertTrue(self.archive.exists())

    def test_wrong_archive_hash_leaves_no_output(self):
        self.metadata["archive_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "archive SHA-256"):
            fetch_reference_image(self.metadata, self.output, self.archive)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.rglob("*.partial")), [])

    def test_wrong_raw_hash_cleans_partial_output(self):
        self.metadata["raw_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Decompressed reference image"):
            fetch_reference_image(self.metadata, self.output, self.archive)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])

    def test_differing_existing_output_is_preserved(self):
        self.output.parent.mkdir()
        original = b"x" * len(self.raw)
        self.output.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Existing output.*preserved"):
            fetch_reference_image(self.metadata, self.output, self.archive)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_identical_existing_output_needs_no_download(self):
        self.output.parent.mkdir()
        self.output.write_bytes(self.raw)
        with patch("scripts.fetch_reference_image.shutil.which", side_effect=AssertionError("No download needed")):
            self.assertEqual(fetch_reference_image(self.metadata, self.output), self.output)

    def test_missing_github_cli_has_actionable_error(self):
        with patch("scripts.fetch_reference_image.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "gh auth login.*--archive"):
                fetch_reference_image(self.metadata, self.output)
        self.assertFalse(self.output.exists())

    def test_download_failure_removes_partial_archive(self):
        download_directories = []

        def fail_download(command, **kwargs):
            directory = Path(command[command.index("--dir") + 1])
            download_directories.append(directory)
            (directory / self.metadata["asset_name"]).write_bytes(b"partial download")
            raise subprocess.CalledProcessError(1, command)

        with patch("scripts.fetch_reference_image.shutil.which", return_value="/mock/gh"), \
             patch("scripts.fetch_reference_image.subprocess.run", side_effect=fail_download):
            with self.assertRaisesRegex(RuntimeError, "gh auth status"):
                fetch_reference_image(self.metadata, self.output)
        self.assertEqual(len(download_directories), 1)
        self.assertFalse(download_directories[0].exists())
        self.assertFalse(self.output.exists())

    def test_metadata_loading_and_filename_validation(self):
        path = self.root / "reference_image.json"
        path.write_text(json.dumps(self.metadata))
        self.assertEqual(load_metadata(path), self.metadata)
        self.metadata["filename"] = "../outside.tif"
        path.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "plain filename"):
            load_metadata(path)


if __name__ == "__main__":
    unittest.main()
