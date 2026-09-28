"""Exercise the portable aggregation CLI with synthetic production exports.

Run without the image-analysis environment:
    python -m unittest discover -s tests -p test_summarize_outputs.py
"""
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_outputs.py"
BARCODES = {"GENE1": "ACGTACGTACGT", "GENE2": "CCCCAAAATTTT"}
IDENTITY = (
    "run_id", "source_image", "source_path", "plate", "well", "tile",
    "experiment_date", "run_started_utc", "position_idx",
)
CELL_FIELDS = (
    "nucleus_uid", "nucleus_id", "has_detectable_spot", "has_signal",
    "called_sequence", "called_library_barcode", "assigned_gene", "assigned_barcode",
    "hamming_distance", "nearest_hamming_distance", "decoding_status",
)


def cell(gene="GENE1", *, status="matched", hd=0, nearest=None, spot=True):
    accepted = status in {"matched", "ambiguous_gene"}
    barcode = BARCODES.get(gene, "GATTACAGATTA") if accepted else ""
    return {
        "has_detectable_spot": spot,
        "has_signal": status != "no_signal",
        "called_sequence": "ACGTACGTACGT",
        "called_library_barcode": "ACGTACGTACGT",
        "assigned_gene": gene if status == "matched" else "",
        "assigned_barcode": barcode,
        "hamming_distance": hd,
        "nearest_hamming_distance": hd if nearest is None else nearest,
        "decoding_status": status,
    }


class SummarizeOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()

    def make_run(self, name, rows=None, *, plate="P1", well="B2", tile="001",
                 date="2026-09-28", hour=12, source=None, status="complete"):
        """The absolute export paths deliberately refer to a different host."""
        rows = copy.deepcopy([cell()] if rows is None else rows)
        folder = self.inputs / name
        folder.mkdir()
        source = source or f"{plate}_{well}_{tile}_RGB.tif"
        metadata = {
            "run_id": name, "source_image": source, "source_path": f"/old/images/{source}",
            "source_images": [source], "source_paths": [f"/old/images/{source}"],
            "plate": plate, "well": well, "tile": tile, "experiment_date": date,
            "run_started_utc": f"2026-09-29T{hour:02d}:00:00Z", "position_idx": 0,
            "output_dir": f"/old/results/{name}", "filename_prefix": name,
        }
        for index, row in enumerate(rows, 1):
            row.update({key: metadata[key] for key in IDENTITY})
            row.update(nucleus_id=index, nucleus_uid=f"{name}:{index}")
        self.write_csv(folder / f"{name}_nucleus_barcode_calls.csv", rows)
        config = {
            "analysis_mode": "manual", "plate_id": plate, "well_id": well,
            "tile_id": tile, "experiment_date": date, "random_seed": 42,
            "channel_names": ["Red", "Green"], "has_dapi_cycle": True,
            "preprocessing": {"camera_offset": 0.0, "do_field_flattening": False,
                              "registration_method": "none"},
            "manual_analysis": {
                "nuclear_diameter": 55.0, "radius_scale": 1.6, "batch_size": 64,
                "signal_probability_threshold": 0.7, "detector_params": {},
                "base_model_path": "/old/models/base_caller.joblib",
                "blob_model_path": "/old/models/blob_classifier.joblib",
            },
            "decoding": {
                "barcode_column": "barcode_new", "codebook_path": "/old/library.csv",
                "min_score_margin": 0.8, "min_confidence": 0.6,
                "max_hamming_distance": 2, "match_reverse_complement": True,
                "allow_recombination": False,
            },
            "output": {"output_dir": "/old/results", "save_diagnostics": True},
        }
        matched = [r for r in rows if r["decoding_status"] == "matched"]
        signal = [r for r in rows if r["has_signal"]]
        statuses = dict(Counter(r["decoding_status"] for r in rows))
        ambiguous = statuses.get("ambiguous_gene", 0)
        summary = {
            "total_nuclei": len(rows),
            "nuclei_with_detector_spots": sum(r["has_detectable_spot"] for r in rows),
            "nuclei_with_classified_signal": len(signal),
            "nuclei_without_classified_signal": len(rows) - len(signal),
            "accepted_library_assignments": len(matched),
            "accepted_barcode_sequences": len(matched) + ambiguous,
            "gene_ambiguous_barcode_assignments": ambiguous,
            "unique_assigned_genes": len({r["assigned_gene"] for r in matched}),
            "decoding_status_counts": statuses,
            "signal_nuclei_nearest_hamming_histogram": dict(Counter(
                str(r["nearest_hamming_distance"]) for r in signal
                if r["nearest_hamming_distance"] not in ("", None))),
            "accepted_assignments_selected_hamming_histogram": dict(Counter(
                str(r["hamming_distance"]) for r in matched
                if r["hamming_distance"] not in ("", None))),
            "run": metadata, "config": config,
            "model_and_codebook_sha256": {
                "/old/models/base_caller.joblib": "a" * 64,
                "/old/models/blob_classifier.joblib": "b" * 64,
                "/old/library.csv": "c" * 64,
            },
            "image_metadata": {"source_path": metadata["source_path"]},
            "runtime_seconds": 1.0, "peak_rss_mib": 10.0,
        }
        self.write_json(folder / f"{name}_run_summary.json", summary)
        manifest = {
            **metadata, "status": status, "run_completed_utc": "2026-09-29T15:00:00Z",
            "saved_files": {
                "summary_json": f"/old/results/{name}/{name}_run_summary.json",
                "barcode_calls_csv": f"/old/results/{name}/{name}_nucleus_barcode_calls.csv",
            },
        }
        self.write_json(folder / f"{name}_run_manifest.json", manifest)
        return folder

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*IDENTITY, *CELL_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def csv_rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def cli(self, *arguments, ok=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.inputs), *arguments],
            cwd=self.output, capture_output=True, text=True, timeout=30,
        )
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def summary(self, label="all"):
        return json.loads((self.output / f"combined_{label}_summary.json").read_text())["overall"]

    def test_genes_are_unioned_but_nuclei_and_gene_counts_are_summed(self):
        self.make_run("first", [cell(), cell("GENE2", hd=1),
                               cell("MULTIPLE", status="ambiguous_gene"),
                               cell(status="no_signal", spot=False)])
        self.make_run("second", [cell(hd=2, nearest=1), cell(status="unmapped", hd=3),
                                cell(status="ambiguous", hd=1)], tile="002")
        self.cli()
        overall = self.summary()
        expected = {
            "n_runs": 2, "total_nuclei": 7, "nuclei_with_detector_spots": 6,
            "nuclei_with_classified_signal": 6, "nuclei_without_classified_signal": 1,
            "accepted_library_assignments": 3, "accepted_barcode_sequences": 4,
            "gene_ambiguous_barcode_assignments": 1, "unique_assigned_genes": 2,
            "unique_assigned_barcodes": 3,
        }
        for key, value in expected.items():
            self.assertEqual(overall[key], value, key)
        self.assertEqual(overall["signal_nuclei_nearest_hamming_histogram"], {"0": 2, "1": 3, "3": 1})
        self.assertEqual(overall["accepted_assignments_selected_hamming_histogram"], {"0": 1, "1": 1, "2": 1})
        self.assertAlmostEqual(overall["percent_signal_nearest_hd_le1"], 5 / 6 * 100)
        self.assertAlmostEqual(overall["percent_gene_assigned"], 3 / 7 * 100)
        self.assertAlmostEqual(overall["percent_signal_gene_assigned"], 50)
        genes = self.csv_rows(self.output / "combined_all_genes.csv")
        self.assertEqual({r["gene"]: int(r["cell_count"]) for r in genes}, {"GENE1": 2, "GENE2": 1})
        gene1 = next(r for r in genes if r["gene"] == "GENE1")
        self.assertAlmostEqual(float(gene1["percent_mapped"]), 2 / 3 * 100, places=3)
        self.assertAlmostEqual(float(gene1["percent_total"]), 2 / 7 * 100, places=3)
        detail = self.csv_rows(self.output / "combined_all_gene_counts_by_group.csv")
        self.assertEqual({r["run_id"] for r in detail if r["group_level"] == "run"}, {"first", "second"})
        self.assertTrue((self.output / "combined_all_summary.md").is_file())

    def test_copied_run_id_is_counted_once(self):
        original = self.make_run("original")
        shutil.copytree(original, self.inputs / "backup_copy")
        self.cli()
        self.assertEqual(self.summary()["total_nuclei"], 1)
        self.assertEqual(self.summary()["n_runs"], 1)
        self.assertEqual(len(self.csv_rows(self.output / "combined_all_skipped_runs.csv")), 1)

    def test_saved_reference_counts_and_histograms_are_reproduced(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("Optional reference-table check requires pyarrow from the analysis environment")
        source = ROOT / "reports/reference_run"
        original = json.loads((source / "run_summary.json").read_text())
        rows = pq.read_table(source / "Plate1_B2_14_nuclei.parquet").to_pylist()
        cells = []
        for row in rows:
            record = {key: row.get(key) for key in CELL_FIELDS if key not in ("nucleus_id", "nucleus_uid")}
            for key in ("hamming_distance", "nearest_hamming_distance"):
                if record[key] is not None:
                    record[key] = int(record[key])
            cells.append(record)
        folder = self.make_run("reference", cells)
        shutil.copytree(folder, self.inputs / "copied_reference")
        self.cli()
        overall = self.summary()
        for key, value in original.items():
            if key in overall:
                self.assertEqual(overall[key], value, key)
        self.assertEqual(overall["n_runs"], 1)
        self.assertEqual(len(self.csv_rows(self.output / "combined_all_genes.csv")), 464)

    def test_conflicting_copies_of_same_run_id_fail(self):
        original = self.make_run("original")
        duplicate = self.inputs / "backup_copy"
        shutil.copytree(original, duplicate)
        path = duplicate / "original_nucleus_barcode_calls.csv"
        rows = self.csv_rows(path)
        rows[0]["assigned_gene"] = "CHANGED_GENE"
        self.write_csv(path, rows)
        result = self.cli(ok=False)
        self.assertIn("Conflicting copies", result.stderr)

    def test_duplicate_nucleus_ids_within_run_fail(self):
        folder = self.make_run("duplicate_labels", [cell(), cell("GENE2")])
        path = folder / "duplicate_labels_nucleus_barcode_calls.csv"
        rows = self.csv_rows(path)
        rows[1]["nucleus_id"] = rows[0]["nucleus_id"]
        rows[1]["nucleus_uid"] = rows[0]["nucleus_uid"]
        self.write_csv(path, rows)
        result = self.cli(ok=False)
        self.assertIn("duplicate nucleus_id", result.stderr)

    def test_model_signal_without_detector_spot_is_valid(self):
        self.make_run("valid", [cell(spot=False)])
        self.cli()
        overall = self.summary()
        self.assertEqual(overall["nuclei_with_detector_spots"], 0)
        self.assertEqual(overall["nuclei_with_classified_signal"], 1)
        self.assertEqual(overall["accepted_library_assignments"], 1)

    def test_n_read_with_unknown_distance_remains_in_signal_denominator(self):
        rejected = cell(status="rejected_ambiguous_bases", hd="", nearest="")
        rejected["called_sequence"] = "ACGTNCGTACGT"
        rejected["called_library_barcode"] = "ACGTACGNACGT"
        self.make_run("valid", [cell(), rejected, cell(status="no_signal")])
        self.cli()
        overall = self.summary()
        self.assertEqual(overall["total_nuclei"], 3)
        self.assertEqual(overall["nuclei_with_classified_signal"], 2)
        self.assertEqual(overall["accepted_library_assignments"], 1)
        self.assertEqual(overall["signal_nuclei_nearest_hamming_histogram"], {"0": 1})
        self.assertEqual(overall["decoding_status_counts"]["rejected_ambiguous_bases"], 1)
        self.assertEqual(overall["percent_signal_nearest_hd_le1"], 50)

    def test_latest_rerun_default_and_explicit_alternatives(self):
        self.make_run("older", [cell()], hour=10)
        self.make_run("newer", [cell(), cell("GENE2")], hour=11)
        self.cli()
        self.assertEqual(self.summary()["total_nuclei"], 2)
        self.assertEqual(self.summary()["n_runs"], 1)
        self.cli("--reruns", "earliest", "--overwrite")
        self.assertEqual(self.summary()["total_nuclei"], 1)
        self.cli("--reruns", "error", "--overwrite", ok=False)

    def test_dates_plates_and_wells_keep_separate_groups_and_filters(self):
        self.make_run("one")
        self.make_run("two", plate="P2")
        self.make_run("three", date="2026-09-29")
        self.make_run("four", well="C3")
        self.cli()
        self.assertEqual(self.summary()["total_nuclei"], 4)
        groups = self.csv_rows(self.output / "combined_all_groups.csv")
        wells = [r for r in groups if r["group_level"] == "well"]
        self.assertEqual(len(wells), 4)
        self.assertEqual({r["group_level"] for r in groups}, {"all", "experiment", "plate", "well"})
        self.cli("--well", "b2")
        self.assertEqual(self.summary("B2")["total_nuclei"], 3)
        self.cli("--well", "B2", "--plate", "P1", "--experiment-date", "2026-09-28", "--overwrite")
        self.assertEqual(self.summary("B2")["total_nuclei"], 1)

    def test_incomplete_run_does_not_require_its_unwritten_files(self):
        self.make_run("valid")
        incomplete = self.make_run("unfinished", tile="002", status="started")
        (incomplete / "unfinished_run_summary.json").unlink()
        (incomplete / "unfinished_nucleus_barcode_calls.csv").unlink()
        self.cli()
        self.assertEqual(self.summary()["n_runs"], 1)
        self.assertEqual(len(self.csv_rows(self.output / "combined_all_skipped_runs.csv")), 1)

    def test_complete_missing_or_corrupt_export_fails(self):
        folder = self.make_run("valid")
        path = folder / "valid_nucleus_barcode_calls.csv"
        original = path.read_bytes()
        path.unlink()
        self.cli(ok=False)
        path.write_text("not,the,required,columns\n1,2,3,4\n")
        self.cli(ok=False)
        path.write_bytes(original)
        (folder / "valid_run_summary.json").write_text("{broken json")
        self.cli(ok=False)

    def test_csv_summary_count_disagreement_fails(self):
        folder = self.make_run("valid")
        path = folder / "valid_run_summary.json"
        report = json.loads(path.read_text())
        report["total_nuclei"] += 1
        self.write_json(path, report)
        self.cli(ok=False)

    def test_model_and_scientific_setting_differences_require_override(self):
        self.make_run("one")
        folder = self.make_run("two", tile="002")
        path = folder / "two_run_summary.json"
        original = json.loads(path.read_text())
        for change in ("model", "diameter"):
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                if change == "model":
                    report["model_and_codebook_sha256"]["/old/models/base_caller.joblib"] = "d" * 64
                else:
                    report["config"]["manual_analysis"]["nuclear_diameter"] = 60
                self.write_json(path, report)
                self.cli("--overwrite", ok=False)
                self.cli("--allow-mixed-analysis", "--overwrite")
                self.assertEqual(self.summary()["total_nuclei"], 2)

    def test_identical_models_moved_to_another_host_remain_compatible(self):
        self.make_run("one")
        folder = self.make_run("two", tile="002")
        path = folder / "two_run_summary.json"
        report = json.loads(path.read_text())
        report["model_and_codebook_sha256"] = {
            name.replace("/old/", "/cluster/project/"): digest
            for name, digest in report["model_and_codebook_sha256"].items()
        }
        for key in ("base_model_path", "blob_model_path"):
            report["config"]["manual_analysis"][key] = report["config"]["manual_analysis"][key].replace("/old/", "/cluster/project/")
        report["config"]["decoding"]["codebook_path"] = "/cluster/project/library.csv"
        report["config"]["output"]["output_dir"] = "/cluster/output"
        self.write_json(path, report)
        self.cli()
        self.assertEqual(self.summary()["n_runs"], 2)

    def test_empty_completed_image_can_omit_barcode_csv(self):
        folder = self.make_run("empty", [])
        (folder / "empty_nucleus_barcode_calls.csv").unlink()
        path = folder / "empty_run_manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["saved_files"]["barcode_calls_csv"]
        self.write_json(path, manifest)
        self.cli()
        overall = self.summary()
        self.assertEqual(overall["n_runs"], 1)
        for field in ("total_nuclei", "unique_assigned_genes", "unique_assigned_barcodes"):
            self.assertEqual(overall[field], 0, field)
        for field in ("percent_signal_positive", "percent_gene_assigned",
                      "percent_signal_gene_assigned", "percent_signal_nearest_hd_le1"):
            self.assertIsNone(overall[field], field)
        self.assertEqual(self.csv_rows(self.output / "combined_all_genes.csv"), [])

    def test_existing_output_is_preserved_without_overwrite(self):
        self.make_run("valid")
        sentinel = self.output / "combined_all_summary.md"
        sentinel.write_text("KEEP THIS REPORT")
        self.cli(ok=False)
        self.assertEqual(sentinel.read_text(), "KEEP THIS REPORT")
        self.cli("--overwrite")
        self.assertNotEqual(sentinel.read_text(), "KEEP THIS REPORT")

    def test_active_publication_lock_blocks_overwrite_and_preserves_files(self):
        self.make_run("valid")
        self.cli()
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        lock = self.output / ".combined_all.lock"
        lock_contents = "Owned by another publication; do not remove.\n"
        lock.write_text(lock_contents)
        result = self.cli("--overwrite", ok=False)
        self.assertIn("publication holds", result.stderr)
        self.assertEqual(lock.read_text(), lock_contents)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.output.iterdir() if path != lock},
            before,
        )

    def test_invalid_accepted_read_or_impossible_distance_fails(self):
        bad_character = cell()
        bad_character["called_sequence"] = "ACGTXCGTACGT"
        empty_sequence = cell()
        empty_sequence["called_sequence"] = ""
        impossible_distance = cell(hd=1, nearest=2)
        for name, row in (("bad_character", bad_character),
                          ("empty_sequence", empty_sequence),
                          ("impossible_distance", impossible_distance)):
            with self.subTest(case=name):
                # make_run derives count/histogram totals from each row, so the
                # failure must be scientific row validation, not stale totals.
                folder = self.make_run(name, [row])
                result = self.cli(ok=False)
                self.assertIn("invalid accepted barcode/read/distance", result.stderr)
                shutil.rmtree(folder)

    def test_same_field_with_conflicting_image_filename_fails(self):
        self.make_run("one", source="image_one.tif", hour=10)
        self.make_run("two", source="image_two.tif", hour=11)
        self.cli(ok=False)

    def test_empty_folder_or_empty_filter_fails(self):
        self.cli(ok=False)
        self.make_run("valid")
        self.cli("--well", "H12", ok=False)


if __name__ == "__main__":
    unittest.main()
