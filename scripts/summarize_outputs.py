#!/usr/bin/env python3
"""Combine completed POPCROP runs without counting copied outputs or reruns twice.

Only Python's standard library is required. Read manifests, run_summary.json and
nucleus_barcode_calls.csv; do not load images, masks or per-cycle measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


IDENTITY = ("run_id", "source_image", "source_path", "plate", "well", "tile",
            "experiment_date", "run_started_utc", "position_idx")
COUNT_FIELDS = ("total_nuclei", "nuclei_with_detector_spots", "nuclei_with_classified_signal",
                "nuclei_without_classified_signal", "accepted_library_assignments",
                "accepted_barcode_sequences", "gene_ambiguous_barcode_assignments",
                "unique_assigned_genes")
HIST_FIELDS = ("decoding_status_counts", "signal_nuclei_nearest_hamming_histogram",
               "accepted_assignments_selected_hamming_histogram")
GROUP_COLUMNS = ("group_level", "experiment_date", "plate", "well", "run_id")
LIMITATIONS = [
    "Spatial overlap between tiles is not corrected. Nucleus counts are observations across retained fields, not guaranteed unique biological cells.",
    "Field identity uses experiment date, plate, well, tile and position index. Existing outputs lack image content hashes and stage coordinates; renamed copies or incorrect field metadata cannot always be recognized.",
    "Compatibility checks cover recorded model/library hashes and analysis settings, not unrecorded software revisions or externally supplied segmentation masks.",
    "Gene names are preserved exactly as assigned by the library. Gene aliases are not merged. Ambiguous gene assignments are excluded from gene abundance.",
]


class InputError(ValueError):
    """An input is incomplete, inconsistent or unsafe to combine silently."""


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise InputError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"Expected a JSON object: {path}")
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def text(value):
    return "" if value is None else str(value)


def integer(value, name, *, optional=False):
    if optional and text(value).strip() == "":
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise InputError(f"{name}: expected a nonnegative integer, got {value!r}") from exc
    if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
        raise InputError(f"{name}: expected a nonnegative integer, got {value!r}")
    return int(parsed)


def boolean(value, name):
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise InputError(f"{name}: expected True/False, got {value!r}")


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone is missing")
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise InputError(f"Invalid run_started_utc: {value!r}") from exc


def relative_export(manifest_path, manifest, key, *, optional=False):
    """Use the local sibling, never a stale absolute path on the producing host."""
    recorded = manifest.get("saved_files", {}).get(key)
    if not recorded:
        if optional:
            return None
        raise InputError(f"{manifest_path}: missing saved_files.{key}")
    name = str(recorded).replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        raise InputError(f"{manifest_path}: invalid filename for {key}")
    path = manifest_path.parent / name
    if not path.is_file():
        raise InputError(f"{manifest_path}: missing sibling export {name}. Copy the run's manifest, summary and barcode CSV together.")
    return path


def check_identity(observed, expected, source):
    for key in IDENTITY:
        if key not in observed:
            raise InputError(f"{source}: missing identity field {key}")
        if key == "position_idx":
            equal = integer(observed[key], key) == integer(expected[key], key)
        else:
            equal = text(observed[key]) == text(expected[key])
        if not equal:
            raise InputError(f"{source}: {key} disagrees with the run manifest")


@dataclass
class Stats:
    counts: Counter = field(default_factory=Counter)
    statuses: Counter = field(default_factory=Counter)
    nearest: Counter = field(default_factory=Counter)
    selected: Counter = field(default_factory=Counter)
    genes: Counter = field(default_factory=Counter)
    barcodes: set = field(default_factory=set)
    gene_runs: dict = field(default_factory=lambda: defaultdict(set))
    run_ids: set = field(default_factory=set)

    def add(self, other):
        self.counts.update(other.counts)
        self.statuses.update(other.statuses)
        self.nearest.update(other.nearest)
        self.selected.update(other.selected)
        self.genes.update(other.genes)
        self.barcodes.update(other.barcodes)
        self.run_ids.update(other.run_ids)
        for gene, runs in other.gene_runs.items():
            self.gene_runs[gene].update(runs)

    def record(self):
        result = {name: self.counts[name] for name in COUNT_FIELDS}
        result["unique_assigned_genes"] = len(self.genes)
        result.update(n_runs=len(self.run_ids), unique_assigned_barcodes=len(self.barcodes),
                      decoding_status_counts=dict(sorted(self.statuses.items())),
                      signal_nuclei_nearest_hamming_histogram=dict(sorted(self.nearest.items(), key=lambda item: int(item[0]))),
                      accepted_assignments_selected_hamming_histogram=dict(sorted(self.selected.items(), key=lambda item: int(item[0]))))
        total = result["total_nuclei"]
        signal = result["nuclei_with_classified_signal"]
        matched = result["accepted_library_assignments"]
        result.update(percent_signal_positive=percent(signal, total),
                      percent_gene_assigned=percent(matched, total),
                      percent_signal_gene_assigned=percent(matched, signal),
                      percent_signal_nearest_hd_le1=percent(self.nearest["0"] + self.nearest["1"], signal))
        return result


def percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


@contextmanager
def publication_lock(path):
    """Reserve this output label, including for explicit overwrite operations."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise InputError(f"Another report publication holds {path}. Retry after it finishes; remove a stale lock only after confirming its process has stopped.") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


@dataclass
class Run:
    path: Path
    manifest: dict
    summary_path: Path
    summary: dict
    csv_path: Path | None
    stats: Stats | None = None
    signature: dict = field(default_factory=dict)

    @property
    def run_id(self):
        return self.manifest["run_id"]

    @property
    def field_key(self):
        m = self.manifest
        return (text(m["experiment_date"]), m["plate"], m["well"].upper(), m["tile"],
                integer(m["position_idx"], "position_idx"))


def discover(root, args):
    runs, skipped = [], []
    manifests = sorted(root.rglob("*_run_manifest.json"))
    if not manifests:
        raise InputError(f"No *_run_manifest.json files under {root}. Use outputs from the current pipeline, preserving each run's files together.")
    for path in manifests:
        manifest = read_json(path)
        reason = None
        if manifest.get("status") != "complete":
            reason = "incomplete_run"
        elif args.well != ["all"] and text(manifest.get("well")).upper() not in args.well:
            reason = "well_filter"
        elif args.plate and manifest.get("plate") not in args.plate:
            reason = "plate_filter"
        elif args.experiment_date and manifest.get("experiment_date") not in args.experiment_date:
            reason = "experiment_date_filter"
        if reason:
            skipped.append(skip_record(path, manifest, reason))
            continue
        for name in IDENTITY:
            if name not in manifest or (name != "experiment_date" and text(manifest[name]) == ""):
                raise InputError(f"{path}: missing run identity {name}")
        if manifest["experiment_date"]:
            try:
                date.fromisoformat(manifest["experiment_date"])
            except ValueError as exc:
                raise InputError(f"{path}: invalid experiment_date") from exc
        timestamp(manifest["run_started_utc"])
        summary_path = relative_export(path, manifest, "summary_json")
        summary = read_json(summary_path)
        check_identity(summary.get("run", {}), manifest, summary_path)
        total = integer(summary.get("total_nuclei"), "total_nuclei")
        csv_path = relative_export(path, manifest, "barcode_calls_csv", optional=total == 0)
        runs.append(Run(path, manifest, summary_path, summary, csv_path))
    if not runs:
        raise InputError("No completed runs match the requested wells/plates/dates. No report was written.")
    return runs, skipped, len(manifests)


def skip_record(path, manifest, reason, kept_run_id=""):
    return {"manifest": str(path), **{key: manifest.get(key, "") for key in IDENTITY},
            "reason": reason, "kept_run_id": kept_run_id}


def select_runs(runs, skipped, policy):
    by_id = {}
    for run in runs:
        previous = by_id.get(run.run_id)
        if previous is not None:
            same_identity = all(run.manifest.get(key) == previous.manifest.get(key) for key in IDENTITY)
            same_summary = run.summary == previous.summary
            same_csv = ((run.csv_path is None and previous.csv_path is None) or
                        (run.csv_path is not None and previous.csv_path is not None and
                         digest_file(run.csv_path) == digest_file(previous.csv_path)))
            if not (same_identity and same_summary and same_csv):
                raise InputError(f"Conflicting copies of run_id {run.run_id}: {previous.path} and {run.path}")
            skipped.append(skip_record(run.path, run.manifest, "duplicate_run_copy", previous.run_id))
        else:
            by_id[run.run_id] = run
    fields = defaultdict(list)
    for run in by_id.values():
        fields[run.field_key].append(run)
    chosen = []
    for key, candidates in sorted(fields.items()):
        sources = {tuple(run.manifest.get("source_images") or [run.manifest["source_image"]]) for run in candidates}
        if len(sources) > 1:
            raise InputError(f"Conflicting source images for field {key}. Correct the tile/date/plate metadata or select the intended input folders; these cannot safely be treated as reruns.")
        if len(candidates) > 1 and policy == "error":
            raise InputError(f"Repeated field {key}: {len(candidates)} completed reruns. Use --reruns latest or earliest to select one.")
        candidates.sort(key=lambda run: (timestamp(run.manifest["run_started_utc"]), run.run_id))
        keep = candidates[0] if policy == "earliest" else candidates[-1]
        for run in candidates:
            if run is not keep:
                skipped.append(skip_record(run.path, run.manifest, "superseded_rerun", keep.run_id))
        chosen.append(keep)
    return chosen


def analysis_signature(run):
    config = run.summary.get("config")
    hashes = run.summary.get("model_and_codebook_sha256")
    if not isinstance(config, dict) or config.get("analysis_mode") != "manual":
        raise InputError(f"{run.summary_path}: this collector requires the current manual-compatible analysis mode")
    if not isinstance(hashes, dict):
        raise InputError(f"{run.summary_path}: missing model/library fingerprints")
    signature = {key: config.get(key) for key in
                 ("analysis_mode", "channel_names", "has_dapi_cycle", "random_seed", "preprocessing")}
    signature["manual_analysis"] = {key: value for key, value in config.get("manual_analysis", {}).items()
                                    if key not in ("batch_size", "base_model_path", "blob_model_path")}
    signature["decoding"] = {key: value for key, value in config.get("decoding", {}).items()
                             if key != "codebook_path"}
    signature["fingerprints"] = {}
    for role, section, key in (("codebook", "decoding", "codebook_path"),
                               ("base_model", "manual_analysis", "base_model_path"),
                               ("blob_model", "manual_analysis", "blob_model_path")):
        recorded = config.get(section, {}).get(key)
        if not recorded:
            if role != "blob_model":
                raise InputError(f"{run.summary_path}: missing configured {role}")
            signature["fingerprints"][role] = None
            continue
        digest = hashes.get(recorded)
        if digest is None:
            name = str(recorded).replace("\\", "/").rsplit("/", 1)[-1]
            candidates = {value for path, value in hashes.items()
                          if path.replace("\\", "/").rsplit("/", 1)[-1] == name}
            if len(candidates) == 1:
                digest = candidates.pop()
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise InputError(f"{run.summary_path}: cannot identify SHA-256 for {role}")
        signature["fingerprints"][role] = digest.lower()
    return signature


def read_nuclei(run):
    stats = Stats()
    stats.run_ids.add(run.run_id)
    if run.csv_path is not None:
        required = {*IDENTITY, "nucleus_id", "nucleus_uid", "has_detectable_spot", "has_signal",
                    "assigned_gene", "assigned_barcode", "decoding_status", "hamming_distance",
                    "nearest_hamming_distance", "called_sequence"}
        seen = set()
        with run.csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise InputError(f"{run.csv_path}: missing or duplicate CSV headers")
            missing = required - set(reader.fieldnames)
            if missing:
                raise InputError(f"{run.csv_path}: missing columns {sorted(missing)}")
            for number, row in enumerate(reader, start=2):
                location = f"{run.csv_path}:{number}"
                if None in row or any(value is None for value in row.values()):
                    raise InputError(f"{location}: malformed CSV row")
                check_identity(row, run.manifest, location)
                nucleus_id = integer(row["nucleus_id"], f"{location} nucleus_id")
                if nucleus_id in seen:
                    raise InputError(f"{location}: duplicate nucleus_id {nucleus_id} within a run")
                seen.add(nucleus_id)
                if row["nucleus_uid"] != f"{run.run_id}:{nucleus_id}":
                    raise InputError(f"{location}: nucleus_uid disagrees with run_id/nucleus_id")
                has_spot = boolean(row["has_detectable_spot"], location)
                has_signal = boolean(row["has_signal"], location)
                status = row["decoding_status"]
                if not status:
                    raise InputError(f"{location}: missing decoding_status")
                if (status == "no_signal") != (not has_signal):
                    raise InputError(f"{location}: decoding_status and has_signal disagree")
                stats.counts["total_nuclei"] += 1
                stats.counts["nuclei_with_detector_spots"] += has_spot
                stats.counts["nuclei_with_classified_signal"] += has_signal
                stats.counts["nuclei_without_classified_signal"] += not has_signal
                stats.statuses[status] += 1
                nearest = integer(row["nearest_hamming_distance"], location, optional=True)
                distance = integer(row["hamming_distance"], location, optional=True)
                if has_signal and nearest is not None:
                    stats.nearest[str(nearest)] += 1
                if status in ("matched", "ambiguous_gene"):
                    if (not re.fullmatch(r"[ACGT]+", row["assigned_barcode"])
                            or not re.fullmatch(r"[ACGT]+", row["called_sequence"])
                            or len(row["assigned_barcode"]) != len(row["called_sequence"])
                            or distance is None or nearest is None
                            or not 0 <= nearest <= distance <= len(row["called_sequence"])):
                        raise InputError(f"{location}: invalid accepted barcode/read/distance")
                    stats.counts["accepted_barcode_sequences"] += 1
                    stats.barcodes.add(row["assigned_barcode"])
                    if status == "matched":
                        gene = row["assigned_gene"]
                        if not gene.strip() or gene in ("NONE", "UNASSIGNED"):
                            raise InputError(f"{location}: matched nucleus has no assigned gene")
                        stats.genes[gene] += 1
                        stats.gene_runs[gene].add(run.run_id)
                        stats.counts["accepted_library_assignments"] += 1
                        stats.selected[str(distance)] += 1
                    else:
                        stats.counts["gene_ambiguous_barcode_assignments"] += 1
    observed = stats.record()
    for name in COUNT_FIELDS:
        expected = integer(run.summary.get(name), f"{run.summary_path}: {name}")
        if observed[name] != expected:
            raise InputError(f"{run.summary_path}: {name}={expected} disagrees with nucleus CSV ({observed[name]})")
    for name in HIST_FIELDS:
        raw = run.summary.get(name)
        if not isinstance(raw, dict):
            raise InputError(f"{run.summary_path}: missing {name}")
        expected = {str(key): integer(value, name) for key, value in raw.items() if value != 0}
        if observed[name] != expected:
            raise InputError(f"{run.summary_path}: {name} disagrees with nucleus CSV")
    run.stats = stats


def group_keys(run):
    m = run.manifest
    d, p, w = text(m["experiment_date"]), m["plate"], m["well"].upper()
    return [("all", "", "", "", ""), ("experiment", d, "", "", ""),
            ("plate", d, p, "", ""), ("well", d, p, w, "")]


def group_identity(key):
    return dict(zip(GROUP_COLUMNS, key))


def gene_rows(stats):
    for gene, count in sorted(stats.genes.items(), key=lambda item: (-item[1], item[0])):
        yield {"gene": gene, "cell_count": count,
               "percent_mapped": percent(count, stats.counts["accepted_library_assignments"]),
               "percent_total": percent(count, stats.counts["total_nuclei"]),
               "n_runs": len(stats.gene_runs[gene])}


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key in columns})


def markdown_report(report):
    s = report["overall"]
    lines = ["# Combined optical pooled screening report", "", f"Selection: {report['selection_label']}",
             f"Input folder: {report['input_directory']}", f"Created (UTC): {report['created_utc']}", "",
             f"Included fields/runs: {s['n_runs']:,}",
             f"Nuclei segmented: {s['total_nuclei']:,}",
             f"Nuclei with detector spots: {s['nuclei_with_detector_spots']:,}",
             f"Nuclei with classified signal: {s['nuclei_with_classified_signal']:,}",
             f"Nuclei without classified signal: {s['nuclei_without_classified_signal']:,}",
             f"Accepted gene assignments: {s['accepted_library_assignments']:,}",
             f"Barcode assignments ambiguous between genes: {s['gene_ambiguous_barcode_assignments']:,}",
             f"Unique assigned genes: {s['unique_assigned_genes']:,}",
             f"Unique accepted barcode sequences: {s['unique_assigned_barcodes']:,}", ""]
    for title, name in (("Signal-positive / all nuclei", "percent_signal_positive"),
                        ("Gene-assigned / all nuclei", "percent_gene_assigned"),
                        ("Gene-assigned / signal-positive nuclei", "percent_signal_gene_assigned"),
                        ("Nearest library distance 0 or 1 / signal-positive nuclei", "percent_signal_nearest_hd_le1")):
        value = s[name]
        lines.append(f"{title}: {value:.2f}%" if value is not None else f"{title}: not applicable (zero denominator)")
    lines += ["", "Nearest Hamming distances among signal-positive nuclei: " + json.dumps(s[HIST_FIELDS[1]]) + ".",
              "Hamming distances of accepted gene assignments: " + json.dumps(s[HIST_FIELDS[2]]) + ".",
              "Decoding statuses: " + json.dumps(s[HIST_FIELDS[0]]) + ".", "", "## Plate and well totals", ""]
    for group in report["groups"]:
        if group["group_level"] != "well":
            continue
        lines.append(f"{group['experiment_date'] or 'Date unspecified'} / {group['plate']} / {group['well']}: "
                     f"{group['n_runs']:,} fields; {group['total_nuclei']:,} nuclei; "
                     f"{group['nuclei_with_classified_signal']:,} signal-positive; "
                     f"{group['accepted_library_assignments']:,} gene assignments; "
                     f"{group['unique_assigned_genes']:,} unique genes.")
    lines += ["", "## Selection and deduplication", "",
              f"Rerun policy: {report['rerun_policy']} completed run per field, ranked by run_started_utc (not match rate).",
              "Excluded manifests: " + json.dumps(report["excluded_reason_counts"], sort_keys=True) + ".",
              "Copied run IDs are counted once. Different nuclei sharing a gene/barcode remain separate observations. Gene totals use set union, and abundances sum accepted nucleus counts.",
              "Percentages are recalculated from combined counts, never averaged across images. Date/plate/well groups retain their identities; do not sum rows from different group levels.",
              "", "## Limits and QC", "", *report["warnings"], *LIMITATIONS, "", "## Output files", ""]
    lines.extend(f"{name}: {path}" for name, path in report["outputs"].items())
    return "\n\n".join(line for line in lines if line != "") + "\n"


def summarize(args):
    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Input folder does not exist: {root}")
    runs, skipped, discovered = discover(root, args)
    runs = select_runs(runs, skipped, args.reruns)
    signatures = {}
    warnings = []
    for run in runs:
        run.signature = analysis_signature(run)
        signatures.setdefault(canonical_digest(run.signature), run.signature)
    if len(signatures) > 1:
        if not args.allow_mixed_analysis:
            raise InputError("Included runs used different models, libraries or analysis settings. Select compatible runs or explicitly use --allow-mixed-analysis; no report was written.")
        warnings.append(f"MIXED ANALYSIS: {len(signatures)} distinct model/library/settings signatures were combined by explicit request. Interpret combined mapping rates accordingly.")
    if any(not run.manifest["experiment_date"] or any(text(run.manifest[name]).lower() in ("", "unspecified")
           for name in ("plate", "well", "tile")) for run in runs):
        warnings.append("Some experiment dates or field identifiers are unspecified. They are kept in explicit unknown groups; fields from separate acquisitions with identical missing metadata cannot be distinguished.")
    groups = defaultdict(Stats)
    for run in runs:
        read_nuclei(run)
        for key in group_keys(run):
            groups[key].add(run.stats)
    all_stats = groups[("all", "", "", "", "")]
    group_rows = [{**group_identity(key), **stats.record()} for key, stats in sorted(groups.items())]
    label = "all" if args.well == ["all"] else "_".join(args.well)
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip(".-_")[:80] or "selection"
    output = args.out_dir.expanduser().resolve()
    names = {key: output / f"combined_{safe_label}_{suffix}" for key, suffix in {
        "summary_report": "summary.md", "summary_json": "summary.json", "groups": "groups.csv",
        "genes": "genes.csv", "genes_by_group": "gene_counts_by_group.csv", "runs": "runs.csv",
        "skipped_runs": "skipped_runs.csv"}.items()}
    existing = [str(path) for path in names.values() if path.exists()]
    if existing and not args.overwrite:
        raise InputError("Output already exists; choose --out-dir or explicitly --overwrite: " + ", ".join(existing))
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_directory": str(root), "selection_label": label,
        "filters": {"wells": args.well, "plates": args.plate, "experiment_dates": args.experiment_date},
        "rerun_policy": args.reruns, "spatial_overlap_corrected": False,
        "discovered_manifests": discovered, "excluded_reason_counts": dict(Counter(row["reason"] for row in skipped)),
        "overall": all_stats.record(), "groups": group_rows, "warnings": warnings, "limitations": LIMITATIONS,
        "analysis_signatures": signatures, "outputs": {key: str(path) for key, path in names.items()},
    }
    run_rows = []
    for run in runs:
        run_rows.append({**{key: run.manifest[key] for key in IDENTITY}, **run.stats.record(),
                         "manifest": str(run.path), "summary_json": str(run.summary_path),
                         "barcode_calls_csv": str(run.csv_path) if run.csv_path else "",
                         "analysis_signature": canonical_digest(run.signature),
                         "summary_sha256": digest_file(run.summary_path),
                         "barcode_csv_sha256": digest_file(run.csv_path) if run.csv_path else ""})
    report["included_runs"] = run_rows
    report["excluded_runs"] = skipped
    metric_columns = list(all_stats.record())
    gene_columns = ["gene", "cell_count", "percent_mapped", "percent_total", "n_runs"]
    def grouped_genes():
        for key, stats in sorted(groups.items()):
            for row in gene_rows(stats):
                yield {**group_identity(key), **row}
        for run in runs:
            m = run.manifest
            key = ("run", text(m["experiment_date"]), m["plate"], m["well"].upper(), run.run_id)
            for row in gene_rows(run.stats):
                yield {**group_identity(key), **row}
    output.mkdir(parents=True, exist_ok=True)
    # Stage all outputs first. Publish the JSON completion record last. Refuse
    # an existing destination by default, including a concurrent publisher.
    with publication_lock(output / f".combined_{safe_label}.lock"), \
            tempfile.TemporaryDirectory(prefix=".popcrop-summary-", dir=output) as temporary:
        if not args.overwrite and any(path.exists() for path in names.values()):
            raise InputError("Output appeared while this report was being prepared; no existing report was changed.")
        staged = {key: Path(temporary) / path.name for key, path in names.items()}
        write_csv(staged["groups"], group_rows, [*GROUP_COLUMNS, *metric_columns])
        write_csv(staged["genes"], gene_rows(all_stats), gene_columns)
        write_csv(staged["genes_by_group"], grouped_genes(), [*GROUP_COLUMNS, *gene_columns])
        write_csv(staged["runs"], run_rows, [*IDENTITY, *metric_columns, "manifest", "summary_json", "barcode_calls_csv",
                                           "analysis_signature", "summary_sha256", "barcode_csv_sha256"])
        write_csv(staged["skipped_runs"], skipped, ["manifest", *IDENTITY, "reason", "kept_run_id"])
        staged["summary_report"].write_text(markdown_report(report))
        staged["summary_json"].write_text(json.dumps({"status": "complete", **report}, indent=2, allow_nan=False) + "\n")
        if args.overwrite:
            names["summary_json"].unlink(missing_ok=True)
        published = []
        try:
            for key in [name for name in names if name != "summary_json"] + ["summary_json"]:
                if args.overwrite:
                    staged[key].replace(names[key])
                else:
                    os.link(staged[key], names[key])
                published.append(names[key])
        except OSError:
            if not args.overwrite:
                for path in published:
                    path.unlink(missing_ok=True)
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder searched recursively for completed run manifests")
    parser.add_argument("--well", nargs="+", default=["all"], help="all (default), or well names such as B2 B3; case-insensitive")
    parser.add_argument("--plate", nargs="+", help="Optional exact plate identifiers")
    parser.add_argument("--experiment-date", nargs="+", help="Optional acquisition dates, YYYY-MM-DD")
    parser.add_argument("--reruns", choices=("latest", "earliest", "error"), default="latest")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Output folder, default current directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace this selection's existing aggregate reports")
    parser.add_argument("--allow-mixed-analysis", action="store_true", help="Explicitly combine different recorded models/libraries/settings")
    args = parser.parse_args()
    args.well = sorted(set(name.upper() for name in args.well))
    if "ALL" in args.well:
        if len(args.well) != 1:
            parser.error("--well all cannot be combined with individual well names")
        args.well = ["all"]
    for value in args.experiment_date or []:
        try:
            date.fromisoformat(value)
        except ValueError:
            parser.error(f"Invalid experiment date: {value}")
    csv.field_size_limit(64 * 1024 * 1024)
    try:
        report = summarize(args)
    except (InputError, OSError, csv.Error, ValueError, TypeError) as exc:
        parser.exit(1, f"Cannot combine outputs: {exc}\n")
    stats = report["overall"]
    print(f"Included {stats['n_runs']:,} fields: {stats['total_nuclei']:,} nuclei; "
          f"{stats['nuclei_with_classified_signal']:,} signal-positive; "
          f"{stats['accepted_library_assignments']:,} gene assignments; {stats['unique_assigned_genes']:,} unique genes.")
    print(f"Excluded manifests: {json.dumps(report['excluded_reason_counts'], sort_keys=True)}")
    print("Spatial overlap between tiles was not corrected.")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Summary: {report['outputs']['summary_report']}")


if __name__ == "__main__":
    main()
