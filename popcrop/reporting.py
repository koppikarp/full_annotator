"""Auditable nucleus-level counts and unmodified read-to-library distance QC."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
from popcrop.io.run_identity import IDENTITY_COLUMNS


def write_run_report(output_dir, nuclei, components, config, image_metadata, *, prefix=None, run_metadata=None):
    output_dir = Path(output_dir)
    name_prefix = f"{prefix}_" if prefix else ""
    total = len(nuclei)
    def count(column):
        return int(nuclei[column].sum()) if column in nuclei else 0
    statuses = nuclei.decoding_status.value_counts().to_dict() if total else {}
    signal = nuclei[nuclei.has_signal] if "has_signal" in nuclei else nuclei
    matched = nuclei[nuclei.decoding_status == "matched"] if total else nuclei
    hist = signal.nearest_hamming_distance.value_counts().sort_index().to_dict() if "nearest_hamming_distance" in signal else {}
    selected_hist = matched.hamming_distance.value_counts().sort_index().to_dict() if "hamming_distance" in matched else {}
    counts = {
        "total_nuclei": total,
        "nuclei_with_detector_spots": count("has_detectable_spot"),
        "nuclei_with_classified_signal": count("has_signal"),
        "nuclei_without_classified_signal": total-count("has_signal"),
        "accepted_library_assignments": len(matched),
        "accepted_barcode_sequences": len(matched) + statuses.get("ambiguous_gene", 0),
        "gene_ambiguous_barcode_assignments": statuses.get("ambiguous_gene", 0),
        "unique_assigned_genes": int(matched.best_gene.nunique()) if len(matched) else 0,
        "decoding_status_counts": statuses,
        "signal_nuclei_nearest_hamming_histogram": {str(int(k)): int(v) for k,v in hist.items()},
        "accepted_assignments_selected_hamming_histogram": {str(int(k)): int(v) for k,v in selected_hist.items()},
    }
    fingerprints = {}
    paths = [config.decoding.codebook_path]
    if config.analysis_mode == "manual":
        paths += [config.manual_analysis.base_model_path, config.manual_analysis.blob_model_path]
    for name in paths:
        if name and Path(name).is_file():
            fingerprints[str(Path(name).resolve())] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    metadata = {k: v for k, v in image_metadata.items() if k not in ("dapi_stack", "reference_dapi")}
    report = {**counts, "run": run_metadata or {}, "config": config.to_dict(), "image_metadata": metadata,
              "model_and_codebook_sha256": fingerprints, "python_version": platform.python_version(),
              "definitions": {
                  "has_detectable_spot": "At least one cycle has a per-channel detector mask",
                  "has_signal": "At least one cycle has P(A)+P(C)+P(T) at or above the configured threshold",
                  "p_real_blob": "Compatibility column: maximum per-cycle base-model signal score; not a calibrated nucleus probability",
                  "posterior_confidence": "Likelihood support normalized across the entire supplied library; not empirical assignment accuracy",
                  "best_barcode": "Likelihood candidate, including rejected candidates; use assigned_barcode/status for accepted assignments",
                  "no_signal": "No classified detectable signal; does not establish lack of transduction",
                  "blob_mask": "Diagnostic union of raw per-cycle spot masks; overlapping cells use first owner. Probability map is the maximum across overlapping owners. Exact cycle masks are in cycles.parquet",
              }}
    json_path = output_dir / f"{name_prefix}run_summary.json"
    json_path.write_text(json.dumps(report, indent=2, default=str)+"\n")
    lines = ["# Optical pooled screening analysis", "",
             f"Nuclei segmented: {total:,}.",
             f"Nuclei with detector spots: {counts['nuclei_with_detector_spots']:,}.",
             f"Nuclei with classified signal: {counts['nuclei_with_classified_signal']:,}.",
             f"Accepted library assignments: {len(matched):,}.",
             f"Accepted barcode sequences with ambiguous genes: {statuses.get('ambiguous_gene', 0):,}.",
             f"Unique assigned genes: {counts['unique_assigned_genes']:,}.", "",
             "Nearest Hamming distances among signal-positive nuclei: " + json.dumps(counts['signal_nuclei_nearest_hamming_histogram'])+".",
             "Hamming distances for accepted likelihood assignments: " + json.dumps(counts['accepted_assignments_selected_hamming_histogram'])+".", "",
             "Each nucleus has one observed sequence and its reverse complement in the nucleus table. The selected likelihood reference and the nearest Hamming reference are reported separately. Rows with decoding_status=matched have a barcode and unique gene assignment; ambiguous_gene rows identify a barcode with multiple candidate genes and are excluded from gene abundance. N-containing reads are excluded from assignment. All per-cycle raw distributions, exact detector masks, settings and model probabilities are preserved in the cycles table.", "",
             "A negative signal call is an imaging/model result, not a determination that the cell was untransduced. Library-relative likelihood confidence is not a calibrated probability of correctness. Full-field sensitivity requires representative manually reviewed positive and negative nuclei.", ""]
    if run_metadata:
        lines[2:2] = [
            f"Source image: {run_metadata['source_image']}",
            f"Plate: {run_metadata['plate']}; well: {run_metadata['well']}; tile: {run_metadata['tile']}.",
            f"Experiment date: {run_metadata['experiment_date'] or 'unspecified'}; run started (UTC): {run_metadata['run_started_utc']}.",
            f"Run ID: {run_metadata['run_id']}", "",
        ]
    md_path = output_dir / f"{name_prefix}run_summary_report.md"
    md_path.write_text("\n".join(lines))
    gene_path = output_dir / f"{name_prefix}gene_abundance_summary.csv"
    if len(matched):
        genes = matched.best_gene.value_counts().rename_axis("gene").reset_index(name="cell_count")
    else:
        import pandas as pd
        genes = pd.DataFrame(columns=["gene", "cell_count"])
    genes["percent_mapped"] = genes.cell_count/max(1,len(matched))*100
    genes["percent_total"] = genes.cell_count/max(1,total)*100
    if run_metadata:
        for name in reversed(IDENTITY_COLUMNS):
            genes.insert(0, name, run_metadata[name])
    genes.to_csv(gene_path,index=False)
    return {"summary_json": str(json_path), "summary_report": str(md_path), "gene_abundance": str(gene_path)}
