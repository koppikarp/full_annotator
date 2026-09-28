"""Audit complete manual reads and grouped holdout predictions without loading images.

The deployed model is fitted on all annotations. Its regenerated-feature results
are reconstruction diagnostics, not an independent accuracy estimate. The saved
grouped holdout predictions provide the separate out-of-sample comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "popcrop-matplotlib"))

import numpy as np
import pandas as pd

from popcrop.color_calling.caller import ModelBasedBaseCaller
from popcrop.decoding import BarcodeDecoder, call_bases_from_probabilities
from popcrop.io.codebook import BASE_TO_COLOR, Codebook, reverse_complement


IDENTITY = ["tiff_filename", "well", "tile", "cell"]


def nearest_reference(sequence, codebook):
    if "N" in sequence:
        return {"nearest_hamming_distance": None, "nearest_barcodes": [], "nearest_genes": []}
    expected = np.asarray([list(barcode) for barcode in codebook.target_barcodes])
    distances = np.count_nonzero(expected != np.asarray(list(sequence)), axis=1)
    closest = np.flatnonzero(distances == distances.min())
    return {
        "nearest_hamming_distance": int(distances.min()),
        "nearest_barcodes": [codebook.barcodes[i] for i in closest],
        "nearest_genes": list(dict.fromkeys(gene for i in closest for gene in codebook.gene_candidates[i])),
    }


def compare_complete_reads(data, codebook, target_column, probability_column):
    decoder = BarcodeDecoder(codebook)
    rows, cycle_rows = [], []
    required_cycles = set(range(1, codebook.barcode_length + 1))
    for key, group in data.groupby(IDENTITY, dropna=True, sort=True):
        if set(group["cycle"]) != required_cycles:
            continue
        if len(group) != len(required_cycles):
            raise ValueError(f"Duplicate annotated cycles in complete read {key}; resolve before validation")
        group = group.sort_values("cycle")
        observed = "".join(group[target_column].astype(str))
        probabilities = group[probability_column].tolist()
        predicted = call_bases_from_probabilities(probabilities)
        result = decoder.decode_component(probabilities)
        nearest = nearest_reference(observed, codebook)
        identity = dict(zip(IDENTITY, key))
        rows.append({
            **identity, "x": float(group["x"].iloc[0]), "y": float(group["y"].iloc[0]),
            "manual_sequence": observed, "manual_library_orientation": reverse_complement(observed),
            "manual_nearest_hamming_distance": nearest["nearest_hamming_distance"],
            "manual_nearest_barcodes": nearest["nearest_barcodes"],
            "manual_nearest_genes": nearest["nearest_genes"],
            "prediction_vs_manual_hamming_distance": sum(a != b for a, b in zip(observed, predicted)),
            **result,
        })
        for (_, row), base, probability in zip(group.iterrows(), predicted, probabilities):
            target = str(row[target_column])
            cycle_rows.append({
                **identity, "cycle": int(row["cycle"]), "manual_base": target,
                "predicted_base": base, "agrees_with_manual": base == target,
                **{f"p_{base_name}": probability[color] for base_name, color in BASE_TO_COLOR.items()},
            })
    return pd.DataFrame(rows), pd.DataFrame(cycle_rows)


def save_csv(data, path):
    export = data.copy()
    for column in export:
        export[column] = export[column].map(
            lambda value: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
        )
    export.to_csv(path, index=False)


def histogram(data, column):
    if data.empty:
        return {}
    return {str(int(key)): int(value) for key, value in data[column].dropna().value_counts().sort_index().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/manual_parity")
    parser.add_argument("--codebook", type=Path, default=ROOT / "data/library/v4.3_library_barcodes.csv")
    parser.add_argument("--barcode-column", default="barcode_new")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/manual_read_validation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    codebook = Codebook.from_file(args.codebook, barcode_col=args.barcode_column)
    features = pd.read_parquet(args.model_dir / "regenerated_base_features.parquet")
    model = ModelBasedBaseCaller(args.model_dir / "base_caller.joblib", strict_features=True)
    features["probabilities"] = model.predict_cycle_probabilities(features)
    reads, cycles = compare_complete_reads(features, codebook, "annotated_base", "probabilities")
    save_csv(reads, args.output_dir / "complete_manual_reads_reconstruction.csv")
    save_csv(cycles, args.output_dir / "complete_manual_cycles_reconstruction.csv")
    summary = {
        "barcode_column": codebook.barcode_col,
        "library_rows": len(codebook.df), "unique_library_sequences": len(codebook),
        "sequence_gene_collisions": sum(len(genes) > 1 for genes in codebook.gene_candidates),
        "reconstruction_scope": "Deployed all-data model; these are training-data reconstruction diagnostics, not independent validation.",
        "complete_manual_reads": len(reads), "complete_manual_cycles": len(cycles),
        "manual_nearest_hamming_histogram": histogram(reads, "manual_nearest_hamming_distance"),
        "reconstructed_nearest_hamming_histogram": histogram(reads, "nearest_hamming_distance"),
        "reconstructed_vs_manual_read_hamming_histogram": histogram(reads, "prediction_vs_manual_hamming_distance"),
        "reconstructed_cycle_agreement": float(cycles.agrees_with_manual.mean()) if len(cycles) else None,
        "grouped_holdout_metrics": model.metadata.get("metrics", model.metadata.get("evaluation_metrics", {})),
    }
    holdout_path = args.model_dir / "base_holdout_predictions.csv"
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path)
        missing_identity = [column for column in IDENTITY if column not in holdout]
        if missing_identity:
            # Earlier training runs saved the coordinates but omitted the image
            # identity columns. Recover only through a unique, verified join.
            join = ["cell", "cycle", "x", "y"]
            holdout = holdout.merge(features[join + missing_identity], on=join,
                                    how="left", validate="one_to_one")
            if holdout[missing_identity].isna().any().any():
                raise ValueError("Holdout rows do not uniquely identify regenerated annotation rows")
        target_column = "target" if "target" in holdout else "annotated_base"
        holdout["probabilities"] = [
            {color: float(row.get(f"p_{base}", 0.0)) for base, color in BASE_TO_COLOR.items()}
            for row in holdout.to_dict("records")
        ]
        held_reads, held_cycles = compare_complete_reads(holdout, codebook, target_column, "probabilities")
        save_csv(held_reads, args.output_dir / "complete_manual_reads_grouped_holdout.csv")
        save_csv(held_cycles, args.output_dir / "complete_manual_cycles_grouped_holdout.csv")
        summary.update({
            "grouped_holdout_complete_reads": len(held_reads),
            "grouped_holdout_complete_read_hamming_histogram": histogram(held_reads, "prediction_vs_manual_hamming_distance"),
            "grouped_holdout_nearest_library_hamming_histogram": histogram(held_reads, "nearest_hamming_distance"),
            "grouped_holdout_complete_cycle_agreement": float(held_cycles.agrees_with_manual.mean()) if len(held_cycles) else None,
            "grouped_holdout_all_cycle_agreement": float((holdout[target_column].astype(str) == holdout.predicted.astype(str)).mean()),
            "grouped_holdout_all_cycles": len(holdout),
        })
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
