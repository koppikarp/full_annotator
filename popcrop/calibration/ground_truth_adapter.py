"""Ground truth data ingestion adapter for point-value intensities and nuclear distributions."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def derive_blob_presence_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Recover channel presence from a judgement of detector correctness.

    Correctness is not presence: a correctly displayed absence is a true negative.
    Keep the original columns intact, including the historically misnamed
    ``is_real_blob``, so the derived training table is auditable.
    """
    df = data.copy()
    status_col = next((c for c in ("call_status", "blob_call_status") if c in df), None)
    if status_col is None and not all(c in df for c in ("green_called_correctly", "red_called_correctly")):
        raise ValueError("Cannot recover presence without per-channel correctness judgements.")

    def boolean_column(name: str) -> pd.Series:
        if name not in df:
            raise ValueError(f"Cannot recover presence: missing {name}.")
        normalized = df[name].astype(str).str.strip().str.lower()
        parsed = normalized.map({"true": True, "false": False, "1": True, "0": False,
                                 "1.0": True, "0.0": False})
        if parsed.isna().any():
            raise ValueError(f"Missing or invalid boolean values in {name}; original detector state is required.")
        return parsed.astype(bool)

    status = df[status_col].astype(str).str.strip().str.lower() if status_col else None
    if status is not None and not status.isin(["both", "only_green", "only_red", "neither"]).all():
        raise ValueError("Unrecognized or missing channel-correctness annotation.")
    for channel, correct_statuses in (("green", ("both", "only_green")), ("red", ("both", "only_red"))):
        detected = boolean_column(f"{channel}_detected")
        correctness_name = f"{channel}_called_correctly"
        correct = boolean_column(correctness_name) if correctness_name in df else status.isin(correct_statuses)
        if status is not None and not (correct == status.isin(correct_statuses)).all():
            raise ValueError(f"{channel} correctness columns disagree with call_status.")
        df[f"{channel}_signal_present"] = detected.eq(correct)
    df["any_signal_present"] = df["green_signal_present"] | df["red_signal_present"]
    df["derived_base"] = [
        {(False, False): "G", (True, False): "A", (False, True): "T", (True, True): "C"}[g, r]
        for g, r in zip(df["green_signal_present"], df["red_signal_present"])
    ]
    df["label_semantics"] = "presence_from_original_detection_and_correctness_v1"
    return df


def annotation_groups(df: pd.DataFrame) -> np.ndarray:
    """Group all cycles of a nucleus together, retaining image/well/tile identity."""
    import json
    def normalized(value):
        if pd.isna(value):
            return None
        # CSV columns containing one absent ID become floats. 459 and 459.0
        # must retain the same identity across independently loaded tables.
        if isinstance(value, (float, int, np.integer, np.floating)):
            return str(int(value)) if float(value).is_integer() else str(value)
        return str(value)
    keys = []
    for index, row in df.iterrows():
        identity = [normalized(row.get(c)) for c in ("tiff_filename", "well", "tile")]
        if pd.notna(row.get("cell")):
            identity.extend(["cell", normalized(row["cell"])])
        elif pd.notna(row.get("x")) and pd.notna(row.get("y")):
            identity.extend(["position", normalized(row["x"]), normalized(row["y"])])
        else:
            identity.extend(["row", str(index)])
        keys.append(json.dumps(identity))
    return np.asarray(keys)


class GroundTruthAdapter:
    """
    Ingests and parses ground truth calibration datasets provided by the user:
      1. Point-value intensities (e.g. single-spot / peak measurements per channel and base).
      2. Nuclear intensity distributions (e.g. histogram / pixel distribution quantiles per base).
    """

    def __init__(self):
        self.samples_per_base: Dict[str, pd.DataFrame] = {}

    def load_point_values(
        self,
        file_path: str | Path,
        base_col: str = "annotated_base",
        red_col: str = "red_intensity",
        green_col: str = "green_intensity"
    ) -> GroundTruthAdapter:
        """
        Loads ground truth from a table containing point measurements.
        Columns: base, red_intensity, green_intensity.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep)

        # Auto-detect column names if not exact match
        if base_col not in df.columns:
            possible = [c for c in df.columns if "base" in c.lower()]
            if possible:
                base_col = possible[0]
        if red_col not in df.columns:
            possible = [c for c in df.columns if "red" in c.lower() or "cy3" in c.lower() or "561" in c.lower()]
            if possible:
                red_col = possible[0]
        if green_col not in df.columns:
            possible = [c for c in df.columns if "green" in c.lower() or "gfp" in c.lower() or "488" in c.lower()]
            if possible:
                green_col = possible[0]

        logger.info(f"Loaded point-value ground truth with columns: base='{base_col}', red='{red_col}', green='{green_col}'")

        df_clean = df[[base_col, red_col, green_col]].dropna().copy()
        df_clean[base_col] = df_clean[base_col].astype(str).str.strip().str.upper()
        df_clean.rename(columns={base_col: "base", red_col: "red", green_col: "green"}, inplace=True)

        for b in ["A", "C", "G", "T"]:
            sub = df_clean[df_clean["base"] == b]
            self.samples_per_base[b] = sub

        return self

    def load_nuclear_distributions(
        self,
        distribution_records: List[Dict[str, Any]]
    ) -> GroundTruthAdapter:
        """
        Loads ground truth formatted as nuclear distributions.
        Each record has:
          {'base': 'A', 'red_values': [...], 'green_values': [...]}
        Extracts robust upper quantiles (e.g. 85th percentile / core) for each nucleus.
        """
        rows = []
        for rec in distribution_records:
            b = str(rec["base"]).strip().upper()
            r_vals = np.asarray(rec["red_values"])
            g_vals = np.asarray(rec["green_values"])
            if len(r_vals) == 0 or len(g_vals) == 0:
                continue

            # Extract core signal: 85th percentile of intensity distribution
            r_core = float(np.percentile(r_vals, 85))
            g_core = float(np.percentile(g_vals, 85))
            rows.append({"base": b, "red": r_core, "green": g_core})

        df_dist = pd.DataFrame(rows)
        for b in ["A", "C", "G", "T"]:
            self.samples_per_base[b] = df_dist[df_dist["base"] == b]

        return self

    def compute_channel_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Computes empirical statistics (median, mean, std, ratio) for each base state.
        """
        stats = {}
        for b, df in self.samples_per_base.items():
            if df.empty:
                continue
            r = df["red"].values.astype(np.float32)
            g = df["green"].values.astype(np.float32)
            stats[b] = {
                "count": len(df),
                "red_median": float(np.median(r)),
                "red_mean": float(np.mean(r)),
                "red_std": float(np.std(r)),
                "green_median": float(np.median(g)),
                "green_mean": float(np.mean(g)),
                "green_std": float(np.std(g)),
                "green_to_red_ratio": float(np.median(g) / max(1e-2, np.median(r))),
            }
        return stats

    def load_base_ground_truth(self, file_path: str | Path) -> GroundTruthAdapter:
        """Loads ground truth exported by the base caller annotator."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Base ground truth file not found: {path}")
        sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep)
        
        base_col = "annotated_base" if "annotated_base" in df.columns else "base"
        red_col = "red_trimmed_mean" if "red_trimmed_mean" in df.columns else ("red_median" if "red_median" in df.columns else "red")
        green_col = "green_trimmed_mean" if "green_trimmed_mean" in df.columns else ("green_median" if "green_median" in df.columns else "green")
        
        return self.load_point_values(file_path, base_col=base_col, red_col=red_col, green_col=green_col)

    def load_blob_ground_truth(self, file_path: str | Path) -> pd.DataFrame:
        """Load source records and expose corrected presence alongside original labels."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Blob ground truth file not found: {path}")
        sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep)
        if "call_status" in df or "blob_call_status" in df or "green_called_correctly" in df:
            return derive_blob_presence_labels(df)
        return df
