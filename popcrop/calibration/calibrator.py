"""Calibrator for spectral unmixing matrices and core-color distribution parameters."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import logging

from popcrop.calibration.ground_truth_adapter import GroundTruthAdapter

logger = logging.getLogger(__name__)


class PipelineCalibrator:
    """Estimates unmixing matrices, channel ratios, and basecaller parameters from controls."""

    def __init__(self):
        self.calibration_results: Dict[str, Any] = {}

    def calibrate_from_ground_truth(self, adapter: GroundTruthAdapter) -> Dict[str, Any]:
        """
        Calibrates parameters from single-color and dual-color controls in the ground truth adapter.
        """
        stats = adapter.compute_channel_statistics()
        logger.info(f"Calibrating pipeline from base stats: {list(stats.keys())}")

        # 1. Spectral crosstalk estimation
        # Green control (Base A): Green is primary fluorophore. Any red measured is bleedthrough.
        alpha_bleed_green_to_red = 0.0
        if "A" in stats and stats["A"]["count"] > 0:
            a_red = stats["A"]["red_median"]
            a_green = stats["A"]["green_median"]
            alpha_bleed_green_to_red = max(0.0, min(0.35, a_red / max(1.0, a_green)))

        # Red control (Base T): Red is primary fluorophore. Any green measured is bleedthrough.
        beta_bleed_red_to_green = 0.0
        if "T" in stats and stats["T"]["count"] > 0:
            t_red = stats["T"]["red_median"]
            t_green = stats["T"]["green_median"]
            beta_bleed_red_to_green = max(0.0, min(0.35, t_green / max(1.0, t_red)))

        unmixing_matrix = [
            [1.0, float(alpha_bleed_green_to_red)],
            [float(beta_bleed_red_to_green), 1.0]
        ]

        # 2. Expected Green:Red ratio for Yellow / dual-positive (Base C)
        expected_ratio_yellow = 2.5
        if "C" in stats and stats["C"]["count"] > 0:
            expected_ratio_yellow = stats["C"]["green_to_red_ratio"]

        # 3. Dark state noise floor (Base G)
        dark_snr_cutoff = 2.0
        if "G" in stats and stats["G"]["count"] > 0:
            g_red = stats["G"]["red_median"]
            g_green = stats["G"]["green_median"]
            logger.info(f"Dark baseline floor: Red={g_red:.1f}, Green={g_green:.1f}")

        self.calibration_results = {
            "unmixing_matrix": unmixing_matrix,
            "expected_green_red_ratio_yellow": float(expected_ratio_yellow),
            "dark_snr_cutoff": float(dark_snr_cutoff),
            "base_statistics": stats
        }

        logger.info(
            f"Calibration complete: Unmixing matrix={unmixing_matrix}, "
            f"Yellow Green:Red ratio={expected_ratio_yellow:.2f}"
        )
        return self.calibration_results

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.calibration_results, f, indent=2)
        logger.info(f"Saved calibration parameters to {path}")

    def load(self, file_path: str | Path) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with open(path, "r") as f:
            self.calibration_results = json.load(f)
        logger.info(f"Loaded calibration parameters from {path}")
        return self.calibration_results
