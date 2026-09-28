"""Multi-panel diagnostic overlays showing nuclei boundaries, blob masks, cores, rims, and color calls."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import find_boundaries
import logging

logger = logging.getLogger(__name__)


def generate_diagnostic_overlays(
    dapi_image: Optional[np.ndarray],
    sbs_stack: np.ndarray,
    nuclear_mask: np.ndarray,
    blob_mask: np.ndarray,
    df_nuclei: Any,
    output_path: str | Path,
    max_cells: int = 12,
    crop_size: int = 100
) -> None:
    """
    Generates high-resolution diagnostic galleries showing individual cells:
      - Raw DAPI & nuclear outline
      - Combined SBS signal & blob mask
      - Core vs Rim separation
      - Decoded barcode and confidence score
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Filter for cells that have detected blobs or calls
    if hasattr(df_nuclei, "query"):
        # Show reproducible examples of each outcome, not only edge nuclei
        # at the beginning of a scan-order label list.
        groups = [g.sample(frac=1, random_state=42) for _,g in df_nuclei.groupby("decoding_status",sort=True)]
        indices = []
        for i in range(max_cells):
            for group in groups:
                if i < len(group) and len(indices) < max_cells:
                    indices.append(group.index[i])
        sample_cells = df_nuclei.loc[indices]
    else:
        sample_cells = df_nuclei[:max_cells]

    if len(sample_cells) == 0:
        logger.info("No nuclei to generate diagnostic overlays for.")
        return

    n_cells = len(sample_cells)
    fig, axes = plt.subplots(n_cells, 4, figsize=(16, 3.5 * n_cells))
    if n_cells == 1:
        axes = axes[None, :]

    H, W = nuclear_mask.shape
    # Composite SBS representation: cycle 0 or mean across cycles of Red & Green

    for idx, (_, row) in enumerate(sample_cells.iterrows()):
        n_id = int(row["nucleus_id"])
        cy, cx = int(row["centroid_y"]), int(row["centroid_x"])

        # Compute bounding box
        half = crop_size // 2
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)

        # 1. DAPI with Nuclear Outline
        ax0 = axes[idx, 0]
        if dapi_image is not None:
            dapi_crop = dapi_image[y0:y1, x0:x1]
            ax0.imshow(dapi_crop, cmap="Blues_r")
        else:
            ax0.imshow(np.zeros((y1 - y0, x1 - x0)), cmap="gray")

        nuc_crop = nuclear_mask[y0:y1, x0:x1] == n_id
        nuc_bound = find_boundaries(nuc_crop, mode="inner")
        ax0.contour(nuc_bound, colors="cyan", linewidths=1.5)
        ax0.set_title(f"Nucleus {n_id}\nMax cycle P(signal)={row['p_real_blob']:.2f}", fontsize=10, fontweight="bold")
        ax0.axis("off")

        # 2. SBS Red / Green Fluorescence Crop
        ax1 = axes[idx, 1]
        rgb_crop = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        r_crop = np.mean(sbs_stack[:, 0, y0:y1, x0:x1], axis=0)
        g_crop = np.mean(sbs_stack[:, 1, y0:y1, x0:x1], axis=0)
        rgb_crop[..., 0] = r_crop / max(1.0, np.percentile(r_crop, 99))
        rgb_crop[..., 1] = g_crop / max(1.0, np.percentile(g_crop, 99))
        ax1.imshow(np.clip(rgb_crop, 0, 1))
        ax1.set_title(f"SBS Red/Green\nStatus: {row['decoding_status']}", fontsize=10)
        ax1.axis("off")

        # 3. Blob Mask & Core/Rim Outlines
        ax2 = axes[idx, 2]
        blob_crop = blob_mask[y0:y1, x0:x1] == n_id if "analysis_version" in row else blob_mask[y0:y1, x0:x1] > 0
        ax2.imshow(np.clip(rgb_crop, 0, 1))
        if np.any(blob_crop):
            blob_bound = find_boundaries(blob_crop, mode="inner")
            ax2.contour(blob_bound, colors="magenta", linewidths=2.0)
        ax2.set_title(f"Target proposals across cycles\nDetected cycles={row.get('detected_cycles', 'N/A')}", fontsize=10)
        ax2.axis("off")

        # 4. Barcode Call Summary
        ax3 = axes[idx, 3]
        ax3.axis("off")
        bc = row.get("best_barcode", "N/A")
        gene = row.get("best_gene", "N/A")
        assigned = row.get("assigned_gene", gene)
        seq = row.get("called_sequence", "N/A")
        margin = row.get("score_margin", 0.0)
        conf = row.get("posterior_confidence", 0.0)

        info_text = (
            f"Assigned gene: {assigned}\n"
            f"Candidate: {gene} ({bc})\n"
            f"Called Bases: {seq}\n"
            f"Selected/nearest HD: {row.get('hamming_distance', 'N/A')}/{row.get('nearest_hamming_distance', 'N/A')}\n"
            f"Margin: {margin:.2f}\n"
            f"Conf: {conf:.1%}\n"
            f"QC: {row.get('qc_flags', 'PASS')}"
        )
        ax3.text(
            0.1, 0.5, info_text, fontsize=11, family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc")
        )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved diagnostic overlay gallery to {path}")
