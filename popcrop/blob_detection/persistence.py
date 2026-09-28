"""Cross-cycle top-k energy aggregation and temporal persistence counting."""

from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


def aggregate_cross_cycle_signal(
    cycle_energies: np.ndarray,
    top_k: int = 3,
    persistence_threshold: float = 2.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines cycle energies across time into robust spatial signal maps:
      1. top_k_energy: Mean of the top k brightest cycles at each pixel (robust upper quantile).
         Guarantees that cycles with base 'G' (dark) do not penalize the detection of authentic blobs.
      2. persistence_count: Integer map showing how many cycles have e[t](x) > persistence_threshold.
      3. combined_metric: M(x) = top_k_energy * sqrt(1 + persistence_count).
      
    Args:
      cycle_energies: shape (T, Y, X), float32
      top_k: number of top cycles to average (defaults to 3 or max(2, T//3))
      persistence_threshold: minimum energy to count as persistent cycle
      
    Returns:
      top_k_map: shape (Y, X), float32
      persistence_map: shape (Y, X), int32
      combined_metric: shape (Y, X), float32
    """
    T, H, W = cycle_energies.shape
    k = max(1, min(top_k, T))

    # Sort energies descending along cycle axis (axis=0)
    # Using partition for fast top-k extraction
    sorted_energies = np.sort(cycle_energies, axis=0)[::-1]
    top_k_map = np.mean(sorted_energies[:k], axis=0).astype(np.float32)

    # Count how many cycles exceed persistence threshold
    persistence_map = np.sum(cycle_energies >= float(persistence_threshold), axis=0).astype(np.int32)

    # Combined detection metric: rewards both high peak SNR and temporal persistence
    combined_metric = (top_k_map * np.sqrt(1.0 + persistence_map.astype(np.float32))).astype(np.float32)

    return top_k_map, persistence_map, combined_metric
