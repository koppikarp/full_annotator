"""Likelihood-based barcode ranking with explicit distance and rejection QC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from popcrop.io.codebook import (
    COLOR_NAMES,
    COLOR_TO_BASE,
    Codebook,
    reverse_complement,
)


def _probability_matrix(cycle_probabilities: List[Dict[str, float]]) -> np.ndarray:
    probs = np.asarray(
        [[p.get(color, 0.0) for color in COLOR_NAMES] for p in cycle_probabilities],
        dtype=np.float64,
    ).reshape(-1, len(COLOR_NAMES))
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("Cycle probabilities must be finite and nonnegative.")
    totals = probs.sum(axis=1)
    if np.any(totals <= 0) or not np.all(np.isfinite(totals)):
        raise ValueError("Each cycle must contain finite positive probability mass.")
    return probs / totals[:, None]


def _hard_indices(probs: np.ndarray) -> np.ndarray:
    indices = probs.argmax(axis=1)
    # Equal maxima do not define a base, regardless of dictionary key order.
    tied = (probs == probs.max(axis=1, keepdims=True)).sum(axis=1) > 1
    indices[tied] = COLOR_NAMES.index("ambiguous")
    return indices


def call_bases_from_probabilities(cycle_probabilities: List[Dict[str, float]]) -> str:
    """Return full-precision hard calls, assigning N to ties or N-class winners."""
    return "".join(COLOR_TO_BASE[COLOR_NAMES[index]]
                   for index in _hard_indices(_probability_matrix(cycle_probabilities)))


class BarcodeDecoder:
    """Rank references without replacing the observed sequence with a whitelist read.

    ``posterior_confidence`` is relative support within the entire supplied
    codebook, under uniform priors and independent cycles. It is not an empirical
    probability that the barcode or nucleus is correct.
    """

    def __init__(
        self,
        codebook: Codebook,
        min_score_margin: float = 0.80,
        min_confidence: float = 0.60,
        eps: float = 1e-6,
        max_hamming_distance: Optional[int] = 2,
    ):
        self.codebook = codebook
        self.min_score_margin = float(min_score_margin)
        self.min_confidence = float(min_confidence)
        self.eps = float(eps)
        if not np.isfinite(self.min_score_margin) or self.min_score_margin < 0:
            raise ValueError("min_score_margin must be finite and nonnegative.")
        if not np.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must lie between zero and one.")
        if not 0 < self.eps < 1:
            raise ValueError("eps must lie strictly between zero and one.")
        if max_hamming_distance is not None and (
            not isinstance(max_hamming_distance, (int, np.integer))
            or max_hamming_distance < 0
        ):
            raise ValueError("max_hamming_distance must be a nonnegative integer or None.")
        self.max_hamming_distance = max_hamming_distance
        self._cycle_indices = np.arange(codebook.barcode_length)

    def _unassigned(self, called_sequence: str, status: str) -> Dict[str, Any]:
        called_library = (
            reverse_complement(called_sequence)
            if self.codebook.match_reverse_complement
            else called_sequence
        )
        return {
            "best_barcode": "UNASSIGNED",
            "best_gene": "NONE",
            "gene_candidates": [],
            "best_score": -999.0,
            "second_best_score": -999.0,
            "score_margin": 0.0,
            "posterior_confidence": 0.0,
            "decoding_status": status,
            "called_sequence": called_sequence,
            "called_library_barcode": called_library,
            "matched_sequence": "N" * len(called_sequence),
            "hamming_distance": None,
            "likelihood_selected_hamming_distance": None,
            "nearest_hamming_distance": None,
            "nearest_barcode": "UNASSIGNED",
            "nearest_gene": "NONE",
            "nearest_barcode_count": 0,
            "nearest_barcodes": [],
            "nearest_genes": [],
            "num_ambiguous_cycles": called_sequence.count("N"),
        }

    def decode_component(
        self,
        cycle_probabilities: List[Dict[str, float]],
        p_real_blob: float = 1.0,
        prob_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Score every full-length reference and report both distance criteria.

        ``hamming_distance`` and ``likelihood_selected_hamming_distance`` refer
        to the likelihood winner. ``nearest_hamming_distance`` refers to the
        closest whitelist sequence by hard-call distance, which can differ.
        N-containing observations are retained for QC but excluded from mapping.
        """
        if not np.isfinite(p_real_blob) or not 0 <= p_real_blob <= 1:
            raise ValueError("p_real_blob must lie between zero and one.")
        if not np.isfinite(prob_threshold) or not 0 <= prob_threshold <= 1:
            raise ValueError("prob_threshold must lie between zero and one.")
        n_cycles = len(cycle_probabilities)
        if n_cycles != self.codebook.barcode_length:
            raise ValueError(
                f"Read has {n_cycles} cycles but codebook barcodes have "
                f"{self.codebook.barcode_length} bases; implicit truncation is not allowed."
            )
        probs = _probability_matrix(cycle_probabilities)
        hard_indices = _hard_indices(probs)
        called_seq = "".join(COLOR_TO_BASE[COLOR_NAMES[index]] for index in hard_indices)

        if p_real_blob < prob_threshold:
            return self._unassigned(called_seq, "rejected_artifact")
        if "N" in called_seq:
            return self._unassigned(called_seq, "rejected_ambiguous_bases")

        log_probs = np.log(np.maximum(self.eps, probs))
        expected = self.codebook.expected_color_indices
        # A single N_barcodes x N_cycles gather replaces the Python barcode loop.
        scores = log_probs[self._cycle_indices, expected].sum(axis=1)
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        second_best_score = (
            float(np.partition(scores, -2)[-2]) if len(scores) > 1 else float("-inf")
        )
        score_margin = best_score - second_best_score
        confidence = float(1.0 / np.exp(scores - best_score).sum())

        distances = np.count_nonzero(expected != hard_indices[None, :], axis=1)
        selected_distance = int(distances[best_idx])
        nearest_distance = int(distances.min())
        nearest_indices = np.flatnonzero(distances == nearest_distance)
        nearest_barcodes = [self.codebook.barcodes[index] for index in nearest_indices]
        nearest_genes = list(dict.fromkeys(
            gene for index in nearest_indices for gene in self.codebook.gene_candidates[index]
        ))
        gene_candidates = self.codebook.gene_candidates[best_idx]

        if self.max_hamming_distance is not None and selected_distance > self.max_hamming_distance:
            status = "unmapped"
        elif (
            score_margin > 0
            and score_margin >= self.min_score_margin
            and confidence >= self.min_confidence
        ):
            status = "ambiguous_gene" if len(gene_candidates) != 1 else "matched"
        else:
            status = "ambiguous"

        return {
            "best_barcode": self.codebook.barcodes[best_idx],
            "best_gene": self.codebook.genes[best_idx],
            "gene_candidates": gene_candidates,
            "best_score": best_score,
            "second_best_score": second_best_score,
            "score_margin": float(score_margin),
            "posterior_confidence": confidence,
            "decoding_status": status,
            "called_sequence": called_seq,
            "called_library_barcode": (
                reverse_complement(called_seq)
                if self.codebook.match_reverse_complement
                else called_seq
            ),
            "matched_sequence": self.codebook.target_barcodes[best_idx],
            "hamming_distance": selected_distance,
            "likelihood_selected_hamming_distance": selected_distance,
            "nearest_hamming_distance": nearest_distance,
            "nearest_barcode": nearest_barcodes[0] if len(nearest_barcodes) == 1 else "AMBIGUOUS",
            "nearest_gene": nearest_genes[0] if len(nearest_barcodes) == len(nearest_genes) == 1 else "AMBIGUOUS",
            "nearest_barcode_count": len(nearest_barcodes),
            "nearest_barcodes": nearest_barcodes,
            "nearest_genes": nearest_genes,
            "num_ambiguous_cycles": 0,
        }

    def decode_nucleus_multicomponent(
        self,
        component_results: List[Dict[str, Any]],
        p_nucleus: float = 1.0,
    ) -> Dict[str, Any]:
        """Select a passing component and flag disagreement among passing matches.

        Ambiguous or unmapped components cannot replace a passing match merely
        because they have higher absolute log likelihood. Their records remain
        in the component table for review.
        """
        valid_comps = [c for c in component_results if c["decoding_status"] != "rejected_artifact"]
        if not valid_comps:
            result = self._unassigned("N" * self.codebook.barcode_length, "rejected_artifact")
            result["num_valid_components"] = 0
            return result

        matched = [c for c in valid_comps if c["decoding_status"] in ("matched", "ambiguous_gene")]
        eligible = matched or valid_comps
        primary = max(
            eligible,
            key=lambda c: (c["posterior_confidence"], c["score_margin"], c["best_score"]),
        )
        # Copy decoder fields only, avoiding another copy of per-cycle image
        # features or other large component metadata into the nucleus record.
        keys = self._unassigned("", "").keys()
        result = {key: primary[key] for key in keys if key in primary}
        if len({c["best_barcode"] for c in matched}) > 1:
            result["decoding_status"] = "mixed"
        result["num_valid_components"] = len(valid_comps)
        result["primary_component_id"] = primary.get("component_id")
        return result
