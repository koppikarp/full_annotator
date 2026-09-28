"""Reference barcode codebook loader and base-to-color mapping."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Two-color mapping used by this assay and the manual annotator:
# A -> Green
# T -> Red
# C -> Yellow (Cy3 + GFP dual-positive)
# G -> Dark (unlabeled)
BASE_TO_COLOR = {
    "A": "green",
    "T": "red",
    "C": "yellow",
    "G": "dark",
    "N": "ambiguous",
}

COLOR_TO_BASE = {v: k for k, v in BASE_TO_COLOR.items()}
COLOR_NAMES = ("red", "green", "yellow", "dark", "ambiguous")
COLOR_TO_INDEX = {color: index for index, color in enumerate(COLOR_NAMES)}


def reverse_complement(seq: str) -> str:
    """Computes the reverse complement of a DNA sequence."""
    tr = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tr)[::-1]


class Codebook:
    """Represents a reference barcode library with base-to-color mapping."""

    def __init__(
        self,
        df_codebook: pd.DataFrame,
        barcode_col: str = "barcode",
        gene_col: Optional[str] = "gene",
        match_reverse_complement: bool = True
    ):
        self.df = df_codebook.copy()
        if barcode_col not in self.df.columns:
            raise ValueError(f"Barcode column '{barcode_col}' not found in codebook.")
        
        self.barcode_col = barcode_col
        self.gene_col = gene_col if gene_col in self.df.columns else None
        self.match_reverse_complement = bool(match_reverse_complement)

        # Invalid references must fail here, before any cell is processed. In
        # particular, silently truncating unequal lengths creates false matches.
        if self.df.empty:
            raise ValueError("Codebook must contain at least one barcode.")
        if self.df[barcode_col].isna().any():
            raise ValueError("Codebook contains missing barcodes.")
        self.df[barcode_col] = self.df[barcode_col].astype(str).str.strip().str.upper()
        invalid = ~self.df[barcode_col].str.fullmatch("[ACGT]+")
        if invalid.any():
            raise ValueError("Codebook barcodes must contain only A, C, G and T.")
        lengths = self.df[barcode_col].str.len().unique()
        if len(lengths) != 1:
            raise ValueError("All codebook barcodes must have the same length.")
        self.barcode_length = int(lengths[0])
        # Identical observed sequences are one likelihood hypothesis, even if
        # multiple library constructs contain them. Retain every possible gene
        # rather than assigning an arbitrary duplicate row or duplicate priors.
        grouped = self.df.groupby(barcode_col, sort=False, dropna=False)
        self.barcodes = list(grouped.groups)
        self.gene_candidates = [
            list(dict.fromkeys(str(gene) for gene in group[self.gene_col].dropna()))
            if self.gene_col else [barcode]
            for barcode, group in grouped
        ]
        self.reference_row_counts = [len(group) for _, group in grouped]
        self.genes = [genes[0] if len(genes) == 1 else "AMBIGUOUS" if genes else "NONE"
                      for genes in self.gene_candidates]
        
        # In SBS chemistry, in situ sequencing synthesizes the reverse complement
        # of the reference library barcode template.
        if self.match_reverse_complement:
            self.target_barcodes = [reverse_complement(bc) for bc in self.barcodes]
        else:
            self.target_barcodes = list(self.barcodes)

        self.rc_barcodes = [reverse_complement(bc) for bc in self.barcodes]

        # Precompute expected color matrix: shape (N_barcodes, N_cycles)
        self.expected_colors = [
            [BASE_TO_COLOR.get(base, "ambiguous") for base in bc]
            for bc in self.target_barcodes
        ]
        self.expected_color_indices = np.asarray(
            [[COLOR_TO_INDEX[color] for color in colors] for colors in self.expected_colors],
            dtype=np.uint8,
        )

    def __len__(self) -> int:
        return len(self.barcodes)

    def get_expected_colors_for_barcode(self, barcode: str, is_library_barcode: bool = True) -> List[str]:
        seq = barcode.strip().upper()
        if self.match_reverse_complement and is_library_barcode:
            seq = reverse_complement(seq)
        return [BASE_TO_COLOR.get(base, "ambiguous") for base in seq]

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        barcode_col: str = "barcode",
        gene_col: Optional[str] = "gene",
        match_reverse_complement: bool = True
    ) -> Codebook:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Codebook file not found: {path}")

        logger.info(f"Loading codebook from {path}")
        sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep)
        
        # Automatically detect barcode column if not found
        if barcode_col not in df.columns:
            if barcode_col != "barcode":
                raise ValueError(f"Explicit barcode column '{barcode_col}' not found in codebook.")
            possible_cols = [c for c in df.columns if "barcode" in c.lower() or "seq" in c.lower()]
            if possible_cols:
                barcode_col = possible_cols[0]
            else:
                # If single column, use it
                barcode_col = df.columns[0]

        # Automatically detect gene column
        if gene_col not in df.columns:
            possible_gene_cols = [c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()]
            gene_col = possible_gene_cols[0] if possible_gene_cols else None

        logger.info(f"Loaded codebook with {len(df)} entries. Barcode col='{barcode_col}', Gene col='{gene_col}', RC_match={match_reverse_complement}")
        return cls(df, barcode_col=barcode_col, gene_col=gene_col, match_reverse_complement=match_reverse_complement)
