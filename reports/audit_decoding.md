# Barcode decoding audit, 28 September 2026

## Assay interpretation and reference orientation

The manual annotation semantics are A = real green only, T = real red only,
C = real green and red, and G = neither channel contains real signal. The raw
cycle-order sequence remains an observation. With reverse-complement matching
enabled, its reverse complement is the sequence in the library orientation.
An N-containing read is retained for quality control and excluded from mapping.
The decoder also uses N for a tied maximum cycle probability, so dictionary
ordering cannot manufacture a base call.

The library contains two relevant barcode columns. `barcode_new` differs from
`barcode` in 500 of 15,766 rows. The user identified the replacement column as
the likely synthesized library, so the production configuration explicitly uses
`barcode_new`; the column remains configurable. This distinction cannot be
resolved from the nine complete manual reads, whose nearest hits are the same
under both columns. The replacement column contains 15,757 unique sequences:
nine sequences occur twice and each collision has different gene assignments.

The codebook now consolidates identical sequences into one likelihood hypothesis
and retains all associated genes. Otherwise duplicated rows would count twice
in the prior and a tie could be mistaken for a uniquely identified gene. A
uniquely mapped sequence with several possible genes receives `ambiguous_gene`;
`gene_candidates` preserves the alternatives and gene-abundance summaries must
exclude that observation from uniquely assigned genes. Repeated rows for the
same sequence and same gene remain identifiable.

## Problems independently verified in the earlier decoder

The earlier implementation described its scoring loop as vectorized, but
iterated over every library barcode in Python and reconstructed expected color
indices for each read. It computed confidence over only the top 20 candidates,
which overstates relative support when additional references have appreciable
likelihood. It silently truncated comparisons to the shorter of the read and
reference lengths. N-containing observations were allowed to select a library
reference. Configured maximum Hamming distance was not an acceptance gate.
The output did not distinguish the probability-selected reference from the
closest reference by hard-call Hamming distance. These issues can distort
assignment and confidence independently of spot detection.

## Implemented behavior

The codebook validates nonempty A/C/G/T reference sequences of one exact length
before processing any nuclei. The expected color indices are compiled once as
a compact integer array. The decoder gathers cycle log-probabilities for all
references with NumPy, preserving full probability precision. The score is the
sum of per-cycle log-probabilities, with an explicit numerical floor of 1e-6.
Read length must exactly match the library length.

`posterior_confidence` now normalizes across all unique library sequences. It
is a model-relative quantity under uniform sequence priors and independent
cycles. It is not an empirically calibrated probability that the nucleus has
the reported construct. Margin, confidence, and maximum Hamming distance are
all required for acceptance, and exact likelihood ties remain ambiguous even
if the configurable thresholds are zero.

`called_sequence` is the unchanged optical hard call;
`called_library_barcode` is its library-oriented sequence;
`best_barcode` and `matched_sequence` describe the likelihood-ranked reference;
`hamming_distance` and `likelihood_selected_hamming_distance` measure distance
to that reference. `nearest_hamming_distance`, `nearest_barcodes`,
`nearest_genes`, and `nearest_barcode_count` separately describe every nearest
hard-call match. The Hamming acceptance gate applies to the likelihood winner.
The decoder does not replace an implausible likelihood winner with a convenient
nearby sequence to improve reported mapping rates. Accepted assignment columns
are produced by the pipeline and remain distinct from ranking candidates.

The main analysis emits one read per nucleus, as requested. The old
multi-component summarizer remains compatible with historical runs; it now
prioritizes accepted components and flags disagreement among accepted reads.

## What the saved manual annotations establish

The saved base table contains 1,776 annotated cycles, including 16 N labels.
Its cycle coverage is strongly unbalanced: 1,139 rows are cycle 1, while cycles
6, 7, 8, 10, 11, and 12 each have only nine annotations. Grouping by TIFF,
well, tile, and cell, and requiring all twelve cycles, yields nine complete
reads. None has conflicting duplicate cycles. Their nearest reverse-complement
Hamming-distance histogram is 0: 2 reads, 1: 4, 2: 2, and 3: 1.

Cell 112 is an exact TFCP2 read and cell 4308 is an exact RPS12 read. Cells 820,
1422, 1697, and 1897 have unique distance-one nearest matches. Cell 962 has a
unique distance-two nearest match to PSD3. Cell 954 has three equally close
distance-two matches, STXBP4, HM13, and RPS15. Cell 1736 has six distance-three
nearest candidates. Thus six of nine complete manual reads are within one
substitution. These selected cells support the reported manual performance,
but cannot establish image-wide signal sensitivity or justify promising that
every automated read will be within distance one.

`scripts/validate_manual_reads.py` reproduces the complete-read comparison on
the regenerated shared features and exports per-cycle and per-nucleus evidence
under `reports/manual_read_validation/`. The deployed all-data model is clearly
identified as a reconstruction diagnostic. Separate saved grouped holdout
predictions supply out-of-sample per-cycle results and, where complete reads
exist in that split, out-of-sample barcode comparisons. Near-whitelist reads
alone are insufficient validation: library density, model uncertainty,
correlated cycle errors, and gene collisions must remain visible.

The current shared-feature model reconstructs all 108 annotated cycles in
those nine complete reads. This is an in-sample result. Its grouped holdout
contains 335 cycles from 307 nuclei and has 94.93% base accuracy. Only one
complete twelve-cycle read falls in that holdout; all twelve bases agree with
the manual annotation and its library distance is zero. That sample is too
small to estimate a stable complete-barcode success rate.

## Checks and performance

Focused decoder tests cover forward and reverse-complement orientation,
all-library normalization, explicit distance gating, nearest-reference ties,
ambiguous-cycle rejection, exact length, invalid inputs, full precision,
duplicate-sequence gene collisions, and the shared cycle hard-call rule.
All 38 tests passed after the decoder and duplicate-reference changes.

A local 300-call benchmark against the original 15,766 reference sequences
measured approximately 1.01 ms per read including ranking and all Hamming
diagnostics. The compiled 12-cycle reference array occupies 189,192 bytes.
This supports a fast decoder, without implying that image segmentation and
feature extraction share that runtime. The proposed 0.3 ms performance and
an eventual 1,200 to 2,000 real-amplicon count were not treated as requirements
or validation targets.
