"""Unit tests for codebook decoding."""

import pandas as pd
import pytest

from popcrop.io.codebook import Codebook
from popcrop.decoding import BarcodeDecoder, call_bases_from_probabilities


def test_barcode_decoder():
    # 3-barcode reference library with forward matching:
    # B1: "ATCG" -> [Green, Red, Yellow, Dark]
    # B2: "TTTT" -> [Red, Red, Red, Red]
    # B3: "AAAA" -> [Green, Green, Green, Green]
    df_cb = pd.DataFrame({
        "barcode": ["ATCG", "TTTT", "AAAA"],
        "gene": ["GENE_A", "GENE_T", "GENE_G"]
    })
    cb = Codebook(df_cb, match_reverse_complement=False)
    decoder = BarcodeDecoder(cb, min_score_margin=1.0)

    # Probabilities matching B1: ATCG
    probs = [
        {"green": 0.95, "red": 0.01, "yellow": 0.01, "dark": 0.01, "ambiguous": 0.02},
        {"green": 0.01, "red": 0.95, "yellow": 0.01, "dark": 0.01, "ambiguous": 0.02},
        {"green": 0.01, "red": 0.01, "yellow": 0.95, "dark": 0.01, "ambiguous": 0.02},
        {"green": 0.01, "red": 0.01, "yellow": 0.01, "dark": 0.95, "ambiguous": 0.02},
    ]

    res = decoder.decode_component(probs, p_real_blob=1.0)
    assert res["best_barcode"] == "ATCG"
    assert res["best_gene"] == "GENE_A"
    assert res["decoding_status"] == "matched"
    assert res["score_margin"] > 5.0
    assert res["posterior_confidence"] > 0.90
    assert res["called_sequence"] == "ATCG"
    assert res["called_library_barcode"] == "ATCG"
    assert res["matched_sequence"] == "ATCG"


def test_barcode_decoder_reverse_complement():
    """
    Verifies that in situ sequencing chemistry matches against the Reverse Complement
    of reference library barcodes while returning the canonical library barcode and gene.
    TFCP2 library barcode: AAGCTCATCGTA -> in situ synthesized/optical read: TACGATGAGCTT
    RPS12 library barcode: AGTCTACTGTAT -> in situ synthesized/optical read: ATACAGTAGACT
    """
    df_cb = pd.DataFrame({
        "barcode": ["AAGCTCATCGTA", "AGTCTACTGTAT"],
        "gene": ["TFCP2", "RPS12"]
    })
    # match_reverse_complement=True is default
    cb = Codebook(df_cb, match_reverse_complement=True)
    decoder = BarcodeDecoder(cb, min_score_margin=1.0)

    # 1. Test TFCP2 optical read: TACGATGAGCTT
    color_map = {"A": "green", "T": "red", "C": "yellow", "G": "dark"}
    optical_tfcp2 = "TACGATGAGCTT"
    probs_tfcp2 = []
    for base in optical_tfcp2:
        c = color_map[base]
        p = {k: 0.01 for k in ["green", "red", "yellow", "dark", "ambiguous"]}
        p[c] = 0.96
        probs_tfcp2.append(p)

    res_tfcp2 = decoder.decode_component(probs_tfcp2, p_real_blob=1.0)
    assert res_tfcp2["best_barcode"] == "AAGCTCATCGTA"  # Canonical library barcode reported
    assert res_tfcp2["best_gene"] == "TFCP2"
    assert res_tfcp2["called_sequence"] == "TACGATGAGCTT"  # Raw optical sequence read
    assert res_tfcp2["called_library_barcode"] == "AAGCTCATCGTA"  # Reverse complement matches library
    assert res_tfcp2["matched_sequence"] == "TACGATGAGCTT"  # Expected optical sequence
    assert res_tfcp2["decoding_status"] == "matched"
    assert res_tfcp2["posterior_confidence"] > 0.95

    # 2. Test RPS12 optical read: ATACAGTAGACT
    optical_rps12 = "ATACAGTAGACT"
    probs_rps12 = []
    for base in optical_rps12:
        c = color_map[base]
        p = {k: 0.01 for k in ["green", "red", "yellow", "dark", "ambiguous"]}
        p[c] = 0.96
        probs_rps12.append(p)

    res_rps12 = decoder.decode_component(probs_rps12, p_real_blob=1.0)
    assert res_rps12["best_barcode"] == "AGTCTACTGTAT"  # Canonical library barcode reported
    assert res_rps12["best_gene"] == "RPS12"
    assert res_rps12["called_sequence"] == "ATACAGTAGACT"
    assert res_rps12["called_library_barcode"] == "AGTCTACTGTAT"
    assert res_rps12["matched_sequence"] == "ATACAGTAGACT"
    assert res_rps12["decoding_status"] == "matched"


def test_ambiguous_decoding():
    df_cb = pd.DataFrame({
        "barcode": ["AAAA", "TAAA"],
        "gene": ["GENE_1", "GENE_2"]
    })
    cb = Codebook(df_cb, match_reverse_complement=False)
    decoder = BarcodeDecoder(cb, min_score_margin=2.0)

    # Cycle 0 is ambiguous between Green (A) and Red (T)
    probs = [
        {"green": 0.50, "red": 0.49, "yellow": 0.00, "dark": 0.00, "ambiguous": 0.01},
        {"green": 0.95, "red": 0.01, "yellow": 0.01, "dark": 0.01, "ambiguous": 0.02},
        {"green": 0.95, "red": 0.01, "yellow": 0.01, "dark": 0.01, "ambiguous": 0.02},
        {"green": 0.95, "red": 0.01, "yellow": 0.01, "dark": 0.01, "ambiguous": 0.02},
    ]

    res = decoder.decode_component(probs, p_real_blob=1.0)
    # Score margin is near zero -> should be ambiguous
    assert res["decoding_status"] == "ambiguous"


def _probabilities(sequence):
    from popcrop.io.codebook import BASE_TO_COLOR
    return [
        {color: (0.96 if color == BASE_TO_COLOR[base] else 0.01)
         for color in ("red", "green", "yellow", "dark", "ambiguous")}
        for base in sequence
    ]


def test_hamming_gate_retains_observation_and_separate_nearest_reference():
    cb = Codebook(pd.DataFrame({"barcode": ["AAAA", "TTTT"]}), match_reverse_complement=False)
    decoder = BarcodeDecoder(cb, max_hamming_distance=2)
    # Three weak A calls and one decisive T make TTTT the likelihood winner,
    # while AAAA is closer to the unchanged hard call AAAT.
    probabilities = [{"green": 0.51, "red": 0.49}] * 3 + [{"green": 0.001, "red": 0.999}]
    result = decoder.decode_component(probabilities)
    assert result["called_sequence"] == "AAAT"
    assert result["best_barcode"] == "TTTT"
    assert result["likelihood_selected_hamming_distance"] == 3
    assert result["hamming_distance"] == 3
    assert result["nearest_hamming_distance"] == 1
    assert result["nearest_barcode"] == "AAAA"
    assert result["decoding_status"] == "unmapped"


def test_nearest_distance_ties_are_reported_without_arbitrary_gene_assignment():
    cb = Codebook(pd.DataFrame({"barcode": ["AAAA", "AATT"], "gene": ["G1", "G2"]}),
                  match_reverse_complement=False)
    result = BarcodeDecoder(cb).decode_component(_probabilities("AAAT"))
    assert result["nearest_hamming_distance"] == 1
    assert result["nearest_barcode_count"] == 2
    assert result["nearest_barcode"] == "AMBIGUOUS"
    assert result["nearest_barcodes"] == ["AAAA", "AATT"]
    assert result["nearest_genes"] == ["G1", "G2"]
    assert result["decoding_status"] == "ambiguous"


def test_confidence_uses_entire_library_and_vectorized_scores_match_reference():
    import itertools
    import numpy as np
    from popcrop.io.codebook import COLOR_NAMES
    references = ["".join(bases) for bases in itertools.product("ACGT", repeat=3)]
    cb = Codebook(pd.DataFrame({"barcode": references}), match_reverse_complement=False)
    probabilities = [{"green": 0.26, "yellow": 0.25, "dark": 0.245, "red": 0.245}] * 3
    result = BarcodeDecoder(cb).decode_component(probabilities)
    reference_scores = np.array([
        sum(np.log(probabilities[cycle][color]) for cycle, color in enumerate(colors))
        for colors in cb.expected_colors
    ])
    best = reference_scores.max()
    full_confidence = 1 / np.exp(reference_scores - best).sum()
    truncated_confidence = 1 / np.exp(np.sort(reference_scores)[-20:] - best).sum()
    assert result["best_score"] == pytest.approx(best)
    assert result["score_margin"] == pytest.approx(best - np.sort(reference_scores)[-2])
    assert result["posterior_confidence"] == pytest.approx(full_confidence)
    assert result["posterior_confidence"] < truncated_confidence


@pytest.mark.parametrize("probabilities", [
    _probabilities("ANAA"),
    [{"green": 0.5, "red": 0.5}] + _probabilities("AAA"),
])
def test_ambiguous_bases_are_discarded_from_mapping(probabilities):
    cb = Codebook(pd.DataFrame({"barcode": ["AAAA"]}), match_reverse_complement=False)
    result = BarcodeDecoder(cb).decode_component(probabilities)
    assert "N" in result["called_sequence"]
    assert result["best_barcode"] == "UNASSIGNED"
    assert result["decoding_status"] == "rejected_ambiguous_bases"
    assert result["nearest_hamming_distance"] is None
    assert result["num_ambiguous_cycles"] == 1


def test_blob_rejection_preserves_raw_base_observation():
    cb = Codebook(pd.DataFrame({"barcode": ["AAAA"]}), match_reverse_complement=False)
    result = BarcodeDecoder(cb).decode_component(_probabilities("AAAT"), p_real_blob=0.1)
    assert result["decoding_status"] == "rejected_artifact"
    assert result["called_sequence"] == "AAAT"
    assert result["best_barcode"] == "UNASSIGNED"


def test_reverse_complement_distance_uses_canonical_library_orientation():
    cb = Codebook(pd.DataFrame({"barcode": ["AAGCTCATCGTA"]}))
    result = BarcodeDecoder(cb).decode_component(_probabilities("TACGATGATCTT"))
    assert result["called_library_barcode"] == "AAGATCATCGTA"
    assert result["nearest_hamming_distance"] == 1
    assert result["likelihood_selected_hamming_distance"] == 1


@pytest.mark.parametrize("barcodes", [[], [None], [""], ["ACNX"], ["AC", "ACG"]])
def test_invalid_libraries_fail_before_decoding(barcodes):
    with pytest.raises(ValueError):
        Codebook(pd.DataFrame({"barcode": barcodes}))


def test_duplicate_sequence_preserves_gene_collision_without_duplicate_priors():
    cb = Codebook(pd.DataFrame({"barcode": ["AC", "ac", "TT"],
                                "gene": ["GENE_1", "GENE_2", "GENE_3"]}),
                  match_reverse_complement=False)
    assert len(cb) == 2
    assert cb.gene_candidates == [["GENE_1", "GENE_2"], ["GENE_3"]]
    result = BarcodeDecoder(cb).decode_component(_probabilities("AC"))
    assert result["decoding_status"] == "ambiguous_gene"
    assert result["best_barcode"] == "AC"
    assert result["best_gene"] == "AMBIGUOUS"
    assert result["gene_candidates"] == ["GENE_1", "GENE_2"]
    assert result["nearest_barcode_count"] == 1
    assert result["nearest_gene"] == "AMBIGUOUS"
    assert result["nearest_genes"] == ["GENE_1", "GENE_2"]
    unique = Codebook(pd.DataFrame({"barcode": ["AC", "TT"]}), match_reverse_complement=False)
    assert result["posterior_confidence"] == BarcodeDecoder(unique).decode_component(_probabilities("AC"))["posterior_confidence"]


def test_duplicate_sequence_for_same_gene_is_identifiable():
    cb = Codebook(pd.DataFrame({"barcode": ["AC", "ac"], "gene": ["GENE_1", "GENE_1"]}),
                  match_reverse_complement=False)
    result = BarcodeDecoder(cb).decode_component(_probabilities("AC"))
    assert result["decoding_status"] == "matched"
    assert result["best_gene"] == "GENE_1"
    assert result["gene_candidates"] == ["GENE_1"]


def test_explicit_reference_column_cannot_silently_fall_back(tmp_path):
    path = tmp_path / "library.csv"
    pd.DataFrame({"barcode": ["AC"], "gene": ["GENE_1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Explicit barcode column"):
        Codebook.from_file(path, barcode_col="barcode_new")


def test_missing_gene_does_not_create_unique_gene_assignment():
    cb = Codebook(pd.DataFrame({"barcode": ["AC"], "gene": [None]}), match_reverse_complement=False)
    result = BarcodeDecoder(cb).decode_component(_probabilities("AC"))
    assert result["decoding_status"] == "ambiguous_gene"
    assert result["best_gene"] == "NONE"
    assert result["gene_candidates"] == []


def test_mismatched_read_length_fails_instead_of_truncating():
    cb = Codebook(pd.DataFrame({"barcode": ["AAAA"]}), match_reverse_complement=False)
    with pytest.raises(ValueError, match="implicit truncation"):
        BarcodeDecoder(cb).decode_component(_probabilities("AAA"))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_invalid_cycle_probabilities_fail(invalid):
    cb = Codebook(pd.DataFrame({"barcode": ["A"]}), match_reverse_complement=False)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        BarcodeDecoder(cb).decode_component([{"green": invalid}])


def test_multicomponent_summary_prioritizes_passing_matches_and_retains_distance():
    cb = Codebook(pd.DataFrame({"barcode": ["AAAA", "TTTT"]}), match_reverse_complement=False)
    decoder = BarcodeDecoder(cb)
    passing = decoder.decode_component(_probabilities("AAAA"))
    passing["component_id"] = 7
    ambiguous = dict(passing, best_score=0, posterior_confidence=0.99,
                     decoding_status="ambiguous", best_barcode="TTTT", component_id=8)
    result = decoder.decode_nucleus_multicomponent([ambiguous, passing])
    assert result["best_barcode"] == "AAAA"
    assert result["primary_component_id"] == 7
    assert result["decoding_status"] == "matched"
    assert result["nearest_hamming_distance"] == 0
    conflict = decoder.decode_component(_probabilities("TTTT"))
    result = decoder.decode_nucleus_multicomponent([passing, conflict])
    assert result["decoding_status"] == "mixed"


def test_empty_nucleus_sequence_uses_library_length():
    cb = Codebook(pd.DataFrame({"barcode": ["AAA"]}), match_reverse_complement=False)
    result = BarcodeDecoder(cb).decode_nucleus_multicomponent([])
    assert result["called_sequence"] == "NNN"


def test_full_precision_probabilities_retain_weakly_resolved_base():
    cb = Codebook(pd.DataFrame({"barcode": ["A", "T"]}), match_reverse_complement=False)
    probabilities = [{"green": 0.500001, "red": 0.499999}]
    result = BarcodeDecoder(cb, min_score_margin=0, min_confidence=0).decode_component(probabilities)
    assert result["called_sequence"] == "A"
    assert result["posterior_confidence"] == pytest.approx(0.500001, abs=1e-12)
    assert result["score_margin"] > 0
    assert probabilities == [{"green": 0.500001, "red": 0.499999}]


def test_shared_hard_calls_match_decoder_and_ignore_dictionary_order():
    probabilities = [{"red": 0.5, "green": 0.5}, {"green": 0.5, "red": 0.5},
                     {"ambiguous": 0.7, "dark": 0.3}] + _probabilities("ACTG")
    cb = Codebook(pd.DataFrame({"barcode": ["AAAAAAA"]}), match_reverse_complement=False)
    assert call_bases_from_probabilities(probabilities) == "NNNACTG"
    assert BarcodeDecoder(cb).decode_component(probabilities)["called_sequence"] == "NNNACTG"
    assert call_bases_from_probabilities([]) == ""


def test_equal_likelihoods_do_not_map_with_zero_thresholds():
    # Neither cycle has tied hard calls, but the two reference likelihoods tie.
    cb = Codebook(pd.DataFrame({"barcode": ["AT", "TA"]}), match_reverse_complement=False)
    result = BarcodeDecoder(cb, min_score_margin=0, min_confidence=0).decode_component(
        [{"green": 0.8, "red": 0.2}] * 2
    )
    assert result["called_sequence"] == "AA"
    assert result["decoding_status"] == "ambiguous"


@pytest.mark.parametrize("kwargs", [
    {"min_score_margin": -1}, {"min_score_margin": float("nan")},
    {"min_confidence": 1.01}, {"min_confidence": float("nan")},
    {"max_hamming_distance": -1}, {"max_hamming_distance": 1.5},
])
def test_invalid_decoding_thresholds_fail(kwargs):
    cb = Codebook(pd.DataFrame({"barcode": ["A"]}))
    with pytest.raises(ValueError):
        BarcodeDecoder(cb, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"p_real_blob": float("nan")}, {"p_real_blob": 1.01},
    {"prob_threshold": float("nan")}, {"prob_threshold": -0.1},
])
def test_invalid_blob_probability_gate_fails(kwargs):
    cb = Codebook(pd.DataFrame({"barcode": ["A"]}))
    with pytest.raises(ValueError):
        BarcodeDecoder(cb).decode_component(_probabilities("T"), **kwargs)
