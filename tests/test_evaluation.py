"""
Tests for evaluation metrics.
Run:
    python -m pytest tests/test_evaluation.py -v
"""
import pandas as pd
import pytest
import numpy as np
from src.evaluation import (
    jaccard_similarity,
    precision_at_k,
    fold_enrichment,
    rank_candidates,
    compare_go_baselines,
    add_matching_strata,
    matched_random_test,
    benjamini_hochberg,
)


def test_jaccard_similarity():
    a = {"A", "B", "C"}
    b = {"B", "C", "D"}

    assert jaccard_similarity(a, b) == pytest.approx(0.5)
    assert jaccard_similarity(set(), set()) == 1.0


def test_precision_at_k():
    ranking = ["A", "B", "C", "D"]
    positives = {"A", "C"}

    assert precision_at_k(ranking, positives, 2) == pytest.approx(0.5)
    assert precision_at_k(ranking, positives, 4) == pytest.approx(0.5)


def test_fold_enrichment():
    assert fold_enrichment(0.8, 0.4) == pytest.approx(2.0)


def test_rank_candidates():
    df = pd.DataFrame(
        {"score": [0.5, 0.9, 0.9]},
        index=["C", "B", "A"],
    )

    assert rank_candidates(df, "score") == ["A", "B", "C"]


def test_go_baseline_comparison():

    df = pd.DataFrame(
        {
            "track": ["Local"] * 4,
            "degree": [4, 3, 2, 1],
            "N": [0.2, 0.9, 0.7, 0.1],
            "R": [0.8, 0.7, 0.3, 0.2],
            "cadtp_score": [0.5, 0.9, 0.8, 0.1],
        },
        index=["A", "B", "C", "D"],
    )

    positives = {"B", "C"}

    result = compare_go_baselines(
        df,
        positive_ids=positives,
        k_values=[2],
        tracks=["Local"],
    )

    assert len(result) == 4

    cadtp = result.loc[result["method"].eq("CADTP")].iloc[0]
    degree = result.loc[result["method"].eq("Degree")].iloc[0]

    assert cadtp["precision"] == pytest.approx(1.0)
    assert degree["precision"] == pytest.approx(0.5)
    assert cadtp["fold_enrichment"] == pytest.approx(2.0)


def test_invalid_precision():

    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, 2)

    with pytest.raises(ValueError):
        precision_at_k(["A", "A"], {"A"}, 2)


def test_matching_strata():

    df = pd.DataFrame({
        "degree": [5, 6, 20, 21, 60, 61],
        "h_hops": [1, 2, 3, 4, 5, 6],
    })

    result = add_matching_strata(df)

    assert result["stratum"].tolist() == [
        "1_0-5",
        "2_6-20",
        "3_6-20",
        "4_21-60",
        "4_21-60",
        "4_61+",
    ]


def test_matched_randomisation_reproducibility():

    pool = pd.DataFrame({
        "stratum": ["1_0-5"] * 4,
        "proteostasis_any": [True, True, False, False],
    }, index=["A", "B", "C", "D"])

    selected = pool.loc[["A", "B"]]
    first = matched_random_test(
        pool,
        selected,
        rng=np.random.default_rng(3888),
        n_permutations=100,
    )

    second = matched_random_test(
        pool,
        selected,
        rng=np.random.default_rng(3888),
        n_permutations=100,
    )

    assert first == second
    assert first["observed_hits"] == 2
    assert 0 < first["empirical_p_value"] <= 1


def test_benjamini_hochberg():
    results = pd.DataFrame({
        "empirical_p_value": [0.001, 0.02, 0.4],
    })

    adjusted = benjamini_hochberg(results)
    np.testing.assert_allclose(
        adjusted["q_value_BH"],
        [0.003, 0.03, 0.4],
    )

    assert adjusted["significant_FDR_05"].tolist() == [
        True, True, False
    ]
