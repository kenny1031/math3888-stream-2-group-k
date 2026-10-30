"""
Regression tests for CADTP training and held-out validation.

These tests use saved pipeline outputs and do not retrain
Node2Vec or rerun Louvain.

Run:
    python -m pytest tests/test_cadtp.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config


TRACKS = ["Local", "Cross", "Boundary"]


@pytest.fixture(scope="module")
def results():
    cfg = load_config()
    files = {
        "train": cfg.path("rankings") / "cadtp_training_refactor.csv",
        "validation": cfg.path("evaluation") / "cadtp_validation_rankings.csv",
    }

    for name, path in files.items():
        if not path.is_file():
            pytest.fail(
                f"Missing {name} results: {path}\n"
                "Run CADTP training and validation first."
            )

    return {
        name: pd.read_csv(
            path,
            index_col="string_id",
        )
        for name, path in files.items()
    }


# ============================================================
# 1. Candidate universe and track counts
# ============================================================

def test_candidate_counts(results):

    train = results["train"]
    valid = results["validation"]

    assert len(train) == 4439
    assert len(valid) == 4439

    assert train.index.is_unique
    assert valid.index.is_unique
    assert set(train.index) == set(valid.index)

    assert train["track"].value_counts().to_dict() == {
        "Cross": 3985,
        "Boundary": 334,
        "Local": 120,
    }

    assert valid["track"].value_counts().to_dict() == {
        "Cross": 3807,
        "Boundary": 535,
        "Local": 97,
    }


# ============================================================
# 2. Track classification via integer votes
# ============================================================

def test_track_classification(results):
    for df in results.values():
        votes = df["membership_count"].to_numpy()
        n_runs = df["num_runs"].to_numpy()

        expected = np.select(
            [
                votes >= np.ceil(0.8 * n_runs),
                votes <= np.floor(0.2 * n_runs),
            ],
            ["Local", "Cross"],
            default="Boundary",
        )

        np.testing.assert_array_equal(
            df["track"].to_numpy(),
            expected,
        )


# ============================================================
# 3. CADTP score formula
# ============================================================

def test_cadtp_scores(results):

    for df in results.values():

        N = df["N"].to_numpy()
        R = df["R"].to_numpy()
        C = df["C"].to_numpy()

        expected = np.where(
            df["track"].eq("Local").to_numpy(),
            (N + R + C) / 3,
            (N + R) / 2,
        )

        np.testing.assert_allclose(
            df["cadtp_score"].to_numpy(),
            expected,
            rtol=0,
            atol=1e-12,
        )

        assert df["cadtp_score"].between(0, 1).all()


# ============================================================
# 4. Independent ranking within each track
# ============================================================

def test_rank_within_track(results):
    for df in results.values():
        for track in TRACKS:
            subset = df.loc[df["track"].eq(track)]
            ranks = sorted(subset["rank_within_track"].tolist())
            assert ranks == list(range(1, len(subset) + 1))


# ============================================================
# 5. Held-out classification agreement
# ============================================================

def test_validation_agreement(results):

    train = results["train"]
    valid = results["validation"].reindex(train.index)

    same_track = train["track"].eq(valid["track"])

    assert same_track.sum() == 4166
    assert (~same_track).sum() == 273

    matrix = pd.crosstab(
        train["track"],
        valid["track"],
    ).reindex(
        index=TRACKS,
        columns=TRACKS,
        fill_value=0,
    )

    expected = np.array([
        [95, 0, 25],
        [0, 3773, 212],
        [2, 34, 298],
    ])

    np.testing.assert_array_equal(
        matrix.to_numpy(),
        expected,
    )


# ============================================================
# 6. Top-20 stability
# ============================================================

def test_top20_jaccard(results):
    train = results["train"]
    valid = results["validation"]

    expected_overlaps = {
        "Local": 19,
        "Cross": 20,
        "Boundary": 16,
    }

    for track, expected in expected_overlaps.items():

        train_top = set(
            train.loc[train["track"].eq(track)]
            .nsmallest(20, "rank_within_track")
            .index
        )

        val_top = set(
            valid.loc[valid["track"].eq(track)]
            .nsmallest(20, "rank_within_track")
            .index
        )

        assert len(train_top & val_top) == expected


# ============================================================
# 7. Key candidate regression
# ============================================================

def test_key_candidates(results):
    train = results["train"]
    local_top4 = (
        train.loc[train["track"].eq("Local")]
        .nsmallest(4, "rank_within_track")
        ["preferred_name"]
        .tolist()
    )

    cross_top3 = (
        train.loc[train["track"].eq("Cross")]
        .nsmallest(3, "rank_within_track")
        ["preferred_name"]
        .tolist()
    )

    assert local_top4 == ["SSE1", "APJ1", "ERJ5", "JJJ2"]
    assert cross_top3 == ["SSM4", "UFD2", "FES1"]

def test_shared_ranking_inputs(results):
    """Only community features may change between Louvain seed groups."""
    train = results["train"]
    valid = results["validation"].reindex(train.index)
    np.testing.assert_allclose(
        train[["N", "R"]], valid[["N", "R"]], rtol=0, atol=1e-12,
    )
