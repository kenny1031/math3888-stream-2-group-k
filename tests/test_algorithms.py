"""
Unit tests for core graph algorithms.
Run:
    python -m pytest tests/test_algorithms.py -v
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from src.algorithms import (
    compute_rwr,
    run_louvain,
    run_louvain_ensemble,
    community_membership_counts,
    compute_community_affinity,
    mean_community_affinity,
    rank_percentile,
    community_membership_matrix,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_graph():
    """Small weighted network with two communities."""
    G = nx.Graph()

    # Community 1
    G.add_edge("A", "B", confidence=0.9)
    G.add_edge("B", "C", confidence=0.8)
    G.add_edge("A", "C", confidence=0.9)

    # Community 2
    G.add_edge("D", "E", confidence=0.9)
    G.add_edge("E", "F", confidence=0.8)
    G.add_edge("D", "F", confidence=0.9)

    # Weak connection
    G.add_edge("C", "D", confidence=0.05)

    return G


# ============================================================
# 1. RWR
# ============================================================

def test_rwr_probability_distribution(small_graph):
    scores = compute_rwr(
        small_graph,
        target_id="A",
        alpha=0.85,
    )
    assert len(scores) == small_graph.number_of_nodes()
    assert scores.sum() == pytest.approx(1.0)
    assert (scores >= 0).all()
    assert np.isfinite(scores).all()


def test_rwr_stationary_equation(small_graph):
    """Verify the personalised PageRank fixed-point equation."""
    nodes = sorted(small_graph.nodes())
    scores = compute_rwr(
        small_graph,
        target_id="A",
        alpha=0.85,
    ).reindex(nodes)

    W = nx.to_numpy_array(
        small_graph,
        nodelist=nodes,
        weight="confidence",
    )

    strength = W.sum(axis=1)
    P = W / strength[:, None]

    r = scores.to_numpy()

    restart = np.zeros(len(nodes))
    restart[nodes.index("A")] = 1.0

    expected = 0.85 * (P.T @ r) + 0.15 * restart

    np.testing.assert_allclose(
        r,
        expected,
        atol=1e-5,
    )


# ============================================================
# 2. Louvain
# ============================================================

def test_louvain_partition(small_graph):
    communities = run_louvain(small_graph, seed=3888,)

    # Every node must appear in exactly one community
    all_nodes = [
        node
        for community in communities
        for node in community
    ]

    assert len(all_nodes) == small_graph.number_of_nodes()
    assert set(all_nodes) == set(small_graph.nodes())


def test_louvain_ensemble(small_graph):
    seeds = [3888, 3889, 3890]
    result = run_louvain_ensemble(
        small_graph,
        target_id="A",
        seeds=seeds,
    )

    assert set(result) == set(seeds)
    assert all(
        "A" in community
        for community in result.values()
    )


# ============================================================
# 3. Community membership
# ============================================================

def test_membership_counts():

    communities = {
        1: {"A", "B"},
        2: {"A", "C"},
        3: {"A", "B"},
    }

    result = community_membership_counts(
        communities,
        nodes=["A", "B", "C", "D"],
    )

    assert result.loc["A", "membership_count"] == 3
    assert result.loc["B", "membership_count"] == 2
    assert result.loc["C", "membership_count"] == 1
    assert result.loc["D", "membership_count"] == 0
    assert result.loc["B", "membership_frequency"] == pytest.approx(2 / 3)


# ============================================================
# 4. Community affinity
# ============================================================

def test_community_affinity_formula():
    G = nx.Graph()
    G.add_edge("A", "B", confidence=0.8)
    G.add_edge("B", "C", confidence=0.4)

    community = {"A", "B"}

    result = compute_community_affinity(
        G,
        community=community,
        smoothing=1.0,
    )

    # Weighted degrees:
    # A = 0.8, B = 1.2, C = 0.4
    # Total weighted volume = 2.4
    # Community prior = 2.0 / 2.4

    prior = 2.0 / 2.4

    expected_A = (0.8 + prior) / (0.8 + 1.0)
    expected_C = (0.4 + prior) / (0.4 + 1.0)

    assert result["A"] == pytest.approx(expected_A)
    assert result["C"] == pytest.approx(expected_C)


def test_mean_community_affinity():
    G = nx.Graph()
    G.add_edge("A", "B", confidence=0.8)
    G.add_edge("B", "C", confidence=0.4)

    communities = {1: {"A", "B"}, 2: {"B", "C"}}

    mean_result = mean_community_affinity(
        G,
        communities,
        smoothing=1.0,
    )

    first = compute_community_affinity(
        G, {"A", "B"}, smoothing=1.0
    )

    second = compute_community_affinity(
        G, {"B", "C"}, smoothing=1.0
    )

    expected = (first + second) / 2.0

    pd.testing.assert_series_equal(
        mean_result.sort_index(),
        expected.sort_index().rename("mean_community_affinity"),
    )


# ============================================================
# 5. Rank normalisation
# ============================================================

def test_rank_percentile():

    scores = pd.Series({
        "A": 10.0,
        "B": 10.0,
        "C": 5.0,
        "D": 1.0,
    })

    result = rank_percentile(scores)

    # A and B tie for first, with average rank 1.5
    assert result["A"] == pytest.approx(5 / 6)
    assert result["B"] == pytest.approx(5 / 6)
    assert result["C"] == pytest.approx(1 / 3)
    assert result["D"] == pytest.approx(0.0)
    assert result.between(0, 1).all()


# ============================================================
# 6. Invalid input handling
# ============================================================

def test_invalid_parameters(small_graph):
    with pytest.raises(ValueError):
        compute_rwr(
            small_graph,
            target_id="A",
            alpha=1.5,
        )

    with pytest.raises(ValueError):
        run_louvain_ensemble(
            small_graph,
            target_id="A",
            seeds=[1, 1],
        )

    with pytest.raises(ValueError):
        compute_community_affinity(
            small_graph,
            community={"A"},
            smoothing=-1,
        )


def test_community_membership_matrix():

    communities = {
        1: {"H", "A"},
        2: {"H", "B"},
        3: {"H", "A", "B"},
    }

    matrix = community_membership_matrix(
        communities,
        nodes=["H", "A", "B", "C"],
    )

    assert matrix.shape == (4, 3)

    assert matrix.loc["H"].tolist() == [1, 1, 1]
    assert matrix.loc["A"].tolist() == [1, 0, 1]
    assert matrix.loc["B"].tolist() == [0, 1, 1]
    assert matrix.loc["C"].tolist() == [0, 0, 0]
