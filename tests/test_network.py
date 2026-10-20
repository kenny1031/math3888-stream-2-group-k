"""
Tests for STRING network construction.
Run from repository root:
    python -m pytest tests/test_network.py -v
"""
import networkx as nx
import pandas as pd
import pytest
from src.config import load_config
from src.network import (
    load_string_edges,
    build_weighted_graph,
    get_target_component,
    compute_network_features,
    build_networks,
)


# ============================================================
# 1. STRING edge filtering
# ============================================================

def test_load_string_edges(tmp_path):
    data = pd.DataFrame({
        "u": ["A", "A", "B"],
        "v": ["B", "C", "B"],
        "combined_score": [735, 650, 900],
    })

    path = tmp_path / "edges.csv"
    data.to_csv(path, index=False)

    edges = load_string_edges(
        path,
        threshold=700,
    )

    # Only A-B should remain
    assert len(edges) == 1

    assert edges.iloc[0]["u"] == "A"
    assert edges.iloc[0]["v"] == "B"
    assert edges.iloc[0]["combined_score"] == 735


# ============================================================
# 2. Weighted graph construction
# ============================================================

def test_build_weighted_graph():
    edges = pd.DataFrame({
        "u": ["A", "B"],
        "v": ["B", "C"],
        "combined_score": [735, 900],
    })

    G = build_weighted_graph(
        edges,
        nodes=["A", "B", "C", "D"],
    )

    assert isinstance(G, nx.Graph)

    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 2

    # Confidence normalisation
    assert G["A"]["B"]["confidence"] == pytest.approx(0.735)
    assert G["B"]["C"]["confidence"] == pytest.approx(0.900)

    # Preserve isolated nodes
    assert "D" in G
    assert G.degree("D") == 0


# ============================================================
# 3. Target connected component
# ============================================================

def test_target_component():
    G = nx.Graph()
    G.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("D", "E"),
    ])

    component = get_target_component(
        G,
        target_id="A",
    )

    assert set(component.nodes()) == {"A", "B", "C"}
    assert component.number_of_edges() == 2


# ============================================================
# 4. Network features
# ============================================================

def test_network_features():
    G = nx.Graph()
    G.add_edge("A", "B", confidence=0.8)
    G.add_edge("B", "C", confidence=0.9)

    features = compute_network_features(
        G,
        target_id="A",
    )

    assert features.loc["B", "degree"] == 2

    assert features.loc[
        "B", "weighted_degree"
    ] == pytest.approx(1.7)

    assert features.loc["C", "h_hops"] == 2


# ============================================================
# 5. Full dataset regression
# ============================================================

def test_hlj1_network_regression():
    cfg = load_config()
    full, functional, physical = build_networks(cfg)
    target = cfg.project["target_protein"]

    assert full.number_of_nodes() == 6600
    assert full.number_of_edges() == 104188

    assert functional.number_of_nodes() == 5716
    assert functional.number_of_edges() == 104138

    assert physical.number_of_nodes() == 5716
    assert physical.number_of_edges() == 43028

    assert functional.degree(target) == 25
    assert physical.degree(target) == 0

    assert functional.degree(
        target, weight="confidence"
    ) == pytest.approx(20.444, abs=1e-3)

    assert set(functional.nodes()) == set(physical.nodes())
