import pandas as pd
import networkx as nx
from src.plotting import (
    plot_go_precision,
    strongest_shortest_path,
    plot_candidate_network,
    plot_louvain_membership_heatmap,
    TRACKS,
    METHODS
)


def test_go_precision_plot(tmp_path):
    rows = []
    for track in TRACKS:
        for method in METHODS:
            rows.append({
                "track": track,
                "method": method,
                "k": 20,
                "precision": 0.5,
            })

    results = pd.DataFrame(rows)
    paths = plot_go_precision(results, output_dir=tmp_path, k=20)

    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_strongest_shortest_path():
    G = nx.Graph()
    G.add_edge("H", "A", confidence=0.9)
    G.add_edge("A", "T", confidence=0.7)
    G.add_edge("H", "B", confidence=0.8)
    G.add_edge("B", "T", confidence=0.8)

    # Both have two hops.
    # H-B-T has the stronger bottleneck (0.8).
    assert strongest_shortest_path(
        G, "H", "T"
    ) == ["H", "B", "T"]


def test_candidate_network_plot(tmp_path):
    G = nx.Graph()
    G.add_edge("H", "A", confidence=0.9)
    G.add_edge("A", "B", confidence=0.8)

    paths, evidence = plot_candidate_network(
        graph=G,
        target_id="H",
        candidate_ids=["B"],
        protein_names={
            "H": "HLJ1",
            "A": "INTERMEDIATE",
            "B": "CANDIDATE",
        },
        output_dir=tmp_path,
        file_stem="test_network",
        title="Test Network",
    )

    assert all(path.is_file() for path in paths)
    assert len(evidence) == 1
    assert evidence.iloc[0]["hops"] == 2
    assert evidence.iloc[0]["path_genes"] == "HLJ1 -> INTERMEDIATE -> CANDIDATE"


def test_louvain_heatmap(tmp_path):

    membership = pd.DataFrame(
        {
            1: [1, 1, 0],
            2: [1, 1, 0],
            3: [1, 0, 1],
            4: [1, 1, 0],
        },
        index=["H", "A", "B"],
    )

    paths, summary = plot_louvain_membership_heatmap(
        membership=membership,
        protein_ids=["H", "A", "B"],
        protein_names={
            "H": "HLJ1",
            "A": "SSE1",
            "B": "TIM21",
        },
        tracks={
            "H": "Anchor",
            "A": "Local",
            "B": "Boundary",
        },
        training_seeds=[1, 2],
        validation_seeds=[3, 4],
        output_dir=tmp_path,
    )

    assert len(paths) == 2
    assert all(path.is_file() for path in paths)

    assert summary.loc[
        summary["gene"].eq("SSE1"),
        "training_votes",
    ].iloc[0] == 2

    assert summary.loc[
        summary["gene"].eq("TIM21"),
        "validation_votes",
    ].iloc[0] == 1