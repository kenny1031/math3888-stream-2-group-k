"""
Network construction and basic graph features.
Graphs are undirected and weighted using normalised
STRING confidence scores.
"""
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from .config import ExperimentConfig, load_config


# ============================================================
# 1. Load STRING edges
# ============================================================

def load_string_edges(
    path: str | Path,
    threshold: int = 700,
) -> pd.DataFrame:
    """
    Load and filter a processed STRING edge list.

    Required columns:
        u, v, combined_score

    Scores remain on the original 0-1000 scale.
    """
    edges = pd.read_csv(
        path,
        usecols=["u", "v", "combined_score"],
        dtype={"u": str, "v": str},
    )

    edges["combined_score"] = pd.to_numeric(
        edges["combined_score"],
        errors="raise",
    )

    if edges[["u", "v", "combined_score"]].isna().any().any():
        raise ValueError("STRING edge list contains missing values.")

    if not edges["combined_score"].between(0, 1000).all():
        raise ValueError("STRING scores must be between 0 and 1000.")

    edges = edges.loc[
        (edges["combined_score"] >= threshold)
        & (edges["u"] != edges["v"])
    ].copy()

    # Detect duplicate undirected edges.
    # Do not silently combine scores during refactoring.
    endpoints = np.sort(
        edges[["u", "v"]].to_numpy(),
        axis=1,
    )

    if pd.DataFrame(endpoints).duplicated().any():
        raise ValueError(
            "Duplicate undirected edges detected. "
            "Check the original Notebook cleaning policy."
        )

    return edges.reset_index(drop=True)


# ============================================================
# 2. Load protein identifiers
# ============================================================

def load_protein_ids(path: str | Path) -> list[str]:
    """
    Read unique STRING protein IDs from the mapping file.

    The mapping is used to preserve proteins that become
    isolated after thresholding.
    """
    mapping = pd.read_csv(path, dtype=str)
    possible_columns = [
        "string_protein_id",
        "string_id",
        "protein_id",
        "protein",
    ]

    id_col = next(
        (col for col in possible_columns if col in mapping.columns),
        None,
    )

    if id_col is None:
        raise ValueError(
            "Cannot identify STRING ID column. "
            f"Available columns: {list(mapping.columns)}"
        )

    ids = (
        mapping[id_col]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return ids


# ============================================================
# 3. General graph construction
# ============================================================

def build_weighted_graph(
    edges: pd.DataFrame,
    nodes: list[str] | set[str],
    confidence_divisor: float = 1000.0,
    weight_attribute: str = "confidence",
) -> nx.Graph:
    """
    Construct an undirected weighted NetworkX graph.
    All specified nodes are preserved, including isolates.
    """
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    # Only retain edges within the requested node universe.
    node_set = set(graph.nodes())

    selected = edges.loc[
        edges["u"].isin(node_set)
        & edges["v"].isin(node_set)
    ]

    graph.add_weighted_edges_from(
        ((u, v, float(score) / confidence_divisor)
        for u, v, score in selected[
            ["u", "v", "combined_score"]
        ].itertuples(index=False, name=None)),
        weight=weight_attribute,
    )

    return graph


# ============================================================
# 4. Functional network
# ============================================================

def build_functional_network(cfg: ExperimentConfig) -> nx.Graph:
    """
    Construct the full STRING functional network.
    """
    edges = load_string_edges(
        cfg.path("functional_edges"),
        threshold=cfg.network["confidence_threshold"],
    )

    protein_ids = load_protein_ids(cfg.path("protein_mapping"))

    graph = build_weighted_graph(
        edges=edges,
        nodes=protein_ids,
        confidence_divisor=cfg.network["confidence_divisor"],
        weight_attribute=cfg.network["weight_attribute"],
    )

    return graph


# ============================================================
# 5. Target connected component
# ============================================================

def get_target_component(graph: nx.Graph, target_id: str) -> nx.Graph:
    """
    Extract the connected component containing the target.

    In our project, this is the HLJ1 functional component.
    """
    if target_id not in graph:
        raise ValueError(f"Target protein {target_id} is not in the network.")

    nodes = nx.node_connected_component(graph, target_id)

    return graph.subgraph(nodes).copy()


# ============================================================
# 6. Physical network
# ============================================================

def build_physical_network(
    cfg: ExperimentConfig,
    nodes: list[str] | set[str],
) -> nx.Graph:
    """
    Construct the STRING physical association network
    using a specified node universe.

    Physical isolates are intentionally retained.
    """
    edges = load_string_edges(
        cfg.path("physical_edges"),
        threshold=cfg.network["confidence_threshold"],
    )

    return build_weighted_graph(
        edges=edges,
        nodes=nodes,
        confidence_divisor=cfg.network["confidence_divisor"],
        weight_attribute=cfg.network["weight_attribute"],
    )


# ============================================================
# 7. Basic network features
# ============================================================

def compute_network_features(
    graph: nx.Graph,
    target_id: str,
    weight_attribute: str = "confidence",
) -> pd.DataFrame:
    """
    Compute basic topological features for all proteins.

    Features:
        degree          : Number of direct neighbours
        weighted_degree : Sum of edge confidence weights
        h_hops          : Hop distance from target
    """

    if target_id not in graph:
        raise ValueError("Target protein not found in graph.")

    distances = nx.single_source_shortest_path_length(graph, target_id)
    features = pd.DataFrame(
        {
            "degree": dict(graph.degree()),
            "weighted_degree": dict(
                graph.degree(weight=weight_attribute)
            ),
        }
    )

    features.index.name = "string_id"
    features["h_hops"] = pd.Series(distances).reindex(features.index)

    return features


# ============================================================
# 8. Build both networks
# ============================================================

def build_networks(cfg: ExperimentConfig) -> tuple[nx.Graph,
                                                   nx.Graph,
                                                   nx.Graph]:
    """
    Build and return:
        1. Full functional network
        2. HLJ1 functional component
        3. Physical network restricted to HLJ1 component
    """
    target_id = cfg.project["target_protein"]
    full_graph = build_functional_network(cfg)
    target_graph = get_target_component(full_graph, target_id)
    physical_graph = build_physical_network(
        cfg,
        nodes=set(target_graph.nodes()),
    )

    return full_graph, target_graph, physical_graph


# ============================================================
# 9. Smoke test
# ============================================================

if __name__ == "__main__":

    cfg = load_config()

    cfg.check_inputs(
        "functional_edges",
        "physical_edges",
        "protein_mapping",
    )

    full, functional, physical = build_networks(cfg)
    target = cfg.project["target_protein"]

    print("\n--- Network Summary ---")
    print(
        f"Full functional: "
        f"{full.number_of_nodes():,} nodes, "
        f"{full.number_of_edges():,} edges"
    )
    print(
        f"Target component: "
        f"{functional.number_of_nodes():,} nodes, "
        f"{functional.number_of_edges():,} edges"
    )
    print(
        f"Physical layer: "
        f"{physical.number_of_nodes():,} nodes, "
        f"{physical.number_of_edges():,} edges"
    )
    print(f"HLJ1 functional degree: {functional.degree(target)}")
    print(f"HLJ1 physical degree: {physical.degree(target)}")

    features = compute_network_features(functional, target_id=target)

    print("\nHLJ1 features:")
    print(features.loc[target])

    # Regression checks against the original notebooks
    assert functional.number_of_nodes() == 5716
    assert functional.number_of_edges() == 104138
    assert functional.degree(target) == 25
    assert physical.degree(target) == 0

    print("\nNetwork regression checks passed!")