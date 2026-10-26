"""
Core graph algorithms.
Includes:
    - Personalised PageRank / Random Walk with Restart
    - Louvain community detection
    - Community membership frequency
    - Community affinity
    - Rank-percentile normalisation

CADTP-specific ranking logic belongs in src/cadtp.py.
"""
from collections.abc import Mapping, Sequence
import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# 1. Random Walk with Restart
# ============================================================

def compute_rwr(
    graph: nx.Graph,
    target_id: str,
    alpha: float = 0.85,
    weight: str = "confidence",
) -> pd.Series:
    """
    Compute personalised PageRank centred at target_id.

    alpha:
        Probability of following a graph edge.

    Returns:
        Raw stationary probabilities, NOT percentiles.
    """
    if target_id not in graph:
        raise ValueError(f"Target not found: {target_id}")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization={target_id: 1.0},
        weight=weight,
        tol=1e-6,
        max_iter=100,
    )

    return pd.Series(
        scores,
        name="rwr_probability",
        dtype=float,
    )


# ============================================================
# 2. Louvain community detection
# ============================================================

def run_louvain(
    graph: nx.Graph,
    seed: int,
    resolution: float = 1.0,
    weight: str = "confidence",
) -> list[set[str]]:
    """
    Run Louvain once, returning the full partition.
    """

    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    communities = nx.community.louvain_communities(
        graph,
        seed=seed,
        resolution=resolution,
        weight=weight,
    )

    return [set(c) for c in communities]


def run_louvain_ensemble(
    graph: nx.Graph,
    target_id: str,
    seeds: Sequence[int],
    resolution: float = 1.0,
    weight: str = "confidence",
) -> dict[int, set[str]]:
    """
    Run Louvain with multiple independent random seeds.

    For each run, retain only the community containing
    target_id.

    Returns:
        {seed: target_community}
    """
    if target_id not in graph:
        raise ValueError(f"Target not found: {target_id}")

    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique.")

    results = {}

    for seed in seeds:
        communities = run_louvain(
            graph,
            seed=int(seed),
            resolution=resolution,
            weight=weight,
        )

        target_community = next(
            c for c in communities
            if target_id in c
        )

        results[int(seed)] = target_community

    return results


# ============================================================
# 3. Community membership statistics
# ============================================================

def community_membership_counts(
    communities: Mapping[int, set[str]],
    nodes: Sequence[str],
) -> pd.DataFrame:
    """
    Count how often each protein belongs to the target
    community across Louvain runs.

    Integer votes should be used for CADTP classification
    to avoid floating-point boundary errors.
    """
    if not communities:
        raise ValueError("No Louvain communities supplied.")

    node_ids = list(nodes)

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Node identifiers must be unique.")

    votes = np.zeros(len(node_ids), dtype=int)

    for members in communities.values():
        votes += np.fromiter(
            (v in members for v in node_ids),
            dtype=int,
            count=len(node_ids),
        )

    n_runs = len(communities)

    result = pd.DataFrame(
        {
            "membership_count": votes,
            "num_runs": n_runs,
            "membership_frequency": votes / n_runs,
        },
        index=pd.Index(node_ids, name="string_id"),
    )

    return result


# ============================================================
# 4. Community affinity
# ============================================================

def _prepare_affinity_workspace(graph: nx.Graph, weight: str):
    """
    Build the sparse weighted adjacency matrix once.

    Internal helper shared by single-run and ensemble
    community affinity calculations.
    """

    nodes = sorted(graph.nodes())

    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        weight=weight,
        format="csr",
        dtype=float,
    )

    strength = np.asarray(adjacency.sum(axis=1)).ravel()
    total_volume = strength.sum()

    if total_volume <= 0:
        raise ValueError("Graph has no positive weighted volume.")

    return nodes, adjacency, strength, total_volume


def _affinity_from_workspace(
    community: set[str],
    nodes: list[str],
    adjacency,
    strength: np.ndarray,
    total_volume: float,
    smoothing: float,
) -> np.ndarray:
    """
    Compute smoothed affinity to one target community.
    """
    member = np.fromiter(
        (v in community for v in nodes),
        dtype=float,
        count=len(nodes),
    )

    # Total edge weight from each node into the community
    links_to_community = np.asarray(adjacency @ member).ravel()

    # Community's share of the graph's weighted volume
    community_prior = np.dot(strength, member) / total_volume
    denominator = strength + smoothing
    if np.any(denominator <= 0):
        raise ValueError(
            "Affinity denominator is zero. "
            "Use positive smoothing for isolated nodes."
        )

    return (
        links_to_community + smoothing * community_prior
    ) / denominator


def compute_community_affinity(
    graph: nx.Graph,
    community: set[str],
    smoothing: float = 5.0,
    weight: str = "confidence",
) -> pd.Series:
    """
    Compute smoothed community affinity for one partition.

    Higher affinity indicates that a larger share of a
    protein's weighted connections points towards the
    supplied community.

    Returns raw affinity, not percentile scores.
    """
    if smoothing < 0:
        raise ValueError("smoothing cannot be negative.")

    nodes, adjacency, strength, volume = (
        _prepare_affinity_workspace(graph, weight)
    )

    affinity = _affinity_from_workspace(
        community,
        nodes,
        adjacency,
        strength,
        volume,
        smoothing,
    )

    return pd.Series(
        affinity,
        index=pd.Index(nodes, name="string_id"),
        name="community_affinity",
    )


def mean_community_affinity(
    graph: nx.Graph,
    communities: Mapping[int, set[str]],
    smoothing: float = 5.0,
    weight: str = "confidence",
) -> pd.Series:
    """
    Average RAW community affinities across Louvain runs.

    IMPORTANT:
        Average raw affinities first.
        Apply rank_percentile() only afterwards.

    The sparse adjacency matrix is constructed once.
    """
    if not communities:
        raise ValueError("No Louvain communities supplied.")

    if smoothing < 0:
        raise ValueError("smoothing cannot be negative.")

    nodes, adjacency, strength, volume = (
        _prepare_affinity_workspace(graph, weight)
    )

    total_affinity = np.zeros(len(nodes), dtype=float)
    for community in communities.values():
        total_affinity += _affinity_from_workspace(
            community,
            nodes,
            adjacency,
            strength,
            volume,
            smoothing,
        )

    mean_affinity = total_affinity / len(communities)

    return pd.Series(
        mean_affinity,
        index=pd.Index(nodes, name="string_id"),
        name="mean_community_affinity",
    )


# ============================================================
# 5. Rank-percentile normalisation
# ============================================================

def rank_percentile(scores: pd.Series) -> pd.Series:
    """
    Convert scores into descending rank percentiles.

    Highest score -> near 1
    Lowest score  -> near 0

    Ties receive average ranks.

    Important:
        Apply to the full scoring universe before
        filtering eligible candidates.
    """
    if scores.empty:
        raise ValueError("Score series is empty.")

    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Scores contain NaN or infinity.")

    n = len(scores)

    if n == 1:
        return pd.Series(
            1.0,
            index=scores.index,
            name="rank_percentile",
        )

    ranks = scores.rank(ascending=False, method="average")
    result = 1.0 - (ranks - 1.0) / (n - 1)

    return result.rename("rank_percentile")


# ============================================================
# Louvain membership matrix
# ============================================================

def community_membership_matrix(
    communities: Mapping[int, set[str]],
    nodes: Sequence[str],
) -> pd.DataFrame:
    """
    Construct a binary membership matrix.

    Rows:
        STRING protein IDs

    Columns:
        Louvain random seeds

    Entry (v, s):
        1 if protein v belongs to HLJ1's community
        in run s, otherwise 0.
    """

    if not communities:
        raise ValueError("No communities provided.")

    nodes = list(nodes)

    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate protein IDs.")

    matrix = pd.DataFrame(
        {
            int(seed): np.fromiter(
                (int(v in members) for v in nodes),
                dtype=np.uint8,
                count=len(nodes),
            )
            for seed, members in communities.items()
        },
        index=pd.Index(nodes, name="string_id"),
    )

    return matrix


# ============================================================
# 6. Smoke test
# ============================================================

if __name__ == "__main__":
    from .config import load_config
    from .network import build_networks

    cfg = load_config()
    G_full, G_hlj1, _ = build_networks(cfg)
    target = cfg.project["target_protein"]

    # RWR on target component
    rwr = compute_rwr(
        G_hlj1,
        target,
        alpha=cfg.rwr["alpha"],
        weight=cfg.rwr["weight"],
    )

    assert np.isclose(rwr.sum(), 1.0)

    print("\nRWR computed successfully.")
    print("RWR probability sum:", rwr.sum())

    # Only two Louvain runs for a quick smoke test.
    # Full experiments will use all configured seeds.
    test_seeds = cfg.louvain["training_seeds"][:2]
    communities = run_louvain_ensemble(
        G_full,
        target,
        seeds=test_seeds,
        resolution=cfg.louvain["resolution"],
        weight=cfg.louvain["weight"],
    )

    for seed, community in communities.items():
        print(
            f"Louvain seed {seed}: "
            f"{len(community)} proteins in HLJ1 community"
        )

    membership = community_membership_counts(
        communities,
        nodes=list(G_hlj1.nodes()),
    )

    assert membership.loc[
        target, "membership_frequency"
    ] == 1.0

    affinity = mean_community_affinity(
        G_hlj1,
        communities,
        smoothing=cfg.cadtp["affinity_smoothing"],
    )

    assert len(affinity) == G_hlj1.number_of_nodes()
    assert affinity.notna().all()

    # Match the original 5,715 non-HLJ1 scoring universe
    affinity_scores = rank_percentile(affinity.drop(index=target))

    print("\nTop-5 community affinity proteins:")
    print(
        affinity_scores.sort_values(
            ascending=False
        ).head()
    )

    print("\nAlgorithm smoke tests passed!")