"""
Publication-ready figures.
All plotting functions:
    - Accept processed data
    - Save PNG and PDF
    - Return output paths
    - Do not modify experimental results
"""
from pathlib import Path
from collections.abc import Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import networkx as nx


TRACKS = ["Local", "Cross", "Boundary"]
METHODS = ["Degree", "Node2Vec", "RWR", "CADTP"]


def plot_go_precision(
    results: pd.DataFrame,
    output_dir: str | Path,
    k: int = 20,
) -> list[Path]:
    """
    Grouped bar chart comparing GO Precision@k
    across ranking methods and CADTP tracks.
    """
    required = {"track", "method", "k", "precision"}

    if not required.issubset(results.columns):
        raise ValueError(f"Missing columns: {required - set(results.columns)}")

    data = results.loc[results["k"].eq(k)].copy()

    if data.empty:
        raise ValueError(f"No Precision@{k} results found.")

    if data.duplicated(["track", "method"]).any():
        raise ValueError("Duplicate track/method combinations.")

    precision = (
        data.pivot(
            index="track",
            columns="method",
            values="precision",
        )
        .reindex(index=TRACKS, columns=METHODS)
    )

    if precision.isna().any().any():
        raise ValueError("Incomplete GO baseline comparison table.")

    if not precision.to_numpy().min() >= 0 or (precision.to_numpy().max() > 1):
        raise ValueError("Precision values must be in [0, 1].")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(TRACKS))
    width = 0.19

    for i, method in enumerate(METHODS):
        positions = x + (i - (len(METHODS) - 1) / 2) * width
        values = precision[method].to_numpy()
        bars = ax.bar(positions, values, width=width, label=method)
        ax.bar_label(
            bars,
            labels=[f"{value:.0%}" for value in values],
            padding=3,
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Local", "Cross-community", "Boundary"])
    ax.set_ylabel(f"Observed GO Precision@{k}")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Proteostasis-related GO Annotation Enrichment",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax.legend(ncol=4, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for extension in ("png", "pdf"):
        path = output_dir / f"go_precision_at_{k}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)

    plt.close(fig)

    return paths


# HLJ1 candidate network visualisation
def strongest_shortest_path(
    graph: nx.Graph,
    source: str,
    target: str,
    weight: str = "confidence",
) -> list[str]:
    """
    Select a representative path from all shortest paths.

    Priority:
        1. Minimum number of hops
        2. Maximum bottleneck edge confidence
        3. Maximum total edge confidence
        4. Lexicographic tie-breaking

    STRING confidence is used as connection evidence,
    not interpreted as a biological causal probability.
    """

    if source not in graph or target not in graph:
        raise ValueError("Source or target not in graph.")

    if source == target:
        return [source]

    if not nx.has_path(graph, source, target):
        raise nx.NetworkXNoPath(
            f"No path from {source} to {target}"
        )

    paths = nx.all_shortest_paths(
        graph,
        source=source,
        target=target,
        weight=None,  # Shortest means fewest hops
    )

    def path_key(path):
        confidences = [
            float(graph[u][v][weight])
            for u, v in zip(path[:-1], path[1:])
        ]

        return (
            -min(confidences),
            -sum(confidences),
            tuple(path),
        )

    return list(min(paths, key=path_key))


def plot_candidate_network(
    graph: nx.Graph,
    target_id: str,
    candidate_ids: list[str],
    protein_names: dict[str, str],
    output_dir: str | Path,
    file_stem: str,
    title: str,
    candidate_type: str = "Local",
    seed: int = 3888,
    weight: str = "confidence",
) -> tuple[list[Path], pd.DataFrame]:
    """
    Plot a compact functional association network
    connecting HLJ1 to selected candidate proteins.

    Edges are genuine STRING interactions.

    Returns:
        saved_paths : PNG and PDF paths
        evidence    : Shortest-path evidence table
    """

    if candidate_type not in ("Local", "Cross"):
        raise ValueError("Unknown candidate type.")

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate IDs.")

    if target_id in candidate_ids:
        raise ValueError("Target cannot be its own candidate.")

    # --------------------------------------------------------
    # A. Select strongest shortest paths
    # --------------------------------------------------------

    selected_graph = nx.Graph()
    selected_graph.add_node(target_id)

    evidence_records = []

    def gene_name(node):
        name = protein_names.get(node)

        if name is None or pd.isna(name):
            return node.split(".")[-1]

        return str(name)

    for candidate in candidate_ids:

        path = strongest_shortest_path(
            graph,
            source=target_id,
            target=candidate,
            weight=weight,
        )

        confidences = []

        for u, v in zip(path[:-1], path[1:]):

            selected_graph.add_edge(
                u, v, **graph[u][v]
            )

            confidences.append(
                float(graph[u][v][weight])
            )

        evidence_records.append({
            "candidate_id": candidate,
            "candidate_gene": gene_name(candidate),
            "hops": len(path) - 1,
            "path_ids": " -> ".join(path),
            "path_genes": " -> ".join(
                gene_name(v) for v in path
            ),
            "bottleneck_confidence": min(confidences),
            "mean_edge_confidence": np.mean(confidences),
        })

    evidence = pd.DataFrame(evidence_records)

    # --------------------------------------------------------
    # B. Node classification
    # --------------------------------------------------------

    candidate_set = set(candidate_ids)

    roles = {}

    for node in selected_graph.nodes():

        if node == target_id:
            roles[node] = "HLJ1"
        elif node in candidate_set:
            roles[node] = "Candidate"
        else:
            roles[node] = "Intermediate"

    colours = {
        "HLJ1": "#fca5a5",
        "Candidate": (
            "#93c5fd"
            if candidate_type == "Local"
            else "#fdba74"
        ),
        "Intermediate": "#cbd5e1",
    }

    sizes = {
        "HLJ1": 1800,
        "Candidate": 1500,
        "Intermediate": 1200,
    }

    # --------------------------------------------------------
    # C. Deterministic network layout
    # --------------------------------------------------------

    positions = nx.spring_layout(
        selected_graph,
        seed=seed,
        weight=weight,
        iterations=300,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    edges = list(selected_graph.edges())

    edge_widths = [
        1.0 + 3.5 * float(selected_graph[u][v][weight])
        for u, v in edges
    ]

    nx.draw_networkx_edges(
        selected_graph,
        positions,
        ax=ax,
        edgelist=edges,
        width=edge_widths,
        edge_color="#64748b",
        alpha=0.65,
    )

    for role in ("HLJ1", "Candidate", "Intermediate"):

        nodes = [
            v for v in selected_graph.nodes()
            if roles[v] == role
        ]

        if not nodes:
            continue

        nx.draw_networkx_nodes(
            selected_graph,
            positions,
            nodelist=nodes,
            node_size=sizes[role],
            node_color=colours[role],
            edgecolors="#475569",
            linewidths=1.3,
            ax=ax,
        )

    labels = {
        node: gene_name(node)
        for node in selected_graph.nodes()
    }

    nx.draw_networkx_labels(
        selected_graph,
        positions,
        labels=labels,
        font_size=10,
        font_weight="bold",
        font_color="#172554",
        ax=ax,
    )

    # --------------------------------------------------------
    # D. Legend and annotation
    # --------------------------------------------------------

    legend_handles = []

    for role in ("HLJ1", "Candidate", "Intermediate"):

        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=colours[role],
                markeredgecolor="#475569",
                markersize=12,
                label=role,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    ax.set_axis_off()
    ax.margins(0.25)

    fig.text(
        0.5,
        0.035,
        "STRING functional associations (score >= 700). "
        "Selected strongest shortest paths; "
        "edge width reflects confidence.",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    # --------------------------------------------------------
    # E. Save figures
    # --------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for extension in ("png", "pdf"):

        path = output_dir / f"{file_stem}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        saved_paths.append(path)

    plt.close(fig)

    return saved_paths, evidence



# ============================================================
# 3. Louvain community membership heatmap
# ============================================================

def plot_louvain_membership_heatmap(
    membership: pd.DataFrame,
    protein_ids: Sequence[str],
    protein_names: Mapping[str, str],
    tracks: Mapping[str, str],
    training_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    output_dir: str | Path,
    file_stem: str = "louvain_membership_heatmap",
) -> tuple[list[Path], pd.DataFrame]:
    """
    Visualise community membership across two independent
    Louvain ensembles.

    1 = In HLJ1 community
    0 = Outside HLJ1 community

    Returns:
        Figure paths (PNG/PDF)
        Membership summary for selected proteins
    """

    protein_ids = list(protein_ids)
    train_seeds = list(training_seeds)
    val_seeds = list(validation_seeds)
    all_seeds = train_seeds + val_seeds

    if not protein_ids or len(set(protein_ids)) != len(protein_ids):
        raise ValueError("Protein IDs must be nonempty and unique.")

    if not train_seeds or not val_seeds:
        raise ValueError("Both seed groups must be nonempty.")

    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Duplicate seeds.")

    if not set(protein_ids).issubset(membership.index):
        raise ValueError("Missing protein IDs.")

    if not set(all_seeds).issubset(membership.columns):
        raise ValueError("Missing Louvain seeds.")

    data = membership.loc[
        protein_ids, all_seeds
    ].astype(int)

    if not data.isin([0, 1]).all().all():
        raise ValueError("Membership values must be binary.")

    # --------------------------------------------------------
    # A. Membership summary
    # --------------------------------------------------------

    records = []

    for protein_id in protein_ids:
        train_votes = int(data.loc[protein_id, train_seeds].sum())
        val_votes = int(data.loc[protein_id, val_seeds].sum())
        gene = str(protein_names.get(protein_id, protein_id))
        track = str(tracks.get(protein_id, "Unclassified"))
        records.append({
            "string_id": protein_id,
            "gene": gene,
            "training_track": track,
            "training_votes": train_votes,
            "validation_votes": val_votes,
            "training_frequency": train_votes / len(train_seeds),
            "validation_frequency": val_votes / len(val_seeds),
        })

    summary = pd.DataFrame(records)

    # --------------------------------------------------------
    # B. Plot
    # --------------------------------------------------------

    cmap = ListedColormap([
        "#e8edf2",  # Not in HLJ1 community
        "#28688e",  # In HLJ1 community
    ])

    height = max(5.5, len(protein_ids) * 0.52 + 1.5)

    fig, ax = plt.subplots(figsize=(14, height))

    ax.imshow(
        data.to_numpy(),
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    # --------------------------------------------------------
    # C. Axis labels
    # --------------------------------------------------------

    row_labels = []

    for row in summary.itertuples(index=False):
        row_labels.append(
            f"{row.gene} | {row.training_track}"
            f" ({row.training_votes}/{len(train_seeds)}"
            f" → {row.validation_votes}/{len(val_seeds)})"
        )

    ax.set_yticks(np.arange(len(protein_ids)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(np.arange(len(all_seeds)))

    ax.set_xticklabels(
        [str(s) for s in all_seeds],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    # --------------------------------------------------------
    # D. Cell borders and ensemble separator
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(-0.5, len(all_seeds), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(protein_ids), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=0.8,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Separate training and validation ensembles.
    separator = len(train_seeds) - 0.5

    ax.axvline(
        separator,
        color="#111827",
        linewidth=2.5,
    )

    # --------------------------------------------------------
    # E. Group labels and legend
    # --------------------------------------------------------

    total_runs = len(all_seeds)

    train_centre = (
        len(train_seeds) / 2 / total_runs
    )

    val_centre = (
        len(train_seeds)
        + len(val_seeds) / 2
    ) / total_runs

    ax.text(
        train_centre, 1.035,
        "Training ensemble",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        val_centre, 1.035,
        "Held-out validation",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Louvain random seed",
        fontsize=11,
        labelpad=10,
    )

    ax.set_title(
        "Consensus Membership in the HLJ1 Community",
        fontsize=15,
        fontweight="bold",
        pad=50,
    )

    legend = [
        Patch(
            facecolor="#28688e",
            label="In HLJ1 community",
        ),
        Patch(
            facecolor="#e8edf2",
            edgecolor="#9ca3af",
            label="Outside HLJ1 community",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.57, 0.005),
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.33,
        right=0.96,
        top=0.82,
        bottom=0.17,
    )

    # --------------------------------------------------------
    # F. Save PNG and PDF
    # --------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for extension in ("png", "pdf"):

        path = output_dir / f"{file_stem}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        saved_paths.append(path)

    plt.close(fig)

    return saved_paths, summary
