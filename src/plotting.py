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
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import networkx as nx


TRACKS = ["Local", "Cross", "Boundary"]
METHODS = ["Degree", "Node2Vec", "RWR", "CADTP"]



def _normalise_tracks(track: pd.Series) -> pd.Series:
    """Map stored evaluation track keys to report-friendly labels."""
    mapping = {
        "A_local_consensus": "Local",
        "B_cross_consensus": "Cross",
        "C_boundary": "Boundary",
        "C_boundary_consensus": "Boundary",
        "Local": "Local",
        "Cross": "Cross",
        "Boundary": "Boundary",
    }
    values = track.astype(str).map(mapping)
    if values.isna().any():
        unknown = sorted(track[values.isna()].astype(str).unique())
        raise ValueError(f"Unknown track labels: {unknown}")
    return values


def _save_figure(fig, output_dir: str | Path, stem: str) -> list[Path]:
    """Write vector PDF and 300-dpi PNG, then release the figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_go_precision(
    results: pd.DataFrame,
    output_dir: str | Path,
    k: int = 20,
) -> list[Path]:
    """
    Figure 4.2: GO Precision@k in separate CADTP tracks.

    Compatible with both:
      - original plotting schema: track, method, k, precision
      - actual evaluation CSV:   track, method, k, precision_at_k,
                                 hits, background_rate

    Only tracks present in the input are shown. Missing Boundary results
    are NOT fabricated or treated as zero.
    """
    required = {"track", "method", "k"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"GO comparison missing columns: {sorted(missing)}")

    value_key = (
        "precision_at_k" if "precision_at_k" in results
        else "precision" if "precision" in results
        else None
    )
    if value_key is None:
        raise ValueError("GO comparison must contain precision_at_k or precision.")

    data = results.loc[results["k"].eq(k)].copy()
    if data.empty:
        raise ValueError(f"No Precision@{k} results.")
    data["track_label"] = _normalise_tracks(data["track"])

    if data.duplicated(["track_label", "method"]).any():
        raise ValueError("Duplicate GO track/method combinations.")
    present_tracks = [track for track in TRACKS
                      if track in data["track_label"].unique()]
    if not present_tracks:
        raise ValueError("No recognised GO track results.")

    missing_methods = {
        track: sorted(set(METHODS) - set(data.loc[
            data["track_label"].eq(track), "method"]))
        for track in present_tracks
    }
    if any(missing_methods.values()):
        raise ValueError(f"Missing GO methods: {missing_methods}")
    precision = data.pivot(
        index="track_label", columns="method", values=value_key
    ).reindex(index=present_tracks, columns=METHODS).astype(float)
    if not np.isfinite(precision.to_numpy()).all():
        raise ValueError("GO precision contains missing/non-finite values.")
    if ((precision < 0) | (precision > 1)).any().any():
        raise ValueError("GO precision must lie in [0,1].")

    colours = {
        "Degree": "#94a3b8",
        "Node2Vec": "#2563eb",
        "RWR": "#f59e0b",
        "CADTP": "#0d9488",
    }
    fig, axs = plt.subplots(
        1, len(present_tracks), figsize=(5.3 * len(present_tracks), 4.6),
        sharey=True, squeeze=False,
    )
    for ax, track in zip(axs[0], present_tracks):
        subset = data.loc[data["track_label"].eq(track)].set_index("method")
        for i, method in enumerate(METHODS):
            value = float(precision.loc[track, method])
            bars = ax.bar(
                i, value, width=0.66, color=colours[method], zorder=3,
            )
            label = f"{value:.0%}"
            if "hits" in subset.columns and pd.notna(subset.loc[method, "hits"]):
                label += f"\n({int(subset.loc[method, 'hits'])}/{k})"
            ax.text(i, value + 0.022, label, ha="center", va="bottom",
                    fontsize=9, linespacing=1.05)

        if "background_rate" in subset.columns:
            bg = subset["background_rate"].dropna().astype(float).unique()
            if len(bg) == 1:
                ax.axhline(bg[0], color="#475569", linestyle="--",
                           linewidth=1.2, zorder=2)
                ax.text(0.98, bg[0] + 0.015, f"Background: {bg[0]:.1%}",
                        transform=ax.get_yaxis_transform(), ha="right",
                        va="bottom", fontsize=8, color="#475569")

        ax.set_xticks(range(len(METHODS)), METHODS, fontsize=10)
        ax.set_title(f"{track} candidates", fontweight="bold", pad=10)
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(axis="y", linestyle=":", alpha=0.27)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axs[0, 0].set_ylabel(f"GO Precision@{k}")
    fig.suptitle("Proteostasis GO annotation among top-ranked candidates",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(
        0.5, -0.015,
        "Annotations measure consistency with existing GO knowledge, "
        "not experimental effects of HLJ1 deletion.",
        ha="center", fontsize=8.5, color="#475569",
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, f"go_precision_at_{k}")


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
        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

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
            selected_graph.add_edge(u, v, **graph[u][v])
            confidences.append(float(graph[u][v][weight]))

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
        fig.savefig(path, dpi=300, bbox_inches="tight")
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

    data = membership.loc[protein_ids, all_seeds].astype(int)

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

    ax.set_xticks(np.arange(-0.5, len(all_seeds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(protein_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Separate training and validation ensembles.
    separator = len(train_seeds) - 0.5
    ax.axvline(separator, color="#111827", linewidth=2.5)

    # --------------------------------------------------------
    # E. Group labels and legend
    # --------------------------------------------------------

    total_runs = len(all_seeds)
    train_centre = len(train_seeds) / 2 / total_runs
    val_centre = (len(train_seeds) + len(val_seeds) / 2) / total_runs

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
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)

    plt.close(fig)

    return saved_paths, summary


# ============================================================
# 4. Cross-method evidence heatmap (Results Fig. 4.1)
# ============================================================

DEFAULT_EVIDENCE_GENES = (
    "SSE1", "APJ1", "JJJ2", "ERJ5", "MDJ1",
    "UFD2", "SSM4", "FES1", "YOS9", "USA1",
)


def _numeric_column(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        raise ValueError(f"Integrated evidence missing column: {name}")
    return pd.to_numeric(data[name], errors="coerce")


def _selected_evidence(
    integrated: pd.DataFrame,
    genes: Sequence[str] | None,
) -> pd.DataFrame:
    """Preserve caller-specified gene order, without inventing missing data."""
    required = {"gene", "string_id", "cadtp_track"}
    missing = required - set(integrated.columns)
    if missing:
        raise ValueError(f"Integrated evidence missing columns: {sorted(missing)}")
    if integrated["string_id"].duplicated().any():
        raise ValueError("Duplicate STRING IDs in integrated evidence.")
    genes = list(DEFAULT_EVIDENCE_GENES if genes is None else genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Requested genes must be nonempty and unique.")
    rows = []
    for gene in genes:
        matches = integrated.loc[integrated["gene"].eq(gene)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one integrated row for {gene}, "
                f"found {len(matches)}."
            )
        rows.append(matches.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_cross_method_evidence_heatmap(
    integrated: pd.DataFrame,
    output_dir: str | Path,
    genes: Sequence[str] | None = None,
    file_stem: str = "cross_method_evidence_heatmap",
) -> list[Path]:
    """
    Figure 4.1: concise per-protein evidence matrix.

    Coloured columns retain their OWN normalisation and annotated units:
      CADTP: within-track rank, shaded by percentile within THAT track
      MCL:   number of six tested settings sharing HLJ1's cluster
      Deletion: exact normalised betweenness change, % of post score
      ERAD:  relative seed-personalised PageRank change, % of pre score
      Screen: rule-based shortlist membership, NOT an independent label

    Separate colormaps deliberately avoid falsely averaging unrelated scores.
    """
    selected = _selected_evidence(integrated, genes)
    numeric = [
        "cadtp_rank_within_track",
        "mcl_coclustering_count",
        "deletion_betweenness_pct_to_after",
        "erad_pagerank_pct_change",
    ]
    for column in numeric:
        selected[column] = _numeric_column(selected, column)
    if "erad_screen_selected" not in selected.columns:
        raise ValueError("Integrated evidence missing erad_screen_selected.")

    screen = selected["erad_screen_selected"].map({
        True: 1, False: 0, 1: 1, 0: 0,
        "True": 1, "False": 0, "true": 1, "false": 0,
    })
    if screen.isna().any():
        raise ValueError("Unrecognised ERAD selection indicator.")

    # Rank percentile is within each CADTP track; NOT a global rank.
    track_sizes = integrated.groupby("cadtp_track").size()
    track_n = selected["cadtp_track"].map(track_sizes).astype(float)
    rank = selected["cadtp_rank_within_track"].astype(float)
    rank_shade = (track_n - rank) / (track_n - 1).clip(lower=1)
    if ((rank_shade < 0) | (rank_shade > 1)).any():
        raise ValueError("CADTP ranks exceed track sizes.")

    mcl = selected["mcl_coclustering_count"].to_numpy(dtype=float)
    deletion = selected["deletion_betweenness_pct_to_after"].to_numpy(dtype=float)
    erad_pr = selected["erad_pagerank_pct_change"].to_numpy(dtype=float)

    def signed_norm(values):
        finite = np.asarray(values)[np.isfinite(values)]
        limit = max(1.0, float(np.max(np.abs(finite)))) if len(finite) else 1.0
        return SymLogNorm(
            linthresh=min(0.35, limit / 3), linscale=0.75,
            vmin=-limit, vmax=limit,
        )

    columns = [
        (
            rank_shade.to_numpy(dtype=float),
            plt.get_cmap("Blues"),
            Normalize(0, 1),
            [f"#{int(x)}" if np.isfinite(x) else "NA" for x in rank],
            "CADTP\nrank within track",
            "Within-track rank percentile",
        ),
        (
            mcl, plt.get_cmap("YlGnBu"), Normalize(0, 6),
            [f"{int(v)}/6" if np.isfinite(v) else "NA" for v in mcl],
            "MCL\nco-clustering", "Co-clustering count",
        ),
        (
            deletion, plt.get_cmap("RdBu_r"), signed_norm(deletion),
            [f"{v:+.2f}%" if np.isfinite(v) else "NA" for v in deletion],
            "Exact betweenness\npost-relative change",
            "Signed change (%)",
        ),
        (
            erad_pr, plt.get_cmap("RdBu_r"), signed_norm(erad_pr),
            [f"{v:+.2f}%" if np.isfinite(v) else "NA" for v in erad_pr],
            "ERAD PageRank\npre-relative change",
            "Signed change (%)",
        ),
        (
            screen.to_numpy(dtype=float),
            ListedColormap(["#e5e7eb", "#86efac"]), Normalize(0, 1),
            ["Yes" if v else "No" for v in screen],
            "ERAD screen\nselected", "0 = No / 1 = Yes",
        ),
    ]

    n = len(selected)
    fig = plt.figure(figsize=(14.1, max(5.8, 0.47 * n + 2.2)))
    gs = fig.add_gridspec(
        2, len(columns), height_ratios=[1, 0.055], hspace=0.18,
        width_ratios=[1.08, 1.08, 1.48, 1.48, 1.15],
    )
    axs = [fig.add_subplot(gs[0, j]) for j in range(len(columns))]
    caxs = [fig.add_subplot(gs[1, j]) for j in range(len(columns))]
    for j, (ax, col) in enumerate(zip(axs, columns)):
        values, cmap, norm, labels, title, cbar_label = col
        arr = np.ma.masked_invalid(np.asarray(values, dtype=float).reshape(n, 1))
        cmap = cmap.copy()
        cmap.set_bad("#f3f4f6")
        im = ax.imshow(
            arr, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
        )
        ax.set_xticks([0])
        ax.set_xticklabels([title], fontsize=10, fontweight="medium")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_yticks(np.arange(n))
        if j == 0:
            names = [
                f"{r.gene}  |  {r.cadtp_track}"
                for r in selected.itertuples(index=False)
            ]
            ax.set_yticklabels(names, fontsize=10, fontweight="medium")
            ax.tick_params(axis="y", length=0, pad=11, labelleft=True)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xticks(np.arange(-.5, 1, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, label in enumerate(labels):
            z = float(values[i])
            rgba = cmap(norm(z)) if np.isfinite(z) else (0.95, 0.95, 0.95, 1)
            # Keep annotations readable on saturated heatmap cells.
            luminosity = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
            ink = "white" if luminosity < 0.46 else "#111827"
            ax.text(0, i, label, ha="center", va="center",
                    fontsize=9.2, fontweight="medium", color=ink)
        if j < 4:
            bar = fig.colorbar(
                im, cax=caxs[j], orientation="horizontal",
            )
            if j in (2, 3):
                finite_values = np.asarray(values)[np.isfinite(values)]
                cap = max(1.0, float(np.max(np.abs(finite_values))))
                bar.set_ticks([-cap, 0, cap])
                bar.set_ticklabels([f"-{cap:.1f}", "0", f"+{cap:.1f}"])
            bar.ax.tick_params(labelsize=7)
            bar.set_label(cbar_label, fontsize=8)
        else:
            caxs[j].set_axis_off()
            caxs[j].text(.5, .4, "Rule-based selection",
                         transform=caxs[j].transAxes,
                         ha="center", va="center", fontsize=8,
                         color="#475569")

    fig.suptitle("Complementary evidence for prioritised proteins",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.04,
        "Columns use independent scales (signed changes use a symmetric-log "
        "colour scale). MCL 0/6 does not imply no association.",
        ha="center", fontsize=8.5, color="#475569",
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.81,
                        bottom=0.17, wspace=0.23)
    return _save_figure(fig, output_dir, file_stem)


# ============================================================
# 5. Community sensitivity (Results Fig. 4.3)
# ============================================================

def _validate_louvain_heldout(
    integrated: pd.DataFrame,
    comparison: pd.DataFrame,
    confusion: pd.DataFrame | None = None,
    jaccard: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Audit held-out data against original training tracks and raw seed votes.

    Training seeds: 3888..3897; held-out seeds: 3898..3907.
    No labels or stability statistics are inferred from the training split.
    """
    mandatory = {"string_id", "train_track", "val_track"}
    absent = mandatory - set(comparison.columns)
    if absent:
        raise ValueError(f"Validation comparison missing {sorted(absent)}")
    if comparison["string_id"].duplicated().any():
        raise ValueError("Validation comparison contains duplicate STRING IDs.")
    if integrated["string_id"].duplicated().any():
        raise ValueError("Integrated evidence contains duplicate STRING IDs.")

    expected = integrated[["string_id", "cadtp_track"]].copy()
    combined = expected.merge(comparison, on="string_id", how="left",
                              validate="one_to_one", indicator=True)
    if (combined["_merge"] != "both").any() or len(comparison) != len(expected):
        raise ValueError("Validation must match the eligible integrated IDs exactly.")
    combined = combined.drop(columns="_merge")
    if not combined["train_track"].eq(combined["cadtp_track"]).all():
        raise ValueError("Validation train_track differs from the CADTP training track.")
    if not combined["train_track"].isin(TRACKS).all() or not combined["val_track"].isin(TRACKS).all():
        raise ValueError("Unexpected held-out track labels.")
    stable = combined["train_track"].eq(combined["val_track"])
    if "same_track" in combined.columns:
        supplied = combined["same_track"].map({True: True, False: False,
                                                "True": True, "False": False,
                                                1: True, 0: False})
        if supplied.isna().any() or not stable.eq(supplied.astype(bool)).all():
            raise ValueError("same_track is inconsistent with train/val labels.")

    derived = pd.crosstab(combined["train_track"], combined["val_track"])
    derived = derived.reindex(index=TRACKS, columns=TRACKS, fill_value=0)
    if confusion is not None:
        if "train_track" not in confusion or not set(TRACKS).issubset(confusion.columns):
            raise ValueError("Confusion CSV requires train_track, Local, Cross, Boundary.")
        if confusion["train_track"].duplicated().any():
            raise ValueError("Duplicate train_track rows in confusion CSV.")
        supplied = confusion.set_index("train_track").reindex(index=TRACKS)[TRACKS]
        if supplied.isna().any().any() or not np.array_equal(
                supplied.astype(int).to_numpy(), derived.to_numpy()):
            raise ValueError("Supplied confusion matrix contradicts the per-protein comparison.")

    top20: dict[str, float] = {}
    if not {"train_rank", "val_rank"}.issubset(combined.columns):
        raise ValueError("Comparison CSV needs train_rank/val_rank for Top-20 stability.")
    for t in TRACKS:
        tr = set(combined.loc[(combined.train_track == t) &
                              (combined.train_rank <= 20), "string_id"])
        va = set(combined.loc[(combined.val_track == t) &
                              (combined.val_rank <= 20), "string_id"])
        if len(tr) != 20 or len(va) != 20:
            raise ValueError(f"Top-20 sets for {t} do not contain exactly 20 proteins.")
        top20[t] = len(tr & va) / len(tr | va)
    if jaccard is not None:
        required = {"track", "metric", "k", "jaccard"}
        if not required.issubset(jaccard.columns):
            raise ValueError(f"Jaccard CSV missing {sorted(required-set(jaccard.columns))}")
        for t in TRACKS:
            obs = jaccard.loc[(jaccard.track == t) &
                              (jaccard.metric == "top_k") &
                              (jaccard.k == 20), "jaccard"]
            if len(obs) != 1 or not np.isclose(float(obs.iloc[0]), top20[t], atol=1e-10):
                raise ValueError(f"{t} Top-20 Jaccard differs from ranking comparison.")

    if membership is not None:
        if "string_id" in membership.columns:
            membership = membership.set_index("string_id")
        if membership.index.duplicated().any():
            raise ValueError("Duplicate protein IDs in Louvain membership matrix.")
        membership = membership.copy()
        membership.columns = [str(c) for c in membership.columns]
        train_seeds = [str(i) for i in range(3888, 3898)]
        val_seeds = [str(i) for i in range(3898, 3908)]
        needed = set(train_seeds + val_seeds)
        if not needed.issubset(membership.columns):
            raise ValueError(f"Membership matrix missing seeds: {sorted(needed-set(membership.columns))}")
        if not set(expected.string_id).issubset(membership.index):
            raise ValueError("Membership matrix misses eligible STRING IDs.")
        votes = membership.loc[combined.string_id, train_seeds + val_seeds]
        if not votes.isin([0, 1]).all().all():
            raise ValueError("Louvain seed memberships must be binary.")
        def votes_to_track(seeds):
            freq = votes[seeds].mean(axis=1).to_numpy()
            return np.select([freq >= .8, freq <= .2],
                             ["Local", "Cross"], default="Boundary")
        if not np.array_equal(votes_to_track(train_seeds), combined.train_track.to_numpy()):
            raise ValueError("Training labels contradict raw Louvain seed membership.")
        if not np.array_equal(votes_to_track(val_seeds), combined.val_track.to_numpy()):
            raise ValueError("Held-out labels contradict raw Louvain seed membership.")

    if rankings is not None:
        necessary = {"string_id", "track", "rank_within_track", "cadtp_score"}
        if not necessary.issubset(rankings.columns):
            raise ValueError(f"Validation ranking CSV missing {sorted(necessary-set(rankings.columns))}")
        if rankings.string_id.duplicated().any():
            raise ValueError("Duplicate IDs in validation rankings.")
        r = combined.merge(rankings[list(necessary)], on="string_id",
                           how="left", validate="one_to_one")
        if r["track"].isna().any() or not r.val_track.eq(r.track).all():
            raise ValueError("Validation ranking tracks contradict comparison CSV.")
        if not r.val_rank.eq(r.rank_within_track).all():
            raise ValueError("Validation ranking order contradicts comparison CSV.")
        if "val_score" in r and not np.allclose(
                r.val_score.to_numpy(), r.cadtp_score.to_numpy(), rtol=1e-8, atol=1e-10):
            raise ValueError("Validation scores contradict comparison CSV.")

    return combined, derived, top20


def plot_community_sensitivity(
    integrated: pd.DataFrame,
    mcl_results: pd.DataFrame,
    output_dir: str | Path,
    validation: pd.DataFrame | None = None,
    confusion: pd.DataFrame | None = None,
    jaccard: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    file_stem: str = "community_resolution_sensitivity",
) -> list[Path]:
    """Figure 4.3: MCL inflation sensitivity + TRUE held-out Louvain confusion.

    An explicit validation comparison CSV is required. Raw 20-seed memberships,
    the exported confusion, Jaccard and rankings are cross-checked when supplied.
    """
    if validation is None:
        raise ValueError(
            "Figure 4.3 requires cadtp_validation_comparison.csv; "
            "do not substitute MCL agreement for held-out Louvain validation."
        )
    if "gene_name" not in mcl_results.columns:
        raise ValueError("MCL results must contain gene_name.")
    hlj1 = mcl_results.loc[mcl_results["gene_name"].eq("HLJ1")]
    if len(hlj1) != 1:
        raise ValueError("Expected exactly one HLJ1 row in MCL results.")
    hlj1 = hlj1.iloc[0]
    import re
    sizes = []
    for column in mcl_results.columns:
        found = re.fullmatch(r"cluster_size_r(\d+)_(\d+)", column)
        if found:
            r = float(f"{found.group(1)}.{found.group(2)}")
            val = pd.to_numeric(hlj1[column], errors="coerce")
            if pd.notna(val):
                sizes.append((r, float(val)))
    if len(sizes) < 2:
        raise ValueError("MCL results must supply at least two inflation settings.")
    sizes.sort()
    matched, cm, top20 = _validate_louvain_heldout(
        integrated, validation, confusion=confusion, jaccard=jaccard,
        membership=membership, rankings=rankings,
    )
    overall = matched.train_track.eq(matched.val_track).mean()
    count = len(matched)
    stable = int(round(overall * count))

    fig, (ax_mcl, ax_cm) = plt.subplots(
        1, 2, figsize=(12.0, 5.6),
        gridspec_kw={"width_ratios": [1.05, 1]},
    )
    x, y = zip(*sizes)
    ax_mcl.plot(x, y, marker="o", markersize=7, linewidth=2.2,
                color="#2563eb", zorder=3)
    for r, size in sizes:
        ax_mcl.annotate(str(int(size)), (r, size), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)
    ax_mcl.set_xticks(x)
    ax_mcl.set_xlabel("MCL inflation parameter")
    ax_mcl.set_ylabel("HLJ1-containing cluster size (proteins)")
    ax_mcl.set_title("A. MCL resolution sensitivity", fontsize=11, fontweight="bold")
    ax_mcl.set_ylim(0, max(y)*1.24)
    ax_mcl.grid(axis="y", linestyle=":", alpha=.3)
    ax_mcl.spines[["top", "right"]].set_visible(False)
    ax_mcl.set_axisbelow(True)

    values = cm.to_numpy(dtype=int)
    denom = values.sum(axis=1, keepdims=True)
    row_fractions = values / denom
    im = ax_cm.imshow(row_fractions, cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
    ax_cm.set_xticks(range(3), [f"{t}\n(n={int(values[:, i].sum()):,})"
                                for i, t in enumerate(TRACKS)], fontsize=9)
    ax_cm.set_yticks(range(3), [f"{t} (n={int(denom[i, 0]):,})"
                                for i, t in enumerate(TRACKS)], fontsize=9)
    ax_cm.set_xlabel("Held-out Louvain track (seeds 3898–3907)", fontsize=10)
    ax_cm.set_ylabel("Training track (seeds 3888–3897)", fontsize=10)
    ax_cm.set_title(f"B. Held-out track agreement: {overall:.2%}",
                    fontsize=11, fontweight="bold")
    ax_cm.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax_cm.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax_cm.grid(which="minor", color="white", linewidth=2)
    ax_cm.tick_params(which="minor", length=0)
    ax_cm.tick_params(which="major", length=0)
    for i in range(3):
        for j in range(3):
            n = values[i, j]
            pct = row_fractions[i, j]
            ax_cm.text(j, i, f"{n:,}\n({pct:.1%})",
                       ha="center", va="center", fontsize=10,
                       fontweight="semibold" if i == j else "normal",
                       color="white" if pct > .55 else "#172554")
    ax_cm.text(.5, -0.29,
               "Top-20 Jaccard   " + "   |   ".join(
                   f"{t}: {top20[t]:.3f}" for t in TRACKS),
               ha="center", va="top", fontsize=9,
               color="#334155", transform=ax_cm.transAxes)
    fig.suptitle("Community robustness: resolution and held-out seeds",
                 fontsize=13, fontweight="bold", y=.98)
    fig.text(.5, .017,
             f"Held-out validation covers {count:,} eligible proteins; "
             f"{stable:,} retain the same CADTP track. "
             "MCL settings are not experimental replicates.",
             ha="center", fontsize=8.5, color="#475569")
    fig.subplots_adjust(left=.075, right=.98, wspace=.28, bottom=.27, top=.86)
    return _save_figure(fig, output_dir, file_stem)


def plot_heldout_membership_heatmap(
    membership: pd.DataFrame,
    integrated: pd.DataFrame,
    output_dir: str | Path,
    genes: Sequence[str] | None = None,
    file_stem: str = "louvain_heldout_membership_heatmap",
) -> list[Path]:
    """Optional supplement: per-seed membership for prioritised proteins.

    Uses 10 training + 10 held-out Louvain community assignments, not MCL.
    """
    selected = _selected_evidence(integrated, genes)
    matrix = membership.copy()
    if "string_id" in matrix.columns:
        matrix = matrix.set_index("string_id")
    matrix.columns = [int(c) for c in matrix.columns]
    train_seeds = list(range(3888, 3898))
    val_seeds = list(range(3898, 3908))
    paths, _ = plot_louvain_membership_heatmap(
        matrix, selected.string_id.tolist(),
        dict(zip(selected.string_id, selected.gene)),
        dict(zip(selected.string_id, selected.cadtp_track)),
        train_seeds, val_seeds, output_dir, file_stem=file_stem,
    )
    return paths


# ============================================================
# 6. Supplementary: exact node-deletion centrality comparison
# ============================================================

def plot_deletion_betweenness_scatter(
    deletion_results: pd.DataFrame,
    output_dir: str | Path,
    selected_genes: Sequence[str] = ("APJ1", "JJJ2", "SSE1", "UFD2"),
    file_stem: str = "deletion_betweenness_before_after",
) -> list[Path]:
    """
    Before/after NORMALISED exact betweenness (700-threshold graph).
    Log-like axes retain zero-valued centralities; the y=x reference
    does not isolate rerouting because normalisation denominators differ.
    """
    required = {
        "deletion_betweenness_before", "deletion_betweenness_after", "gene"
    }
    if not required.issubset(deletion_results.columns):
        raise ValueError(f"Deletion results missing {sorted(required-set(deletion_results.columns))}")
    before = pd.to_numeric(
        deletion_results["deletion_betweenness_before"], errors="coerce"
    )
    after = pd.to_numeric(
        deletion_results["deletion_betweenness_after"], errors="coerce"
    )
    good = np.isfinite(before) & np.isfinite(after) & (before >= 0) & (after >= 0)
    if not good.any():
        raise ValueError("No finite nonnegative betweenness observations.")

    scale = 1e6
    x = before[good].to_numpy() * scale
    y = after[good].to_numpy() * scale
    bound = max(np.max(x), np.max(y)) * 1.15
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.scatter(x, y, s=7, color="#94a3b8", alpha=.26,
               linewidths=0, rasterized=True)
    ax.plot([0, bound], [0, bound], linestyle="--", color="#334155",
            linewidth=1.4, label="Equal normalised betweenness")
    data = deletion_results.loc[good].copy()
    for gene in selected_genes:
        row = data.loc[data["gene"].eq(gene)]
        if len(row) != 1:
            continue
        bx = float(row.iloc[0]["deletion_betweenness_before"]) * scale
        ay = float(row.iloc[0]["deletion_betweenness_after"]) * scale
        ax.scatter([bx], [ay], color="#dc2626", s=48, zorder=5)
        ax.annotate(gene, (bx, ay), xytext=(6, 5), textcoords="offset points",
                    fontsize=9, color="#991b1b")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_xlabel(r"Before HLJ1 deletion, $10^6\times B(v)$")
    ax.set_ylabel(r"After HLJ1 deletion, $10^6\times B^-(v)$")
    ax.set_title("Exact normalised betweenness after HLJ1 node deletion",
                 fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16)
    fig.text(
        .5, -.01,
        "Shortest paths are unweighted; normalisation and endpoint sets "
        "change after node deletion.",
        ha="center", fontsize=8.5, color="#475569",
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, file_stem)


# ============================================================
# Command line: from project root, python -m src.plotting
# ============================================================

def main(argv: Sequence[str] | None = None) -> list[Path]:
    """Generate Section 4 figures with paths relative to the repo root."""
    import argparse

    parser = argparse.ArgumentParser(description="Publication figures for MATH3888 HLJ1 Results and Validation.")
    parser.add_argument("--integrated", type=Path,
                        default=Path("outputs/integrated/cross_method_evidence_700.csv"))
    parser.add_argument("--mcl", type=Path, default=Path("outputs/junxi/mcl_results.csv"))
    parser.add_argument("--go", type=Path, default=None,
                        help="GO baseline comparison CSV; auto-detects standard paths if omitted.")
    parser.add_argument("--deletion", type=Path,
                        default=Path("outputs/integrated/deletion_betweenness_700.csv"))
    parser.add_argument("--validation", type=Path,
                        default=Path("outputs/evaluation/cadtp_validation_comparison.csv"),
                        help="Per-protein training and held-out Louvain track comparison.")
    parser.add_argument("--confusion", type=Path,
                        default=Path("outputs/evaluation/cadtp_validation_confusion.csv"))
    parser.add_argument("--jaccard", type=Path,
                        default=Path("outputs/evaluation/cadtp_validation_jaccard.csv"))
    parser.add_argument("--membership", type=Path,
                        default=Path("outputs/evaluation/louvain_membership_matrix_700.csv.gz"))
    parser.add_argument("--validation-rankings", type=Path,
                        default=Path("outputs/evaluation/cadtp_validation_rankings.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--genes", nargs="+", default=None)
    parser.add_argument("--only", nargs="+",
                        choices=("heatmap", "go", "community", "membership", "deletion"),
                        default=["heatmap", "go", "community"],
                        help="Figures to generate; membership and deletion are supplementary.")
    args = parser.parse_args(argv)

    go_path = args.go
    if go_path is None:
        possibilities = [
            Path("outputs/evaluation/go_baseline_comparison_refactor.csv"),
            Path("outputs/evaluation/go_baseline_comparison.csv"),
            Path("outputs/go_baseline_comparison.csv"),
            Path("data/processed/go_baseline_comparison.csv"),
            Path("go_baseline_comparison.csv"),
        ]
        go_path = next((p for p in possibilities if p.is_file()), possibilities[0])

    def read(path: Path, purpose: str) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(
                f"{purpose}: {path}\n"
                "Supply the actual local dataset path through the corresponding "
                "command-line option. Run from the repository root."
            )
        return pd.read_csv(path)

    produced: list[Path] = []
    integrated = None
    if any(k in args.only for k in ("heatmap", "community", "membership")):
        integrated = read(args.integrated, "Integrated candidate evidence")
    if "heatmap" in args.only:
        produced.extend(plot_cross_method_evidence_heatmap(
            integrated, args.output_dir, genes=args.genes
        ))
    if "go" in args.only:
        produced.extend(plot_go_precision(
            read(go_path, "GO baseline comparison"), args.output_dir, k=args.k
        ))
    if "community" in args.only:
        produced.extend(plot_community_sensitivity(
            integrated, read(args.mcl, "MCL results"), args.output_dir,
            validation=read(args.validation, "Held-out track comparison"),
            confusion=read(args.confusion, "Held-out confusion matrix"),
            jaccard=read(args.jaccard, "Held-out ranking Jaccard"),
            membership=read(args.membership, "Raw 20-seed Louvain memberships"),
            rankings=read(args.validation_rankings, "Held-out candidate rankings"),
        ))
    if "membership" in args.only:
        produced.extend(plot_heldout_membership_heatmap(
            read(args.membership, "Raw 20-seed Louvain memberships"),
            integrated, args.output_dir, genes=args.genes,
        ))
    if "deletion" in args.only:
        produced.extend(plot_deletion_betweenness_scatter(
            read(args.deletion, "Exact deletion results"), args.output_dir
        ))
    for path in produced:
        print(f"Saved: {path}")
    return produced


if __name__ == "__main__":
    main()
