"""
Reproducible Experiment Pipeline
Stages:
    network  : Construct and validate PPI networks.
    cadtp    : Fit CADTP training ensemble.
    validate : Run held-out Louvain validation.
    all      : Run CADTP training and validation.
Usage:
    python -m src.pipeline --stage all
"""
import argparse
import networkx as nx
import numpy as np
import pandas as pd

from .config import ExperimentConfig, load_config
from .network import build_networks
from .algorithms import rank_percentile, community_membership_matrix
from .cadtp import CADTPRanker
from .evaluation import (
    compare_go_baselines,
    run_matched_go_comparison,
)
from .plotting import (
    plot_go_precision,
    plot_candidate_network,
    plot_louvain_membership_heatmap,
)


TRACKS = ["Local", "Cross", "Boundary"]

TRAIN_COUNTS = {
    "Local": 120,
    "Cross": 3985,
    "Boundary": 334,
}

VALIDATION_COUNTS = {
    "Local": 97,
    "Cross": 3807,
    "Boundary": 535,
}



# ============================================================
# Save Louvain membership for reproducible visualisation
# ============================================================

def save_louvain_membership(
    cfg: ExperimentConfig,
    graph: nx.Graph,
    communities: dict[int, set[str]],
    group: str,
) -> None:
    """
    Persist the exact Louvain ensemble used by CADTP.

    group:
        "training" or "validation"
    """

    if group not in ("training", "validation"):
        raise ValueError(f"Unknown group: {group}")

    target = cfg.project["target_protein"]

    nodes = [target] + sorted(
        node for node in graph.nodes()
        if node != target
    )

    membership = community_membership_matrix(
        communities,
        nodes=nodes,
    )

    threshold = cfg.network["confidence_threshold"]

    output_path = (
        cfg.path("evaluation")
        / f"louvain_membership_{group}_{threshold}.csv.gz"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    membership.to_csv(
        output_path,
        compression="gzip",
    )

    print(f"Louvain membership cached: {output_path}")


# ============================================================
# 1. Network construction
# ============================================================

def prepare_networks(cfg: ExperimentConfig):

    cfg.check_inputs(
        "functional_edges",
        "physical_edges",
        "protein_mapping",
    )
    G_full, G_hlj1, G_physical = build_networks(cfg)
    target = cfg.project["target_protein"]

    print("\n=== Network Summary ===")
    print(
        f"Full functional: "
        f"{G_full.number_of_nodes():,} nodes, "
        f"{G_full.number_of_edges():,} edges"
    )
    print(
        f"HLJ1 component: "
        f"{G_hlj1.number_of_nodes():,} nodes, "
        f"{G_hlj1.number_of_edges():,} edges"
    )
    print(
        f"Physical layer: "
        f"{G_physical.number_of_nodes():,} nodes, "
        f"{G_physical.number_of_edges():,} edges"
    )

    assert G_full.number_of_nodes() == 6600
    assert G_full.number_of_edges() == 104188

    assert G_hlj1.number_of_nodes() == 5716
    assert G_hlj1.number_of_edges() == 104138

    assert G_physical.number_of_edges() == 43028

    assert G_hlj1.degree(target) == 25
    assert G_physical.degree(target) == 0

    print("Network regression checks passed!")

    return G_full, G_hlj1, G_physical


# ============================================================
# 2. Load experiment inputs
# ============================================================

def load_cadtp_inputs(cfg: ExperimentConfig):

    cfg.check_inputs(
        "node2vec_consensus",
        "baseline_features",
        "essential_proteins",
    )

    # Node2Vec consensus from Notebook 04
    n2v = pd.read_csv(
        cfg.path("node2vec_consensus"),
        index_col="string_id",
    )

    if "mean_rank" not in n2v.columns:
        raise ValueError("Node2Vec CSV must contain mean_rank.")

    # Smaller mean rank => higher N
    n2v["N"] = rank_percentile(-n2v["mean_rank"])

    # Protein baseline from Notebook 02
    baseline = pd.read_csv(
        cfg.path("baseline_features"),
        index_col="string_protein_id",
    )

    # Essential protein annotations
    essential_df = pd.read_csv(
        cfg.path("essential_proteins"),
        header=None,
        dtype=str,
    )

    if essential_df.shape[1] < 2:
        raise ValueError(
            "Essential protein CSV requires at least two columns."
        )

    essential_orfs = set(
        essential_df.iloc[:, 1]
        .dropna()
        .str.strip()
    )

    baseline["is_listed_essential"] = baseline["orf"].isin(essential_orfs)

    # Basic validation
    assert n2v.index.is_unique
    assert baseline.index.is_unique

    assert len(n2v) == 5715
    assert n2v.index.isin(baseline.index).all()

    assert np.isfinite(n2v["N"]).all()
    assert n2v["N"].between(0, 1).all()

    print("\n=== Experiment Inputs ===")
    print(f"Node2Vec proteins: {len(n2v):,}")
    print(f"Baseline proteins: {len(baseline):,}")
    print(f"Essential ORFs: {len(essential_orfs):,}")

    return n2v, baseline


# ============================================================
# 3. CADTP training
# ============================================================

def run_cadtp(
    cfg: ExperimentConfig,
    G_full: nx.Graph,
    G_hlj1: nx.Graph,
    n2v: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    print("\n=== CADTP Training ===")
    ranker = CADTPRanker(cfg)
    rankings = ranker.fit(
        G_full=G_full,
        G_hlj1=G_hlj1,
        node2vec=n2v,
        baseline=baseline,
        seeds=cfg.louvain["training_seeds"],
    )

    save_louvain_membership(
        cfg,
        G_hlj1,
        ranker.communities,
        group="training",
    )

    counts = (
        rankings["track"]
        .value_counts()
        .reindex(TRACKS, fill_value=0)
        .to_dict()
    )

    print(f"Eligible candidates: {len(rankings):,}")
    print("Track counts:", counts)

    assert len(rankings) == 4439
    assert counts == TRAIN_COUNTS

    # Display Top-10 from each track
    for track in TRACKS:
        print(f"\n--- Top-10 {track} ---")
        columns = [
            "preferred_name",
            "cadtp_score",
            "rank_within_track",
        ]

        print(
            ranker.top_candidates(track, 10)[columns]
            .round(6)
            .to_string()
        )

    # Save training outputs
    output_dir = cfg.path("rankings")
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(output_dir / "cadtp_training_refactor.csv")
    ranker.all_scores.to_csv(output_dir / "cadtp_all_scores_refactor.csv")

    print("\nCADTP training completed.")

    return rankings


# ============================================================
# 4. Jaccard helpers
# ============================================================

def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def top_k_ids(
    rankings: pd.DataFrame,
    track: str,
    k: int,
) -> set:
    subset = rankings.loc[rankings["track"].eq(track)]
    return set(subset.nsmallest(k, "rank_within_track").index)


# ============================================================
# 5. Held-out validation
# ============================================================

def run_validation(
    cfg: ExperimentConfig,
    G_full: nx.Graph,
    G_hlj1: nx.Graph,
    n2v: pd.DataFrame,
    baseline: pd.DataFrame,
    train: pd.DataFrame | None = None,
):

    print("\n=== Held-out Validation ===")

    # Reuse training rankings when running --stage all.
    # Otherwise load existing training results.
    if train is None:
        train_path = cfg.path("rankings") / "cadtp_training_refactor.csv"
        if not train_path.is_file():
            raise FileNotFoundError(
                "Training rankings not found. "
                "Run --stage cadtp first."
            )

        train = pd.read_csv(train_path, index_col="string_id")

    # Run held-out Louvain ensemble
    ranker = CADTPRanker(cfg)

    valid = ranker.fit(
        G_full=G_full,
        G_hlj1=G_hlj1,
        node2vec=n2v,
        baseline=baseline,
        seeds=cfg.louvain["validation_seeds"],
    )

    save_louvain_membership(
        cfg,
        G_hlj1,
        ranker.communities,
        group="validation",
    )

    assert len(train) == len(valid) == 4439
    assert set(train.index) == set(valid.index)

    counts = (
        valid["track"]
        .value_counts()
        .reindex(TRACKS, fill_value=0)
        .to_dict()
    )

    print("Validation track counts:", counts)
    assert counts == VALIDATION_COUNTS

    # --------------------------------------------------------
    # A. Classification agreement
    # --------------------------------------------------------

    valid_aligned = valid.reindex(train.index)

    comparison = pd.DataFrame(
        {
            "preferred_name": train["preferred_name"],
            "train_track": train["track"],
            "val_track": valid_aligned["track"],
            "train_rank": train["rank_within_track"],
            "val_rank": valid_aligned["rank_within_track"],
            "train_score": train["cadtp_score"],
            "val_score": valid_aligned["cadtp_score"],
        },
        index=train.index,
    )

    comparison["same_track"] = comparison["train_track"].eq(comparison["val_track"])
    agreement_count = int(comparison["same_track"].sum())
    agreement = agreement_count / len(comparison)

    print(
        f"\nClassification agreement: "
        f"{agreement_count:,}/{len(comparison):,} "
        f"({agreement:.4%})"
    )

    assert agreement_count == 4166

    # --------------------------------------------------------
    # B. Transition matrix
    # --------------------------------------------------------

    confusion = pd.crosstab(
        comparison["train_track"],
        comparison["val_track"],
    ).reindex(
        index=TRACKS,
        columns=TRACKS,
        fill_value=0,
    )

    expected_matrix = np.array([
        [95,    0,  25],
        [0,  3773, 212],
        [2,    34, 298],
    ])

    np.testing.assert_array_equal(
        confusion.to_numpy(),
        expected_matrix,
    )

    print("\nTrack transition matrix:")
    print(confusion.to_string())

    # --------------------------------------------------------
    # C. Pool and Top-k Jaccard
    # --------------------------------------------------------

    records = []

    for track in TRACKS:

        train_pool = set(
            train.index[train["track"].eq(track)]
        )

        valid_pool = set(
            valid.index[valid["track"].eq(track)]
        )

        records.append({
            "track": track,
            "metric": "pool",
            "k": None,
            "train_size": len(train_pool),
            "val_size": len(valid_pool),
            "overlap": len(train_pool & valid_pool),
            "jaccard": jaccard(train_pool, valid_pool),
        })

        for k in [10, 20, 50]:

            a = top_k_ids(train, track, k)
            b = top_k_ids(valid, track, k)

            records.append({
                "track": track,
                "metric": "top_k",
                "k": k,
                "train_size": len(a),
                "val_size": len(b),
                "overlap": len(a & b),
                "jaccard": jaccard(a, b),
            })

    stability = pd.DataFrame(records)

    print("\nJaccard stability:")
    print(
        stability[
            ["track", "metric", "k", "overlap", "jaccard"]
        ].round(6).to_string(index=False)
    )

    # Regression checks for Top-20
    expected_top20 = {
        "Local": 19,
        "Cross": 20,
        "Boundary": 15,
    }

    for track, expected in expected_top20.items():

        a = top_k_ids(train, track, 20)
        b = top_k_ids(valid, track, 20)

        assert len(a & b) == expected

    # --------------------------------------------------------
    # D. Save outputs
    # --------------------------------------------------------

    output_dir = cfg.path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    valid.to_csv(output_dir / "cadtp_validation_rankings.csv")
    ranker.all_scores.to_csv(output_dir / "cadtp_validation_all_scores.csv")
    comparison.to_csv(output_dir / "cadtp_validation_comparison.csv")
    confusion.to_csv(output_dir / "cadtp_validation_confusion.csv")
    stability.to_csv(output_dir / "cadtp_validation_jaccard.csv")

    print("\nHeld-out validation completed.")

    return valid, stability



# ============================================================
# 6. GO biological evaluation
# ============================================================

def run_evaluation(cfg: ExperimentConfig):
    """
    Run GO-based evaluation using saved CADTP rankings.

    Includes:
        - GO annotation coverage
        - Precision@k baseline comparison
        - Track-background fold enrichment
        - Degree/distance-matched randomisation
        - BH multiple-testing correction

    Does NOT retrain models or rerun Louvain.
    """
    print("\n=== GO Biological Evaluation ===")
    output_dir = cfg.path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # A. Load saved CADTP training rankings
    # --------------------------------------------------------

    ranking_path = cfg.path("rankings") / "cadtp_training_refactor.csv"
    if not ranking_path.is_file():
        raise FileNotFoundError(
            "CADTP rankings not found. "
            "Run --stage cadtp first."
        )

    rankings = pd.read_csv(ranking_path, index_col="string_id",)

    assert rankings.index.is_unique
    assert len(rankings) == 4439

    # --------------------------------------------------------
    # B. Load saved GO labels
    # --------------------------------------------------------

    go_path = output_dir / "go_biological_process_labels_refactor.csv"

    if not go_path.is_file():
        raise FileNotFoundError(
            "GO labels not found. "
            "Complete Step 8.3 first."
        )

    labels = pd.read_csv(go_path, index_col="orf",)

    assert labels.index.is_unique

    required_labels = [
        "has_BP_annotation",
        "proteostasis_any",
    ]

    for column in required_labels:
        if column not in labels.columns:
            raise ValueError(f"Missing GO label column: {column}")

    # --------------------------------------------------------
    # C. Join GO labels to STRING protein IDs via ORF
    # --------------------------------------------------------

    evaluation = rankings.copy()

    if "orf" not in evaluation.columns:
        raise ValueError("CADTP rankings must contain ORF identifiers.")

    protein_orfs = (
        evaluation["orf"]
        .astype("string")
        .str.strip()
    )

    missing = ~protein_orfs.isin(labels.index)

    if missing.any():
        raise ValueError(
            f"{missing.sum()} candidate ORFs "
            "are missing from the GO label table."
        )

    mapped_labels = labels.reindex(protein_orfs.tolist())
    mapped_labels.index = evaluation.index

    # Explicit boolean parsing, avoiding string truthiness
    for column in required_labels:
        values = (
            mapped_labels[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        parsed = values.map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })

        if parsed.isna().any():
            raise ValueError(f"Invalid GO annotation values in {column}")

        evaluation[column] = parsed.astype(bool)

    print(f"Eligible proteins: {len(evaluation):,}")

    # --------------------------------------------------------
    # D. Annotation coverage by track
    # --------------------------------------------------------

    coverage = (
        evaluation.groupby("track")
        .agg(
            pool_size=("track", "size"),
            bp_annotated=("has_BP_annotation", "sum"),
            go_positive=("proteostasis_any", "sum"),
        )
    )

    coverage["annotation_coverage"] = coverage["bp_annotated"] / coverage["pool_size"]
    coverage["positive_rate"] = coverage["go_positive"] / coverage["pool_size"]
    coverage.to_csv(output_dir / "go_annotation_coverage_refactor.csv")

    print("\n=== GO Annotation Coverage ===")
    print(coverage.round(6).to_string())

    # --------------------------------------------------------
    # E. Baseline comparison
    # --------------------------------------------------------

    positive_ids = set(evaluation.index[evaluation["proteostasis_any"]])

    baselines = compare_go_baselines(
        rankings=evaluation,
        positive_ids=positive_ids,
        k_values=[10, 20, 50],
        tracks=["Local", "Cross", "Boundary"],
    )

    baselines.to_csv(
        output_dir / "go_baseline_comparison_refactor.csv",
        index=False,
    )

    print("\n=== GO Precision@20 ===")

    p20 = baselines.loc[
        baselines["k"].eq(20),
        [
            "track",
            "method",
            "observed_positive",
            "precision",
            "fold_enrichment",
        ],
    ]

    print(p20.round(6).to_string(index=False))

    # --------------------------------------------------------
    # F. Prepare original Notebook 06 evaluation schema
    # --------------------------------------------------------

    track_mapping = {
        "Local": "A_local_consensus",
        "Cross": "B_cross_consensus",
        "Boundary": "C_boundary_consensus",
    }

    evaluation["consensus_track"] = evaluation["track"].map(track_mapping)
    evaluation["hybrid_score"] = evaluation["cadtp_score"]

    if evaluation["consensus_track"].isna().any():
        raise ValueError("Unknown CADTP track.")

    # --------------------------------------------------------
    # G. Matched randomisation + BH correction
    # --------------------------------------------------------

    matched, adjusted = run_matched_go_comparison(
        evaluation=evaluation,
        n_permutations=cfg.evaluation["matched_randomisation"]["n_permutations"],
        seed=cfg.evaluation["matched_randomisation"]["random_seed"],
    )

    matched.to_csv(
        output_dir / "go_matched_randomisation_refactor.csv",
        index=False,
    )

    adjusted.to_csv(
        output_dir / "go_matched_tests_adjusted_refactor.csv",
        index=False,
    )

    print("\n=== Matched Randomisation Results ===")

    print(adjusted[[
        "track",
        "method",
        "observed_hits",
        "null_mean_hits",
        "matched_fold_enrichment",
        "empirical_p_value",
        "q_value_BH",
        "significant_FDR_05",
    ]].round(6).to_string(index=False))

    assert len(matched) == 8
    assert len(adjusted) == 8

    print("\nGO evaluation completed successfully!")

    return baselines, adjusted



# ============================================================
# Publication-ready figures
# ============================================================

def run_plotting(
    cfg: ExperimentConfig,
    G_hlj1: nx.Graph | None = None,
) -> None:
    """
    Generate all core MATH3888 figures.

    Outputs:
        - GO Precision@20
        - HLJ1 Local candidate network
        - HLJ1 Cross candidate network
        - Louvain community membership heatmap

    Uses cached model outputs and Louvain partitions.
    Does not retrain Node2Vec or rerun Louvain.
    """

    print("\n=== Generating Research Figures ===")

    figure_dir = cfg.path("figures")
    evaluation_dir = cfg.path("evaluation")

    figure_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # A. Load rankings and protein names
    # --------------------------------------------------------

    rankings = pd.read_csv(
        cfg.path("rankings")
        / "cadtp_training_refactor.csv",
        index_col="string_id",
    )

    baseline = pd.read_csv(
        cfg.path("baseline_features"),
        index_col="string_protein_id",
    )

    target = cfg.project["target_protein"]

    gene_names = (
        baseline["preferred_name"]
        .dropna()
        .to_dict()
    )
    gene_names[target] = "HLJ1"

    track_map = rankings["track"].to_dict()
    track_map[target] = "Anchor"

    def protein_id(gene: str) -> str:
        """Resolve a gene name from actual ranking metadata."""

        if gene == "HLJ1":
            return target

        matches = rankings.index[
            rankings["preferred_name"].eq(gene)
        ].tolist()

        if len(matches) != 1:
            raise ValueError(
                f"Expected one eligible match for {gene}, "
                f"found {len(matches)}."
            )

        return matches[0]

    # --------------------------------------------------------
    # B. GO baseline comparison figure
    # --------------------------------------------------------

    go_comparison = pd.read_csv(
        evaluation_dir
        / "go_baseline_comparison_refactor.csv"
    )

    paths = plot_go_precision(
        results=go_comparison,
        output_dir=figure_dir,
        k=20,
    )

    print("\nGO Precision@20:")
    for path in paths:
        print(" ", path.name)

    # --------------------------------------------------------
    # C. Network figures
    # --------------------------------------------------------

    if G_hlj1 is None:
        _, G_hlj1, _ = build_networks(cfg)

    networks = {
        "Local": {
            "genes": ["SSE1", "APJ1", "JJJ2"],
            "stem": "hlj1_local_network",
            "title": "HLJ1 Local Candidate Network",
        },
        "Cross": {
            "genes": ["SSM4", "UBX2", "YOS9", "USA1"],
            "stem": "hlj1_cross_network",
            "title": "HLJ1 Cross-community Candidate Network",
        },
    }

    for track, settings in networks.items():

        candidate_ids = [
            protein_id(gene)
            for gene in settings["genes"]
        ]

        paths, evidence = plot_candidate_network(
            graph=G_hlj1,
            target_id=target,
            candidate_ids=candidate_ids,
            protein_names=gene_names,
            output_dir=figure_dir,
            file_stem=settings["stem"],
            title=settings["title"],
            candidate_type=track,
            seed=cfg.visualisation["network"]["seed"],
        )

        evidence.to_csv(
            evaluation_dir
            / f"hlj1_{track.lower()}_path_evidence.csv",
            index=False,
        )

        print(f"\n{track} Network:")
        for path in paths:
            print(" ", path.name)

    # --------------------------------------------------------
    # D. Load cached Louvain membership
    # --------------------------------------------------------

    threshold = cfg.network["confidence_threshold"]

    train_seeds = cfg.louvain["training_seeds"]
    val_seeds = cfg.louvain["validation_seeds"]

    train_cache = (
        evaluation_dir
        / f"louvain_membership_training_{threshold}.csv.gz"
    )

    val_cache = (
        evaluation_dir
        / f"louvain_membership_validation_{threshold}.csv.gz"
    )

    # Compatibility with the earlier Step 9.3 cache
    combined_cache = (
        evaluation_dir
        / f"louvain_membership_matrix_{threshold}.csv.gz"
    )

    def read_membership(path):
        matrix = pd.read_csv(
            path,
            index_col="string_id",
        )
        matrix.columns = matrix.columns.astype(int)
        return matrix

    if train_cache.is_file() and val_cache.is_file():

        training = read_membership(train_cache)
        validation = read_membership(val_cache)

        if set(training.index) != set(validation.index):
            raise ValueError(
                "Training and validation membership "
                "contain different protein sets."
            )

        validation = validation.reindex(training.index)

        membership = pd.concat(
            [training, validation],
            axis=1,
        )

    elif combined_cache.is_file():

        membership = read_membership(combined_cache)

    else:

        raise FileNotFoundError(
            "Louvain membership cache not found. "
            "Run --stage all, or generate the "
            "membership matrix using Step 9.3."
        )

    expected_seeds = set(train_seeds + val_seeds)

    if set(membership.columns) != expected_seeds:
        raise ValueError(
            "Cached Louvain seeds do not match config."
        )

    # Verify the cached partitions against CADTP scores
    for group, seeds, filename in (
        (
            "training",
            train_seeds,
            "cadtp_all_scores_refactor.csv",
        ),
        (
            "validation",
            val_seeds,
            "cadtp_validation_all_scores.csv",
        ),
    ):

        reference = pd.read_csv(
            (
                cfg.path("rankings")
                if group == "training"
                else evaluation_dir
            ) / filename,
            index_col="string_id",
        )

        actual_votes = membership.loc[
            reference.index, seeds
        ].sum(axis=1)

        if not np.array_equal(
            actual_votes.to_numpy(),
            reference["membership_count"].to_numpy(),
        ):
            raise ValueError(
                f"{group} membership cache does not "
                "match saved CADTP scores."
            )

    print("\nLouvain membership cache validated.")

    # --------------------------------------------------------
    # E. Louvain heatmap
    # --------------------------------------------------------

    selected_genes = [
        "HLJ1",
        "SSE1", "APJ1", "JJJ2",
        "SSM4", "UBX2", "YOS9", "USA1",
        "TIM21", "PAM17", "MGR2", "TOM70",
    ]

    selected_ids = [
        protein_id(gene)
        for gene in selected_genes
    ]

    paths, summary = plot_louvain_membership_heatmap(
        membership=membership,
        protein_ids=selected_ids,
        protein_names=gene_names,
        tracks=track_map,
        training_seeds=train_seeds,
        validation_seeds=val_seeds,
        output_dir=figure_dir,
    )

    summary.to_csv(
        evaluation_dir
        / "louvain_membership_selected.csv",
        index=False,
    )

    print("\nLouvain Membership Heatmap:")
    for path in paths:
        print(" ", path.name)

    print("\nAll research figures generated successfully!")


# ============================================================
# 7. Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MATH3888 HLJ1 Network Analysis Pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "network",
            "cadtp",
            "validate",
            "evaluate",
            "plot",
            "all"
        ],
        default="all",
        help="Pipeline stage to execute."
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to experiment configuration.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_output_dirs()

    print("=" * 55)
    print("MATH3888 — HLJ1 Protein Network Analysis")
    print(f"Stage: {args.stage}")
    print("=" * 55)

    # GO evaluation reads saved rankings and labels.
    # No graph construction or Louvain reruns are needed.
    if args.stage == "evaluate":
        run_evaluation(cfg)
        print("\nPipeline completed successfully!")
        return

    if args.stage == "plot":
        run_plotting(cfg)
        print("\nPipeline completed successfully!")
        return

    # Construct graphs only once per invocation
    G_full, G_hlj1, _ = prepare_networks(cfg)

    if args.stage == "network":
        print("\nNetwork stage completed.")
        return

    # Shared experiment inputs
    n2v, baseline = load_cadtp_inputs(cfg)
    train = None

    if args.stage in ("cadtp", "all"):
        train = run_cadtp(
            cfg,
            G_full,
            G_hlj1,
            n2v,
            baseline,
        )

    if args.stage in ("validate", "all"):
        run_validation(
            cfg,
            G_full,
            G_hlj1,
            n2v,
            baseline,
            train=train,
        )

    if args.stage == "all":
        run_evaluation(cfg)
        run_plotting(cfg, G_hlj1=G_hlj1)

    print("\n" + "=" * 55)
    print("Pipeline completed successfully!")
    print("=" * 55)


if __name__ == "__main__":
    main()