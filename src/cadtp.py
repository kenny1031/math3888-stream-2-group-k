"""
Consensus-Aware Dual-Track Prioritisation (CADTP).
Combines:
    - Node2Vec consensus rankings
    - Personalised PageRank
    - Louvain consensus membership
    - Smoothed community affinity
Ranking is performed independently within each track.
"""
import math
from fractions import Fraction
import networkx as nx
import numpy as np
import pandas as pd

from .algorithms import (
    compute_rwr,
    run_louvain_ensemble,
    community_membership_counts,
    mean_community_affinity,
    rank_percentile,
)
from .config import ExperimentConfig


# ============================================================
# 1. Helpers
# ============================================================

def _as_bool(series: pd.Series) -> pd.Series:
    """Convert CSV boolean columns without string truthiness bugs."""

    values = series.astype("string").str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    parsed = values.map(mapping)

    if parsed.isna().any():
        raise ValueError(
            f"Invalid or missing boolean values in {series.name}"
        )

    return parsed.astype(bool)


def _weighted_average(data: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Calculate a weighted average of CADTP components."""

    columns = {
        "node2vec": "N",
        "rwr": "R",
        "community": "C",
    }

    result = pd.Series(
        0.0,
        index=data.index,
        dtype=float,
    )

    total = 0.0

    for name, weight in weights.items():

        if name not in columns:
            raise ValueError(f"Unknown scoring component: {name}")

        if weight < 0:
            raise ValueError("Scoring weights cannot be negative.")

        result += weight * data[columns[name]]
        total += weight

    if total <= 0:
        raise ValueError("Total scoring weight must be positive.")

    return result / total


# ============================================================
# 2. CADTP Ranker
# ============================================================

class CADTPRanker:
    """
    Fit consensus-aware protein prioritisation scores.

    The class stores:
        communities : target communities for each Louvain seed
        all_scores  : scores for all non-target proteins
        rankings    : eligible candidates ranked within tracks
    """

    def __init__(self, cfg: ExperimentConfig):

        self.cfg = cfg

        self.target = cfg.project["target_protein"]

        self.communities = None
        self.all_scores = None
        self.rankings = None

    # --------------------------------------------------------
    # 3. Candidate eligibility
    # --------------------------------------------------------

    def _candidate_mask(self, baseline: pd.DataFrame) -> pd.Series:
        """
        Reproduce the original candidate screening rules.

        Exclusion affects ranking eligibility only.
        Excluded proteins remain in the network.
        """

        mask = _as_bool(
            baseline["candidate_eligible"]
        )

        settings = self.cfg.candidates

        if settings["require_orf"]:
            valid_orf = (
                baseline["orf"].notna()
                & baseline["orf"].astype("string").str.strip().ne("")
            ).fillna(False)

            mask &= valid_orf.astype(bool)

        if settings["exclude_essential"]:
            essential = _as_bool(
                baseline["is_listed_essential"]
            )

            mask &= ~essential

        excluded_ids = set(
            settings["excluded_reference_proteins"]
        ) | set(
            settings["excluded_readout_proteins"]
        )

        if settings["exclude_target"]:
            excluded_ids.add(self.target)

        mask &= ~baseline.index.isin(excluded_ids)

        return mask.rename("eligible")

    # --------------------------------------------------------
    # 4. Fit CADTP
    # --------------------------------------------------------

    def fit(
        self,
        G_full: nx.Graph,
        G_hlj1: nx.Graph,
        node2vec: pd.DataFrame,
        baseline: pd.DataFrame,
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Fit the CADTP training ensemble.

        Parameters
        ----------
        G_full:
            Full functional graph for Louvain.

        G_hlj1:
            Functional connected component containing HLJ1.

        node2vec:
            Node2Vec consensus DataFrame containing column N.

        baseline:
            Notebook 02 protein baseline features.

        seeds:
            Louvain random seeds. Defaults to training seeds.

        Returns
        -------
        pd.DataFrame:
            Eligible candidates with track-specific ranking.
        """

        if self.target not in G_full or self.target not in G_hlj1:
            raise ValueError("HLJ1 is missing from the graphs.")

        if not set(G_hlj1.nodes()).issubset(G_full.nodes()):
            raise ValueError(
                "Target component is not contained in the full graph."
            )

        if not node2vec.index.is_unique:
            raise ValueError("Duplicate Node2Vec protein IDs.")

        if not baseline.index.is_unique:
            raise ValueError("Duplicate baseline protein IDs.")

        if "N" not in node2vec.columns:
            raise ValueError("Node2Vec data requires column N.")

        # Consistent score universe: all non-HLJ1 nodes
        nodes = sorted(
            v for v in G_hlj1.nodes()
            if v != self.target
        )

        node_index = pd.Index(
            nodes,
            name="string_id",
        )

        if not node_index.isin(node2vec.index).all():
            raise ValueError("Missing Node2Vec scores.")

        if not node_index.isin(baseline.index).all():
            raise ValueError("Missing baseline protein features.")

        baseline = baseline.reindex(node_index)

        # ----------------------------------------------------
        # A. Node2Vec component
        # ----------------------------------------------------

        N = node2vec["N"].reindex(node_index).astype(float)

        if not np.isfinite(N).all() or not N.between(0, 1).all():
            raise ValueError("Invalid Node2Vec percentile scores.")

        # ----------------------------------------------------
        # B. Random Walk with Restart
        # ----------------------------------------------------

        rwr_raw = compute_rwr(
            G_hlj1,
            target_id=self.target,
            alpha=self.cfg.rwr["alpha"],
            weight=self.cfg.rwr["weight"],
        )

        R = rank_percentile(
            rwr_raw.drop(index=self.target)
        ).reindex(node_index)

        # ----------------------------------------------------
        # C. Louvain ensemble
        # ----------------------------------------------------

        if seeds is None:
            seeds = self.cfg.louvain["training_seeds"]

        self.communities = run_louvain_ensemble(
            G_full,
            target_id=self.target,
            seeds=seeds,
            resolution=self.cfg.louvain["resolution"],
            weight=self.cfg.louvain["weight"],
        )

        membership = community_membership_counts(
            self.communities,
            nodes=nodes,
        ).reindex(node_index)

        # ----------------------------------------------------
        # D. Consensus community affinity
        # ----------------------------------------------------

        affinity_raw = mean_community_affinity(
            G_hlj1,
            self.communities,
            smoothing=self.cfg.cadtp["affinity_smoothing"],
            weight=self.cfg.network["weight_attribute"],
        )

        # Average raw affinities FIRST,
        # then compute rank percentiles.
        C = rank_percentile(
            affinity_raw.drop(index=self.target)
        ).reindex(node_index)

        # ----------------------------------------------------
        # E. Assemble scores
        # ----------------------------------------------------

        scores = pd.DataFrame(
            {
                "N": N,
                "R": R,
                "C": C,
            },
            index=node_index,
        )

        scores = scores.join(membership)

        votes = scores["membership_count"]
        n_runs = len(self.communities)

        # Exact integer cutoffs avoid floating-point
        # errors at the Local/Cross boundaries.
        local_threshold = Fraction(
            str(self.cfg.cadtp["local_threshold"])
        )

        cross_threshold = Fraction(
            str(self.cfg.cadtp["cross_threshold"])
        )

        min_local_votes = math.ceil(
            local_threshold * n_runs
        )

        max_cross_votes = math.floor(
            cross_threshold * n_runs
        )

        scores["track"] = np.select(
            [
                votes >= min_local_votes,
                votes <= max_cross_votes,
            ],
            [
                "Local",
                "Cross",
            ],
            default="Boundary",
        )

        # ----------------------------------------------------
        # F. Track-specific CADTP scores
        # ----------------------------------------------------

        local_score = _weighted_average(
            scores,
            self.cfg.cadtp["local_weights"],
        )

        cross_score = _weighted_average(
            scores,
            self.cfg.cadtp["cross_weights"],
        )

        boundary_score = _weighted_average(
            scores,
            self.cfg.cadtp["boundary_weights"],
        )

        scores["cadtp_score"] = np.select(
            [
                scores["track"].eq("Local"),
                scores["track"].eq("Cross"),
            ],
            [
                local_score,
                cross_score,
            ],
            default=boundary_score,
        )

        # ----------------------------------------------------
        # G. Candidate filtering
        # ----------------------------------------------------

        scores["eligible"] = self._candidate_mask(baseline)

        # Keep biological identifiers and selected
        # baseline features for downstream analysis.
        metadata_columns = [
            col for col in (
                "orf",
                "preferred_name",
                "degree",
                "h_hops",
                "is_listed_essential",
            )
            if col in baseline.columns
        ]

        scores = scores.join(
            baseline[metadata_columns]
        )

        self.all_scores = scores

        eligible = scores.loc[
            scores["eligible"]
        ].copy()

        # ----------------------------------------------------
        # H. Independent rankings within each track
        # ----------------------------------------------------

        eligible = eligible.reset_index()

        eligible = eligible.sort_values(
            ["track", "cadtp_score", "string_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )

        eligible["rank_within_track"] = (
            eligible.groupby("track").cumcount() + 1
        )

        eligible = eligible.set_index("string_id")

        eligible.index.name = "string_id"

        self.rankings = eligible

        return eligible

    # --------------------------------------------------------
    # 5. Retrieve top candidates
    # --------------------------------------------------------

    def top_candidates(self, track: str, k: int = 20) -> pd.DataFrame:

        if self.rankings is None:
            raise RuntimeError("Fit CADTP before ranking.")

        if track not in ("Local", "Cross", "Boundary"):
            raise ValueError(f"Unknown track: {track}")

        if k <= 0:
            raise ValueError("k must be positive.")

        return (
            self.rankings
            .loc[self.rankings["track"].eq(track)]
            .nsmallest(k, "rank_within_track")
        )
