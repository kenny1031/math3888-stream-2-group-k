"""
Evaluation utilities.
Includes:
    - Jaccard similarity
    - Precision@k
    - GO baseline comparison
    - Track-background fold enrichment
Matched randomisation and GO ontology parsing will
be added separately.
"""
from collections.abc import Iterable, Sequence
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. Jaccard similarity
# ============================================================

def jaccard_similarity(
    a: Iterable[str],
    b: Iterable[str],
) -> float:
    """Jaccard similarity between two sets."""
    A = set(a)
    B = set(b)

    union = A | B

    if not union:
        return 1.0

    return len(A & B) / len(union)


# ============================================================
# 2. Precision@k
# ============================================================

def precision_at_k(
    ranked_ids: Sequence[str],
    positive_ids: set[str],
    k: int,
) -> float:
    """
    Precision@k for a ranked candidate list.

    Missing GO-positive labels are treated as non-positive
    for this observed-annotation metric.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if len(ranked_ids) < k:
        raise ValueError(f"Ranking contains fewer than {k} proteins.")

    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Ranking contains duplicate protein IDs.")

    top = ranked_ids[:k]
    positives = sum(protein in positive_ids for protein in top)

    return positives / k


# ============================================================
# 3. Fold enrichment
# ============================================================

def fold_enrichment(observed_precision: float, background_rate: float) -> float:
    """
    Fold enrichment relative to a track's
    observed GO-positive prevalence.
    """
    if not 0 <= observed_precision <= 1:
        raise ValueError("Observed precision must be in [0, 1].")

    if not 0 <= background_rate <= 1:
        raise ValueError("Background rate must be in [0, 1].")

    if background_rate == 0:
        return np.nan

    return observed_precision / background_rate


# ============================================================
# 4. Deterministic ranking
# ============================================================


def rank_candidates(
    candidates: pd.DataFrame,
    score_column: str,
    tie_breaker: str | None = None,
) -> list[str]:
    """
    Rank proteins by descending score.

    If tie_breaker is provided, break ties using that
    column (e.g. ORF, matching Notebook 06).

    Otherwise break ties by STRING protein ID.
    """
    if score_column not in candidates.columns:
        raise KeyError(
            f"Missing score column: {score_column}"
        )

    if not candidates.index.is_unique:
        raise ValueError("Duplicate protein IDs.")

    scores = pd.to_numeric(
        candidates[score_column],
        errors="raise",
    )

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite ranking scores.")

    if tie_breaker is not None:
        if tie_breaker not in candidates.columns:
            raise KeyError(
                f"Missing tie-breaker column: {tie_breaker}"
            )

        ties = candidates[tie_breaker].astype(str).to_numpy()
    else:
        ties = candidates.index.astype(str).to_numpy()

    ordered = pd.DataFrame({
        "protein_id": candidates.index.astype(str),
        "score": scores.to_numpy(),
        "tie": ties,
    })

    ordered = ordered.sort_values(
        ["score", "tie", "protein_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    return ordered["protein_id"].tolist()


# ============================================================
# 5. GO baseline comparison
# ============================================================

def compare_go_baselines(
    rankings: pd.DataFrame,
    positive_ids: set[str],
    k_values: Sequence[int] = (10, 20, 50),
    tracks: Sequence[str] = ("Local", "Cross", "Boundary"),
) -> pd.DataFrame:
    """
    Compare methods within fixed eligible CADTP tracks.

    Parameters
    ----------
    rankings:
        Eligible CADTP candidates indexed by STRING ID.

        Required columns:
            track, degree, N, R, cadtp_score

    positive_ids:
        Set of STRING IDs annotated as GO-positive.

    k_values:
        Cutoffs for Precision@k.

    Returns
    -------
    DataFrame:
        Per-track, per-method evaluation results.
    """

    required = {
        "track",
        "degree",
        "N",
        "R",
        "cadtp_score",
    }

    missing = required - set(rankings.columns)

    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")

    if not rankings.index.is_unique:
        raise ValueError("Rankings contain duplicate protein IDs.")

    methods = {
        "Degree": "degree",
        "Node2Vec": "N",
        "RWR": "R",
        "CADTP": "cadtp_score",
    }

    records = []

    for track in tracks:
        pool = rankings.loc[rankings["track"].eq(track)]
        if pool.empty:
            continue

        pool_size = len(pool)
        positive_count = sum(protein in positive_ids for protein in pool.index)

        background_rate = positive_count / pool_size

        for method, score_column in methods.items():
            ordered_ids = rank_candidates(
                pool,
                score_column,
                tie_breaker="orf" if "orf" in pool.columns else None,
            )
            for k in k_values:
                if k > pool_size:
                    continue

                precision = precision_at_k(
                    ordered_ids,
                    positive_ids,
                    k,
                )

                records.append({
                    "track": track,
                    "method": method,
                    "k": k,
                    "pool_size": pool_size,
                    "background_positive": positive_count,
                    "background_rate": background_rate,
                    "observed_positive": round(
                        precision * k
                    ),
                    "precision": precision,
                    "fold_enrichment": fold_enrichment(
                        precision,
                        background_rate,
                    ),
                })

    return pd.DataFrame.from_records(records)



# ============================================================
# 6. Degree / distance matching strata
# ============================================================

def add_matching_strata(evaluation: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the exact degree/distance strata used
    in Notebook 06.

    Degree bins:
        0-5, 6-20, 21-60, 61+

    Distance bins:
        1, 2, 3, 4+ (clipped at 4)
    """
    required = {"degree", "h_hops"}

    if not required.issubset(evaluation.columns):
        raise ValueError(f"Missing columns: {required - set(evaluation.columns)}")

    if not np.isfinite(evaluation[["degree", "h_hops"]].to_numpy(dtype=float)).all():
        raise ValueError("Degree/distance contain invalid values.")

    result = evaluation.copy()

    result["degree_bin"] = pd.cut(
        result["degree"],
        bins=[-np.inf, 5, 20, 60, np.inf],
        labels=["0-5", "6-20", "21-60", "61+"],
    )

    result["distance_bin"] = (
        result["h_hops"]
        .clip(upper=4)
        .astype(int)
        .astype(str)
    )

    result["stratum"] = result["distance_bin"] + "_" + result["degree_bin"].astype(str)

    if result["stratum"].isna().any():
        raise ValueError("Some candidates have no matching stratum.")

    return result


# ============================================================
# 7. Matched randomisation
# ============================================================

def matched_random_test(
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    rng: np.random.Generator,
    label: str = "proteostasis_any",
    n_permutations: int = 2000,
) -> dict:
    """
    Degree/distance-matched randomisation test.

    Sampling is:
        - Restricted to the same candidate track
        - Matched to selected degree/distance strata
        - Without replacement within each stratum

    The selected proteins remain in the null sampling pool,
    matching Notebook 06.

    IMPORTANT:
        Pass the same RNG across successive method tests
        to preserve the original random sequence.
    """
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    if not pool.index.is_unique:
        raise ValueError("Pool contains duplicate IDs.")

    if not selected.index.is_unique:
        raise ValueError("Selected candidates contain duplicate IDs.")

    if not selected.index.isin(pool.index).all():
        raise ValueError("Selected candidates must belong to pool.")

    for column in ("stratum", label):
        if column not in pool or column not in selected:
            raise ValueError(f"Missing column: {column}")

    # Number of selected proteins in each stratum.
    # Preserve the value_counts ordering from Notebook 06.
    quotas = selected["stratum"].value_counts()

    groups = {
        stratum: group.index.to_numpy()
        for stratum, group in pool.groupby("stratum")
    }

    for stratum, count in quotas.items():
        if stratum not in groups or count > len(groups[stratum]):
            raise ValueError(f"Insufficient proteins in stratum {stratum}")

    observed = int(selected[label].sum())
    label_map = pool[label].to_dict()
    simulated = np.empty(n_permutations, dtype=int)

    for b in range(n_permutations):
        sampled = []
        for stratum, count in quotas.items():
            choices = rng.choice(
                groups[stratum],
                size=int(count),
                replace=False,
            )
            sampled.extend(choices)

        simulated[b] = sum(label_map[v] for v in sampled)

    null_mean = simulated.mean()

    empirical_p = (1 + np.count_nonzero(simulated >= observed)) / (n_permutations + 1)

    return {
        "observed_hits": observed,
        "null_mean_hits": null_mean,
        "null_std_hits": simulated.std(ddof=1),
        "matched_fold_enrichment": (
            observed / null_mean
            if null_mean > 0
            else np.nan
        ),
        "empirical_p_value": empirical_p,
    }


# ============================================================
# 8. Multiple testing correction
# ============================================================

def benjamini_hochberg(results: pd.DataFrame, alpha: float=0.05) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction across
    the supplied family of hypothesis tests.
    """

    if "empirical_p_value" not in results.columns:
        raise ValueError("Missing empirical_p_value.")

    if results.empty:
        raise ValueError("Cannot correct an empty result set.")

    adjusted = results.copy()

    adjusted["q_value_BH"] = multipletests(
        adjusted["empirical_p_value"],
        alpha=alpha,
        method="fdr_bh",
    )[1]

    adjusted["significant_FDR_05"] = adjusted["q_value_BH"] < alpha

    return adjusted


# ============================================================
# 9. Run the original Notebook 06 comparisons
# ============================================================

def run_matched_go_comparison(
    evaluation: pd.DataFrame,
    n_permutations: int = 2000,
    seed: int = 3888,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce Notebook 06 matched GO tests.

    Uses only Local and Cross tracks.

    The 8 tests share ONE RNG, and execute in the
    original track/method order.
    """

    required = {
        "consensus_track",
        "orf",
        "degree",
        "h_hops",
        "N",
        "R",
        "hybrid_score",
        "proteostasis_any",
    }

    if not required.issubset(evaluation.columns):
        raise ValueError(
            f"Missing evaluation columns: "
            f"{required - set(evaluation.columns)}"
        )

    evaluation = add_matching_strata(evaluation)

    # Preserve original Notebook 06 method order
    methods = {
        "Degree": "degree",
        "Node2Vec": "N",
        "RWR": "R",
        "CADTP": "hybrid_score",
    }

    tracks = ["A_local_consensus", "B_cross_consensus"]

    # Single shared generator across all eight tests
    rng = np.random.default_rng(seed)
    records = []

    for track in tracks:
        pool = evaluation.loc[evaluation["consensus_track"] == track].copy()
        if len(pool) < 20:
            raise ValueError(f"Insufficient candidates in {track}")

        for method, score_col in methods.items():
            # Original Notebook tie-breaking: ORF ascending
            selected = (pool.sort_values(
                [score_col, "orf"],
                ascending=[False, True],
                kind="mergesort",
            ).head(20))

            result = matched_random_test(
                pool=pool,
                selected=selected,
                rng=rng,
                n_permutations=n_permutations,
            )

            records.append({
                "track": track,
                "method": method,
                "k": 20,
                **result,
            })

    matched = pd.DataFrame(records)
    adjusted = benjamini_hochberg(matched)

    return matched, adjusted