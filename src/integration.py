"""Cross-method evidence integration.
Run from repository root:
    python -m src.integration

Expects:
    outputs/rankings/cadtp_training_refactor.csv
    outputs/junxi/mcl_results.csv
    outputs/alex/betweenness_change.csv
    outputs/jadon/all_metrics_tau0.7.csv
    outputs/jadon/final_candidates_tau0.7.csv

No cross-method average score is constructed; each method has different estimands.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["cadtp", "mcl", "alex", "jadon"]
CANDIDATES = [
    "APJ1", "JJJ2", "SSE1", "UFD2", "SSM4",
    "FES1", "MDJ1", "ERJ5", "YOS9", "USA1",
    "YDJ1", "SIS1", "XDJ1", "ERP5",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")

    return pd.read_csv(path, low_memory=False)


def _key(values: pd.Series, label: str) -> pd.Series:
    ids = values.astype("string").str.strip()
    ids = ids.where(ids.str.startswith("4932."), "4932." + ids)

    if ids.isna().any() or (ids == "4932.").any():
        raise ValueError(f"Missing protein IDs in {label}")

    if ids.duplicated().any():
        example = ids[ids.duplicated()].iloc[0]
        raise ValueError(f"Duplicate STRING ID in {label}: {example}")

    return ids


def _set_index(frame: pd.DataFrame, key_col: str, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.Index(_key(frame[key_col], label), name="string_id")

    return frame


def _bool_yes_no(s: pd.Series, label: str) -> pd.Series:
    choices = s.astype("string").str.strip().str.lower()

    if not choices.isin(["yes", "no"]).all():
        raise ValueError(f"Expected yes/no values in {label}")

    return choices.eq("yes").astype(bool)


def _require_cols(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def prepare_cadtp(path: Path) -> pd.DataFrame:
    df = _read(path)
    id_col = next((v for v in ["string_id", "string_protein_id"] if v in df), None)
    if id_col is None:
        raise ValueError("CADTP file must contain string_id or string_protein_id")

    df = _set_index(df, id_col, "CADTP")
    name_col = "preferred_name" if "preferred_name" in df else "gene"
    score_col = "cadtp_score" if "cadtp_score" in df else "hybrid_score"
    rank_col = next((v for v in ["cadtp_train_rank", "train_rank", "track_rank"] if v in df), None)
    _require_cols(df, ["orf", name_col, "track", score_col], "CADTP")
    result = pd.DataFrame(index=df.index)
    result["orf"] = df["orf"].astype("string")
    result["gene"] = df[name_col].astype("string")
    result["cadtp_track"] = df["track"]
    result["cadtp_score"] = pd.to_numeric(df[score_col], errors="raise")

    if rank_col is not None:
        result["cadtp_rank_within_track"] = pd.to_numeric(df[rank_col], errors="raise").astype("Int64")
    else:
        # Deterministic track-specific rank, including STRING-ID tie-breaker.
        order = result.sort_values(["cadtp_track", "cadtp_score"], ascending=[True, False], kind="mergesort")
        order = order.reset_index().sort_values(
            ["cadtp_track", "cadtp_score", "string_id"],
            ascending=[True, False, True],
            kind="mergesort"
        )
        order["rank"] = order.groupby("cadtp_track", sort=False).cumcount() + 1
        result["cadtp_rank_within_track"] = (
            order
            .set_index("string_id")["rank"]
            .reindex(result.index)
            .astype("Int64")
        )

    for input_col, output_col in (
        ("h_hops", "cadtp_hops_to_hlj1"),
        ("membership_count", "cadtp_training_membership_votes"),
        ("N", "cadtp_node2vec_percentile"),
        ("R", "cadtp_rwr_percentile"),
        ("C", "cadtp_community_percentile")):
        if input_col in df:
            result[output_col] = df[input_col]

    return result


def prepare_mcl(path: Path) -> pd.DataFrame:
    df = _read(path)
    _require_cols(df, ["protein_id", "gene_name", "same_HLJ1_group_count"], "MCL")
    df = _set_index(df, "protein_id", "MCL")
    seeds = ["r1_5", "r2_0", "r2_5", "r3_0", "r4_0", "r5_0"]
    columns = [f"same_HLJ1_group_{r}" for r in seeds]
    _require_cols(df, columns, "MCL")
    result = pd.DataFrame(index=df.index)
    result["mcl_gene"] = df["gene_name"].astype("string")
    for suffix, col in zip(seeds, columns):
        result[f"mcl_member_{suffix}"] = _bool_yes_no(df[col], col)
    count = result[[f"mcl_member_{s}" for s in seeds]].sum(axis=1)
    reported = pd.to_numeric(df["same_HLJ1_group_count"], errors="raise").astype(int)
    if not np.array_equal(count.to_numpy(), reported.to_numpy()):
        raise ValueError("MCL co-clustering count differs from six run flags")
    result["mcl_coclustering_count"] = reported
    result["mcl_coclustering_fraction"] = reported / len(seeds)
    return result


def prepare_alex(path: Path) -> pd.DataFrame:
    df = _read(path)
    _require_cols(df, ["protein", "relative", "distance", "essential", "local"], "Alex")
    df = _set_index(df, "protein", "Alex")
    result = pd.DataFrame(index=df.index)
    result["alex_relative_raw"] = pd.to_numeric(df["relative"], errors="raise")
    result["alex_distance_to_hlj1"] = pd.to_numeric(df["distance"], errors="coerce")
    result["alex_essential"] = df["essential"].astype("boolean")
    result["alex_distance_le_2"] = df["local"].astype("boolean")
    # 'relative' follows the source's label: its percentage/normalization
    # interpretation has NOT been confirmed.
    return result


def prepare_jadon(metrics_path: Path, selected_path: Path) -> pd.DataFrame:
    df = _read(metrics_path)
    id_col = "Unnamed: 0" if "Unnamed: 0" in df else "string_id"
    _require_cols(df, [id_col, "name", "pagerank_erad", "pct_change", "bt_before", "bt_pct_change"], "Jadon metrics")
    df = _set_index(df, id_col, "Jadon metrics")
    fields = [
        "degree", "is_seed", "pagerank_erad", "pct_change", "bt_before",
        "bt_pct_change", "d_seed", "d_hlj1", "J_mean", "J_max",
        "with_HLJ1", "with_ERAD", "with_ERAD_after", "ERAD_switch",
        "hlj1_neighbour", "essential",
    ]
    result = pd.DataFrame(index=df.index)
    result["jadon_gene"] = df["name"].astype("string")
    for field in fields:
        if field in df:
            result[f"jadon_{field}"] = df[field]

    chosen = _read(selected_path)
    chosen_id = "Unnamed: 0" if "Unnamed: 0" in chosen else "string_id"
    _require_cols(chosen, [chosen_id, "group", "reasons"], "Jadon final candidates")
    chosen = _set_index(chosen, chosen_id, "Jadon final")
    missing = chosen.index.difference(result.index)
    if len(missing):
        raise ValueError(f"Jadon selected candidates not in metric table: {list(missing[:4])}")
    result["jadon_selected"] = result.index.isin(chosen.index)
    result["jadon_selection_group"] = chosen["group"].reindex(result.index)
    result["jadon_selection_reasons"] = chosen["reasons"].reindex(result.index)
    return result


def integrate(
    cadtp_path: Path,
    mcl_path: Path,
    alex_path: Path,
    jadon_metrics_path: Path,
    jadon_selected_path: Path,
) -> pd.DataFrame:
    base = prepare_cadtp(cadtp_path)
    mcl = prepare_mcl(mcl_path)
    alex = prepare_alex(alex_path)
    jadon = prepare_jadon(jadon_metrics_path, jadon_selected_path)
    output = base.join(mcl, how="left").join(alex, how="left").join(jadon, how="left")
    if len(output) != len(base) or not output.index.is_unique:
        raise AssertionError("Integration changed the CADTP candidate universe")
    for method, reference in (("MCL", mcl), ("Jadon", jadon)):
        missing = base.index.difference(reference.index)
        if len(missing):
            raise ValueError(f"{method} does not cover {len(missing)} CADTP proteins; example: {missing[0]}")
    # Fail on wrong gene labels rather than silently treating it as agreement.
    for column in ("mcl_gene", "jadon_gene"):
        bad = output[column].notna() & output["gene"].notna() & (output[column].astype("string") != output["gene"].astype("string"))
        if bad.any():
            example = output.index[bad][0]
            raise ValueError(f"Protein naming mismatch ({column}) for {example}")
    output["alex_available"] = output["alex_relative_raw"].notna()
    return output.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate four HLJ1 network analyses")
    parser.add_argument("--cadtp", type=Path, default=Path("outputs/rankings/cadtp_training_refactor.csv"))
    parser.add_argument("--mcl", type=Path, default=Path("outputs/junxi/mcl_results.csv"))
    parser.add_argument("--alex", type=Path, default=Path("outputs/alex/betweenness_change.csv"))
    parser.add_argument("--jadon-metrics", type=Path, default=Path("outputs/jadon/all_metrics_tau0.7.csv"))
    parser.add_argument("--jadon-final", type=Path, default=Path("outputs/jadon/final_candidates_tau0.7.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/integrated"))
    args = parser.parse_args()

    combined = integrate(args.cadtp, args.mcl, args.alex, args.jadon_metrics, args.jadon_final)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "cross_method_evidence.csv", index=False)

    key = combined[combined["gene"].isin(CANDIDATES)].copy()
    key["_order"] = key["gene"].map({v: i for i, v in enumerate(CANDIDATES)})
    key = key.sort_values("_order").drop(columns="_order")
    key.to_csv(args.output_dir / "selected_protein_evidence.csv", index=False)

    print(f"CADTP candidate rows: {len(combined):,}")
    print(f"MCL matched: {combined['mcl_coclustering_count'].notna().sum():,}")
    print(f"Jadon matched: {combined['jadon_gene'].notna().sum():,}")
    print(f"Alex matched: {combined['alex_available'].sum():,} (missing is not zero change)")
    print(f"Jadon selected among eligible: {combined['jadon_selected'].sum():,}")
    print("\nSelected proteins:")

    fields = [
        "gene",
        "cadtp_track",
        "cadtp_rank_within_track",
        "mcl_coclustering_count",
        "alex_relative_raw",
        "jadon_pct_change",
        "jadon_selected"
    ]

    print(key[fields].to_string(index=False))
    print(f"\nSaved CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
