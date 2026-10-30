"""Prepare STRING v12.0 yeast data for the MATH3888 HLJ1 network analysis.

Run from any directory:
    python scripts/prepare_string_data.py

Only the four STRING .gz files are needed for this preprocessing step.
The GO and essential-protein inputs are consumed by later analyses.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd


ORF_PATTERN = re.compile(r"^Y[A-P][LR]\d{3}[CW](?:-[A-Z])?$")
REQUIRED_FILES = {
    "functional": "4932.protein.links.full.v12.0.txt.gz",
    "physical": "4932.protein.physical.links.full.v12.0.txt.gz",
    "info": "4932.protein.info.v12.0.txt.gz",
    "aliases": "4932.protein.aliases.v12.0.txt.gz",
}


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lstrip("#") for column in frame.columns]
    return frame


def read_protein_metadata(files: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for role in ("functional", "physical"):
        head = pd.read_csv(files[role], sep=r"\s+", compression="gzip", nrows=0)
        print(f"{role} columns: {clean_columns(head).columns.tolist()}")
    for role in ("info", "aliases"):
        head = pd.read_csv(files[role], sep="\t", compression="gzip", nrows=0)
        print(f"{role} columns: {clean_columns(head).columns.tolist()}")

    info = clean_columns(pd.read_csv(files["info"], sep="\t", compression="gzip"))
    aliases = clean_columns(pd.read_csv(files["aliases"], sep="\t", compression="gzip"))

    if not {"string_protein_id", "preferred_name"} <= set(info.columns):
        raise ValueError(f"Unexpected protein info columns: {info.columns.tolist()}")
    if not {"string_protein_id", "alias"} <= set(aliases.columns):
        raise ValueError(f"Unexpected alias columns: {aliases.columns.tolist()}")
    if not info["string_protein_id"].is_unique:
        raise ValueError("Protein info table has duplicate STRING IDs")

    info["string_protein_id"] = info["string_protein_id"].astype(str)
    aliases["string_protein_id"] = aliases["string_protein_id"].astype(str)
    aliases["alias"] = aliases["alias"].astype(str)
    print(f"STRING proteins: {len(info):,}; alias rows: {len(aliases):,}")
    return info, aliases


def construct_id_map(info: pd.DataFrame, aliases: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    alias_orfs = aliases.loc[
        aliases["alias"].map(lambda value: bool(ORF_PATTERN.fullmatch(value))),
        ["string_protein_id", "alias"],
    ]
    alias_dict = (
        alias_orfs.groupby("string_protein_id")["alias"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )

    id_map = info[["string_protein_id", "preferred_name"]].copy()
    id_map["id_suffix"] = id_map["string_protein_id"].str.split(".", n=1).str[-1]
    id_map["orf_aliases"] = id_map["string_protein_id"].map(
        lambda string_id: alias_dict.get(string_id, [])
    )

    def decide_orf(row: pd.Series) -> pd.Series:
        suffix = row["id_suffix"]
        direct = suffix if ORF_PATTERN.fullmatch(suffix) else None
        candidates = row["orf_aliases"]
        if direct is not None:
            if candidates and direct not in candidates:
                return pd.Series([None, "conflict"])
            return pd.Series([direct, "string_id"])
        if len(candidates) == 1:
            return pd.Series([candidates[0], "unique_alias"])
        if len(candidates) > 1:
            return pd.Series([None, "ambiguous_alias"])
        return pd.Series([None, "unmapped"])

    id_map[["orf", "mapping_status"]] = id_map.apply(decide_orf, axis=1)
    print("ORF mapping statuses:")
    print(id_map["mapping_status"].value_counts().to_string())

    duplicate_orfs = id_map.loc[id_map["orf"].notna()].groupby("orf").size()
    duplicate_orfs = duplicate_orfs[duplicate_orfs > 1]
    if not duplicate_orfs.empty:
        print(f"Warning: {len(duplicate_orfs)} ORFs map to multiple STRING IDs")

    hlj1_hits = id_map.loc[id_map["orf"].eq("YMR161W")]
    if len(hlj1_hits) != 1:
        raise ValueError("HLJ1/YMR161W must have exactly one STRING ID mapping")
    hlj1_id = str(hlj1_hits.iloc[0]["string_protein_id"])
    print(f"HLJ1 STRING ID: {hlj1_id}")
    return id_map, hlj1_id


def prepare_edges(path: Path, min_score: int = 400, chunk_size: int = 150_000):
    """Keep all original STRING columns, matching the notebook output schema."""
    kept_chunks = []
    raw_rows = 0
    self_loops = 0

    reader = pd.read_csv(path, sep=r"\s+", compression="gzip", chunksize=chunk_size)
    for chunk in reader:
        chunk = clean_columns(chunk)
        required = {"protein1", "protein2", "combined_score"}
        if not required <= set(chunk.columns):
            raise ValueError(f"Missing columns in {path.name}: {chunk.columns.tolist()}")
        raw_rows += len(chunk)
        chunk["combined_score"] = pd.to_numeric(chunk["combined_score"], errors="raise")
        if not chunk["protein1"].astype(str).str.startswith("4932.").all():
            raise ValueError(f"Non-4932 protein1 in {path.name}")
        if not chunk["protein2"].astype(str).str.startswith("4932.").all():
            raise ValueError(f"Non-4932 protein2 in {path.name}")

        chunk = chunk.loc[chunk["combined_score"] >= min_score].copy()
        self_loops += int((chunk["protein1"] == chunk["protein2"]).sum())
        chunk = chunk.loc[chunk["protein1"] != chunk["protein2"]].copy()
        left, right = chunk["protein1"].astype(str), chunk["protein2"].astype(str)
        left_first = left < right
        chunk["u"] = left.where(left_first, right)
        chunk["v"] = right.where(left_first, left)
        kept_chunks.append(chunk)

    if not kept_chunks:
        raise ValueError(f"No rows in {path.name} meet the minimum score of {min_score}")

    edges = pd.concat(kept_chunks, ignore_index=True)
    disagreement = edges.groupby(["u", "v"])["combined_score"].nunique().gt(1).sum()
    if disagreement:
        raise ValueError(
            f"{path.name}: {disagreement} undirected pairs have conflicting scores"
        )
    rows_before_dedup = len(edges)
    edges = (
        edges.sort_values(["u", "v", "combined_score"])
        .drop_duplicates(["u", "v"], keep="last")
        .reset_index(drop=True)
    )
    if not edges["combined_score"].between(min_score, 1000).all():
        raise ValueError(f"Unexpected combined_score in {path.name}")

    report = {
        "raw_rows": raw_rows,
        "self_loops_after_score_filter": self_loops,
        "rows_score_ge_400_before_dedup": rows_before_dedup,
        "unique_undirected_edges_ge_400": len(edges),
        "reverse_or_duplicate_rows_removed": rows_before_dedup - len(edges),
    }
    return edges, report


def network_summary(
    edge_tables: dict[str, pd.DataFrame], info: pd.DataFrame, hlj1_id: str
) -> pd.DataFrame:
    rows = []
    for layer, edges in edge_tables.items():
        for cutoff in (400, 700, 900):
            graph = nx.Graph()
            graph.add_nodes_from(info["string_protein_id"])
            selected = edges.loc[
                edges["combined_score"] >= cutoff, ["u", "v", "combined_score"]
            ]
            graph.add_weighted_edges_from(
                (u, v, score / 1000)
                for u, v, score in selected.itertuples(index=False, name=None)
            )
            rows.append(
                {
                    "layer": layer,
                    "threshold": cutoff / 1000,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "isolates": nx.number_of_isolates(graph),
                    "components": nx.number_connected_components(graph),
                    "HLJ1_degree": graph.degree(hlj1_id),
                    "HLJ1_component_size": len(nx.node_connected_component(graph, hlj1_id)),
                }
            )
    return pd.DataFrame(rows)


def prepare_seeds(meta: Path) -> Path:
    """Create the notebook's seed template if absent; preserve existing curated files."""
    seed_path = meta / "erad_seeds.csv"
    if not seed_path.exists():
        pd.DataFrame(
            [{
                "orf": "YMR161W",
                "set": "core",
                "source_url": "https://www.kegg.jp/entry/sce:YMR161W",
                "rationale": "HLJ1; starting anchor for ERAD-related analysis",
            }]
        ).to_csv(seed_path, index=False)
        print(f"Created seed template: {seed_path}")

    pathway_url = "https://www.kegg.jp/entry/sce04141"
    proposed = pd.DataFrame(
        [
            ("YOL013C", "HRD1: ER-associated E3 ubiquitin ligase"),
            ("YLR207W", "HRD3: HRD ubiquitin-ligase complex subunit"),
            ("YBR201W", "DER1: derlin associated with ER protein degradation"),
            ("YMR022W", "UBC7: E2 ubiquitin-conjugating protein"),
            ("YMR264W", "CUE1: ERAD-associated component listed in sce04141"),
        ],
        columns=["orf", "rationale"],
    )
    proposed["set"] = "core"
    proposed["source_url"] = pathway_url
    proposed = proposed[["orf", "set", "source_url", "rationale"]]

    existing = pd.read_csv(seed_path, dtype=str).fillna("")
    required = {"orf", "set", "source_url", "rationale"}
    if not required <= set(existing.columns):
        raise ValueError(f"Missing seed columns: {sorted(required - set(existing.columns))}")
    additions = proposed.loc[~proposed["orf"].isin(existing["orf"])]
    if not additions.empty:
        existing = pd.concat([existing, additions], ignore_index=True)
        existing.to_csv(seed_path, index=False)
        print(f"Added {len(additions)} proposed ERAD seed rows")
    print(f"ERAD seed rows: {len(existing)}")
    return seed_path


def freeze_design(
    meta: Path,
    id_map: pd.DataFrame,
    hlj1_id: str,
    seed_path: Path,
    raw_hashes: dict[str, str],
    allow_design_mismatch: bool,
) -> None:
    seeds = pd.read_csv(seed_path, dtype=str).fillna("")
    required = {"orf", "set", "source_url", "rationale"}
    if not required <= set(seeds.columns):
        raise ValueError(f"Missing seed columns: {sorted(required - set(seeds.columns))}")
    if not seeds["orf"].is_unique:
        raise ValueError("ERAD seed table contains duplicate ORFs")

    core_seeds = seeds.loc[seeds["set"] == "core"].copy()
    if "YMR161W" not in set(core_seeds["orf"]):
        raise ValueError("HLJ1/YMR161W must be a core ERAD seed")

    candidates = (
        id_map.loc[
            id_map["orf"].isin(core_seeds["orf"]),
            ["orf", "string_protein_id", "mapping_status"],
        ]
        .dropna(subset=["orf", "string_protein_id"])
        .drop_duplicates(subset=["orf", "string_protein_id"])
    )
    counts = candidates.groupby("orf")["string_protein_id"].nunique()
    ambiguous_orfs = set(counts.loc[counts > 1].index)
    unique_candidates = candidates.loc[~candidates["orf"].isin(ambiguous_orfs)].copy()
    mapped = core_seeds.merge(unique_candidates, on="orf", how="left", validate="one_to_one")
    mapped["mapping_issue"] = "ok"
    mapped.loc[mapped["orf"].isin(ambiguous_orfs), "mapping_issue"] = "ambiguous"
    mapped.loc[
        mapped["string_protein_id"].isna() & ~mapped["orf"].isin(ambiguous_orfs),
        "mapping_issue",
    ] = "unmapped"

    issues = mapped.loc[mapped["mapping_issue"] != "ok"]
    missing_sources = core_seeds.loc[
        core_seeds["source_url"].eq("") | core_seeds["rationale"].eq("")
    ]
    if len(core_seeds) < 3 or not issues.empty or not missing_sources.empty:
        print("Warning: seed table cannot be frozen:")
        if len(core_seeds) < 3:
            print("  Fewer than 3 core seeds")
        if not issues.empty:
            print(issues.to_string(index=False))
        if not missing_sources.empty:
            print("  Missing seed source URLs or rationales")
        return

    mapped_hlj1 = mapped.loc[mapped["orf"] == "YMR161W", "string_protein_id"].iloc[0]
    if mapped_hlj1 != hlj1_id:
        raise ValueError("HLJ1 seed mapping and protein ID mapping disagree")

    design = {
        "string_release": "12.0",
        "taxon_id": 4932,
        "primary_layer": "functional",
        "sensitivity_layer": "physical",
        "primary_score_threshold": 700,
        "sensitivity_thresholds": [400, 900],
        "hlj1_string_id": hlj1_id,
        "core_seed_orfs": sorted(core_seeds["orf"].tolist()),
        "raw_file_sha256": raw_hashes,
        "seed_table_sha256": sha256(seed_path),
    }

    design_path = meta / "string_analysis_design_v12_0.json"
    if design_path.exists():
        old_design = json.loads(design_path.read_text(encoding="utf-8"))
        if old_design != design:
            msg = (
                f"Frozen design mismatch: {design_path}. Raw files or seed table may differ "
                "from the original analysis. Do not claim exact reproduction. "
                "Use --allow-design-mismatch to proceed without overwriting the record."
            )
            if allow_design_mismatch:
                print("WARNING: " + msg)
            else:
                raise ValueError(msg)
        else:
            print("Inputs match the existing frozen design")
    else:
        design_path.write_text(
            json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Frozen design saved: {design_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument("--chunk-size", type=int, default=150_000)
    parser.add_argument(
        "--allow-design-mismatch",
        action="store_true",
        help="Warn instead of failing on frozen input-hash/config mismatches",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be a positive integer")

    root = args.root.resolve()
    raw = root / "data" / "raw" / "string_v12_0"
    processed = root / "data" / "processed" / "string_v12_0"
    meta = root / "data" / "metadata"
    files = {role: raw / filename for role, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error(
            "Missing STRING v12.0 data files. Download and place them under "
            "data/raw/string_v12_0/:\n" + "\n".join(missing)
        )
    processed.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    raw_hashes = {}
    print("1/5: Verifying raw files")
    for role, path in files.items():
        raw_hashes[role] = sha256(path)
        print(f"  {role}: {path.name} ({path.stat().st_size / 1e6:.2f} MB); "
              f"SHA256={raw_hashes[role]}")

    print("2/5: Reading protein metadata and resolving ORFs")
    info, aliases = read_protein_metadata(files)
    id_map, hlj1_id = construct_id_map(info, aliases)
    id_output = processed / "string_id_to_orf.csv"
    id_map.to_csv(id_output, index=False)
    print(f"  Saved {id_output}")

    print("3/5: Preparing functional and physical edge tables")
    edge_tables = {}
    known_ids = set(info["string_protein_id"])
    for layer in ("functional", "physical"):
        edges, report = prepare_edges(files[layer], chunk_size=args.chunk_size)
        endpoints = set(edges["u"]) | set(edges["v"])
        unknown = endpoints - known_ids
        if unknown:
            raise ValueError(f"{layer} edge table contains {len(unknown)} unknown STRING IDs")
        target = processed / f"{layer}_edges_score_ge_400.csv.gz"
        edges.to_csv(target, index=False, compression="gzip")
        edge_tables[layer] = edges
        print(f"  {layer}: {report}")
        print(f"  Saved {target}")

    print("4/5: Building threshold network summary")
    summary = network_summary(edge_tables, info, hlj1_id)
    summary_path = processed / "network_threshold_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"  Saved {summary_path}")

    print("5/5: Checking ERAD seeds and frozen design")
    seed_path = prepare_seeds(meta)
    freeze_design(meta, id_map, hlj1_id, seed_path, raw_hashes, args.allow_design_mismatch)
    print("STRING preprocessing complete")


if __name__ == "__main__":
    main()