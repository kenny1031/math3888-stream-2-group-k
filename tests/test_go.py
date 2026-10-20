"""
Tests for GO annotation processing.
Run:
    python -m pytest tests/test_go.py -v
"""
import gzip
from src.go import load_filtered_gaf, build_go_labels


def make_gaf_row(
    sgd_id,
    symbol,
    qualifier,
    go_id,
    evidence,
    aspect,
    synonyms="",
):
    """Build a minimal GAF-style record."""
    fields = [""] * 15
    fields[1] = sgd_id
    fields[2] = symbol
    fields[3] = qualifier
    fields[4] = go_id
    fields[6] = evidence
    fields[8] = aspect
    fields[10] = synonyms

    return "\t".join(fields)


def test_gaf_filtering(tmp_path):
    path = tmp_path / "test.gaf.gz"
    rows = [
        make_gaf_row(
            "S1", "YPL106C", "",
            "GO:0000002", "EXP", "P"
        ),
        make_gaf_row(
            "S2", "UNKNOWN", "",
            "GO:0000002", "EXP", "P",
            "YBR201W"
        ),
        make_gaf_row(
            "S3", "YLR207W", "NOT",
            "GO:0000002", "EXP", "P"
        ),
        make_gaf_row(
            "S4", "YMR022W", "",
            "GO:0000002", "IEA", "P"
        ),
        make_gaf_row(
            "S5", "YJL034W", "",
            "GO:0000002", "EXP", "F"
        ),
    ]

    with gzip.open(path, "wt") as f:
        f.write("!gaf-version: 2.2\n")
        f.write("\n".join(rows) + "\n")

    gaf = load_filtered_gaf(path)

    assert len(gaf) == 2
    assert set(gaf["orf"]) == {"YPL106C", "YBR201W"}


def test_go_descendant_propagation(tmp_path):
    obo_path = tmp_path / "go.obo"
    obo_path.write_text(
        "format-version: 1.2\n\n"
        "[Term]\n"
        "id: GO:0000001\n"
        "name: parent\n"
        "namespace: biological_process\n\n"
        "[Term]\n"
        "id: GO:0000002\n"
        "name: child\n"
        "namespace: biological_process\n"
        "is_a: GO:0000001 ! parent\n"
    )

    import pandas as pd

    gaf = pd.DataFrame({
        "orf": ["YPL106C"],
        "go_id": ["GO:0000002"],
    })

    labels = build_go_labels(
        gaf=gaf,
        obo_path=obo_path,
        all_orfs=["YPL106C", "YBR201W"],
        categories={"protein_folding": "GO:0000001"},
    )

    # The child term inherits the parent category.
    assert bool(labels.loc["YPL106C", "protein_folding"])
    assert bool(labels.loc["YPL106C", "proteostasis_any"])
    assert not bool(labels.loc["YBR201W", "has_BP_annotation"])
    assert not bool(labels.loc["YBR201W", "proteostasis_any"])