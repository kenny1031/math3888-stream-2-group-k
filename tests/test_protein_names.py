"""
Regression tests for yeast protein identifiers.

Verify consistency among:
    - STRING protein IDs
    - Systematic ORF identifiers
    - Preferred gene names
    - CADTP outputs
    - Network figure evidence
"""

import pandas as pd
import pytest

from src.config import load_config


# Verified against the project's STRING mapping file.
EXPECTED_GENES = {
    # Target and readouts
    "HLJ1": "YMR161W",
    "KAR2": "YJL034W",
    "YDJ1": "YNL064C",

    # Local candidates
    "SSE1": "YPL106C",
    "APJ1": "YNL077W",
    "JJJ2": "YJL162C",
    "ERJ5": "YFR041C",
    "JID1": "YPR061C",
    "SSA4": "YER103W",
    "SSA3": "YBL075C",
    "ECM10": "YEL030W",
    "MDJ1": "YFL016C",
    "SSA1": "YAL005C",

    # Cross candidates
    "SSM4": "YIL030C",
    "UFD2": "YDL190C",
    "FES1": "YBR101C",
    "UBX2": "YML013W",
    "SIL1": "YOL031C",
    "UBR1": "YGR184C",
    "NPL4": "YBR170C",
    "YOS9": "YDR057W",
    "DFM1": "YDR411C",
    "USA1": "YML029W",

    # Boundary candidates
    "TIM21": "YGR033C",
    "ISU1": "YPL135W",
    "PAM17": "YKR065C",
    "MGR2": "YPL098C",
    "TOM70": "YNL121C",
    "NFU1": "YKL040C",
    "ISA2": "YPR067W",
    "ISA1": "YLL027W",
    "HSP12": "YFL014W",
    "TOM71": "YHR117W",

    # Path intermediates
    "SWA2": "YDR320C",

    # ERAD references
    "HRD1": "YOL013C",
    "DER1": "YBR201W",
    "HRD3": "YLR207W",
    "UBC7": "YMR022W",
    "CUE1": "YMR264W",
}


@pytest.fixture(scope="module")
def registry():
    cfg = load_config()

    mapping = pd.read_csv(
        cfg.path("protein_mapping"),
        dtype=str,
    )

    assert mapping["string_protein_id"].is_unique
    assert mapping["preferred_name"].is_unique

    return mapping.set_index("string_protein_id")


def test_known_gene_orf_pairs(registry):
    """Check all important curated gene identifiers."""

    for gene, orf in EXPECTED_GENES.items():
        protein_id = f"4932.{orf}"

        assert protein_id in registry.index, gene

        actual = registry.loc[protein_id]

        assert actual["orf"] == orf
        assert actual["preferred_name"] == gene, (
            f"{protein_id}: expected {gene}, "
            f"got {actual['preferred_name']}"
        )


def test_baseline_identifiers(registry):
    """Notebook 02 must agree with the mapping registry."""

    cfg = load_config()

    baseline = pd.read_csv(
        cfg.path("baseline_features"),
        index_col="string_protein_id",
        dtype=str,
    )

    assert set(baseline.index) == set(registry.index)

    expected = registry.reindex(baseline.index)

    for column in ["orf", "preferred_name"]:
        actual_values = baseline[column].fillna("")
        expected_values = expected[column].fillna("")

        assert (
            actual_values.to_numpy()
            == expected_values.to_numpy()
        ).all(), f"Mismatch in baseline {column}"


def test_cadtp_ranking_identifiers(registry):
    """All saved CADTP candidates must have correct names."""

    cfg = load_config()

    path = (
        cfg.path("rankings")
        / "cadtp_training_refactor.csv"
    )

    ranking = pd.read_csv(
        path,
        index_col="string_id",
        dtype=str,
    )

    assert len(ranking) == 4439
    assert ranking.index.is_unique

    expected = registry.reindex(ranking.index)

    assert expected["preferred_name"].notna().all()

    assert (
        ranking["preferred_name"].to_numpy()
        == expected["preferred_name"].to_numpy()
    ).all()

    assert (
        ranking["orf"].to_numpy()
        == expected["orf"].to_numpy()
    ).all()


def test_cross_network_uses_usa1(registry):
    """
    Prevent the previous USA1 / DFM1 plotting mistake.
    """

    cfg = load_config()

    evidence = pd.read_csv(
        cfg.path("evaluation")
        / "hlj1_cross_path_evidence.csv",
        dtype=str,
    )

    expected_genes = {
        "SSM4", "UBX2", "YOS9", "USA1"
    }

    assert set(evidence["candidate_gene"]) == expected_genes
    assert "DFM1" not in set(evidence["candidate_gene"])

    for row in evidence.itertuples(index=False):
        assert row.candidate_id in registry.index

        actual_gene = registry.loc[
            row.candidate_id, "preferred_name"
        ]

        assert row.candidate_gene == actual_gene

    usa1 = evidence.loc[
        evidence["candidate_gene"] == "USA1"
    ]

    assert len(usa1) == 1

    assert usa1.iloc[0]["candidate_id"] == "4932.YML029W"

    assert usa1.iloc[0]["path_genes"] == (
        "HLJ1 -> KAR2 -> USA1"
    )
