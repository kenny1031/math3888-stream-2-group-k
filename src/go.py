"""
GO biological-process annotation processing.
Reproduces Notebook 06:
    - GAF filtering and systematic ORF mapping
    - GO descendant expansion using goatools
    - Binary proteostasis-related labels

No downloads or ranking algorithms are performed here.
"""
import hashlib
from pathlib import Path
from goatools.obo_parser import GODag
import pandas as pd


ORF_PATTERN = r"(Y[A-P][LR]\d{3}[CW](?:-[A-Z])?)"
GAF_COLUMNS = [
    "sgd_id",
    "symbol",
    "qualifier",
    "go_id",
    "evidence",
    "aspect",
    "synonyms",
]

# Match Notebook 06 output column names.
CATEGORY_NAMES = {
    "protein_folding": "protein_folding",
    "erad": "ERAD",
    "er_stress": "ER_stress",
    "unfolded_protein_response": "unfolded_protein_response",
}

LABEL_COLUMNS = [
    "has_BP_annotation",
    "protein_folding",
    "ERAD",
    "ER_stress",
    "unfolded_protein_response",
    "proteostasis_any",
]


# ============================================================
# 1. Resolve local GAF file
# ============================================================

def resolve_gaf_path(path: str | Path) -> Path:
    """
    Locate the existing GAF file.

    Notebook 06 may have downloaded gzipped bytes into
    YEAST-mod.gaf instead of YEAST-mod.gaf.gz.
    Support either filename.
    """

    path = Path(path)
    if path.is_file():
        return path

    alternatives = []
    if path.suffix == ".gz":
        alternatives.append(path.with_suffix(""))
    else:
        alternatives.append(Path(str(path) + ".gz"))

    for alternative in alternatives:
        if alternative.is_file():
            return alternative

    raise FileNotFoundError(f"GO annotation file not found: {path}")


# ============================================================
# 2. Load and filter GAF
# ============================================================

def load_filtered_gaf(path: str | Path) -> pd.DataFrame:
    """
    Reproduce Notebook 06's GAF filtering.
    Keep:
        - Biological Process annotations (aspect P)
        - Experimental/curated evidence other than IEA, ND
        - Non-negative annotations
        - Records with systematic yeast ORF identifiers
    """
    path = resolve_gaf_path(path)
    # Detect compression from file contents, not suffix.
    with path.open("rb") as f:
        is_gzip = f.read(2) == b"\x1f\x8b"

    gaf = pd.read_csv(
        path,
        sep="\t",
        comment="!",
        header=None,
        dtype=str,
        usecols=[1, 2, 3, 4, 6, 8, 10],
        low_memory=False,
        compression="gzip" if is_gzip else None,
    )

    gaf.columns = GAF_COLUMNS

    # Exclude negative annotations.
    is_negative = (
        gaf["qualifier"]
        .fillna("")
        .apply(lambda x: "NOT" in x.split("|"))
    )

    gaf = gaf.loc[
        ~is_negative
        & gaf["aspect"].eq("P")
        & ~gaf["evidence"].isin(["IEA", "ND"])
    ].copy()

    # Match identifiers from gene symbol and synonyms.
    identifiers = gaf["symbol"].fillna("") + "|" + gaf["synonyms"].fillna("")

    gaf["orf"] = identifiers.str.extract(ORF_PATTERN, expand=False)
    gaf = gaf.dropna(subset=["orf"]).copy()

    return gaf.reset_index(drop=True)


# ============================================================
# 3. Build GO labels
# ============================================================

def build_go_labels(
    gaf: pd.DataFrame,
    obo_path: str | Path,
    all_orfs,
    categories: dict[str, str],
) -> pd.DataFrame:
    """
    Build biological-process labels for systematic ORFs.

    GO propagation:
        Each category contains its GO term and all
        descendant terms under the ontology's is_a
        relationships, matching Notebook 06.

    Missing annotations are encoded as False.
    Annotation coverage is recorded separately.
    """
    obo_path = Path(obo_path)

    if not obo_path.is_file():
        raise FileNotFoundError(f"GO ontology file not found: {obo_path}")

    required = {"orf", "go_id"}

    if not required.issubset(gaf.columns):
        raise ValueError(f"Missing GAF columns: {required - set(gaf.columns)}")

    go = GODag(str(obo_path))

    term_sets = {}

    for config_name, go_id in categories.items():
        if config_name not in CATEGORY_NAMES:
            raise ValueError(f"Unknown GO category: {config_name}")

        if go_id not in go:
            raise ValueError(f"GO term not found in ontology: {go_id}")

        output_name = CATEGORY_NAMES[config_name]
        term_sets[output_name] = {go_id} | go[go_id].get_all_children()

    go_by_orf = (
        gaf.groupby("orf")["go_id"]
        .agg(set)
        .to_dict()
    )

    # Preserve the original ORF order from Notebook 02.
    orfs = (
        pd.Series(all_orfs, dtype="string")
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    labels = pd.DataFrame(index=pd.Index(orfs, name="orf"))

    labels["has_BP_annotation"] = [
        orf in go_by_orf
        for orf in labels.index
    ]

    for name, terms in term_sets.items():
        labels[name] = [
            bool(go_by_orf.get(orf, set()) & terms)
            for orf in labels.index
        ]

    biological_categories = list(term_sets)

    labels["proteostasis_any"] = labels[biological_categories].any(axis=1)

    # Return only the categories requested by the caller.
    # The full experiment configuration still produces
    # all six original Notebook 06 label columns.
    output_columns = [
        "has_BP_annotation",
        *biological_categories,
        "proteostasis_any",
    ]

    return labels[output_columns]



# ============================================================
# 4. Data provenance
# ============================================================

def sha256_file(path: str | Path) -> str:
    """Compute SHA256 for a local source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()
