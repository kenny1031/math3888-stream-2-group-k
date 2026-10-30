# MATH3888 Group K — HLJ1 Protein Network Analysis

Code accompanying *Integrating Graph Theory and Representation Learning for HLJ1-Centred Protein Interaction Network Analysis*.

The analysis uses STRING v12.0 yeast protein networks, random walks and Node2Vec, Louvain and Markov Clustering (MCL), HLJ1 in-silico deletion, and Gene Ontology (GO) evaluation. **Raw datasets are not stored in this repository.** This guide explains how to restore the inputs and run the code.

## 1. Environment

Python **3.12** is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

If running optional notebooks, use your preferred Jupyter environment (install `jupyterlab` if it is not already in `requirements.txt`). Run commands below from the **repository root**.

## 2. Download the raw data

### A. STRING v12.0 — required for preprocessing

Download from the [STRING v12.0 archived download page](https://version-12-0.string-db.org/cgi/download), selecting **Saccharomyces cerevisiae** (NCBI taxon **4932**). Download the **full** links files, not the smaller basic-links files, and retain the original filenames:

| File | Data |
| --- | --- |
| `4932.protein.links.full.v12.0.txt.gz` | Functional protein associations |
| `4932.protein.physical.links.full.v12.0.txt.gz` | Physical interaction associations |
| `4932.protein.info.v12.0.txt.gz` | Protein names/metadata |
| `4932.protein.aliases.v12.0.txt.gz` | Identifiers and aliases |

Place all four compressed files under `data/raw/string_v12_0/`. **Do not decompress these four `.gz` files.** The script reads them directly. Use **version 12.0** rather than the current default STRING release.

### B. Gene Ontology — required for GO evaluation

Download:

- [`YEAST-mod.gaf.gz`](https://current.geneontology.org/annotations/gaf/YEAST-mod.gaf.gz) — yeast gene annotations;
- [`go-basic.obo`](https://purl.obolibrary.org/obo/go/go-basic.obo) — GO ontology.

Place both downloads in `data/raw/go/` and decompress the annotations to `YEAST-mod.gaf`:

```bash
mkdir -p data/raw/go
gzip -dk data/raw/go/YEAST-mod.gaf.gz
```

The analysis expects `data/raw/go/YEAST-mod.gaf` and `data/raw/go/go-basic.obo`. GO annotations are updated over time; exact numerical reproduction requires the **same annotation release** used for the report.

### C. Essential-protein reference — required for candidate filtering

Obtain the original course-provided `essential_proteins.csv` and save it as:

```text
data/raw/essential_proteins.csv
```

Keep the file's original structure. If this input is not distributed with the course submission, an accessible copy or source link must be supplied to the marker; it cannot be reconstructed from STRING alone.

### Expected input layout

```text
data/
├── metadata/
│   ├── erad_seeds.csv
│   └── string_analysis_design_v12_0.json
└── raw/
    ├── essential_proteins.csv
    ├── go/
    │   ├── YEAST-mod.gaf
    │   └── go-basic.obo
    └── string_v12_0/
        ├── 4932.protein.aliases.v12.0.txt.gz
        ├── 4932.protein.info.v12.0.txt.gz
        ├── 4932.protein.links.full.v12.0.txt.gz
        └── 4932.protein.physical.links.full.v12.0.txt.gz
```

The two `data/metadata/` files are normally tracked in the repository. The preprocessing script creates an ERAD seed template and frozen design record if they are missing.

## 3. Run the analysis

**Step 1 — Prepare STRING data**:

```bash
python scripts/prepare_string_data.py
```

This step:

1. validates the four STRING files and calculates their SHA-256 hashes;
2. resolves STRING protein IDs to yeast ORF identifiers and locates HLJ1 (`YMR161W`);
3. removes self-loops and combines reciprocal edges (checking for conflicting scores);
4. generates summary statistics at STRING confidence thresholds **400, 700 and 900**;
5. checks the ERAD seed table and frozen analysis design.

Outputs:

```text
data/processed/string_v12_0/
├── functional_edges_score_ge_400.csv.gz
├── physical_edges_score_ge_400.csv.gz
├── string_id_to_orf.csv
└── network_threshold_summary.csv
```

The main study uses the **functional network at combined score ≥ 700**; the score ≥ 400 edge tables allow threshold comparisons without downloading the data again.

If the script reports **`Frozen design mismatch`**, the downloaded STRING files and/or ERAD seed table differ from the recorded experiment. Check the versions and input files first. To inspect outputs despite a mismatch, run `python scripts/prepare_string_data.py --allow-design-mismatch`. This **does not overwrite** the frozen design and does **not** establish exact reproduction.

**Step 2 — Run the main project pipeline:**

```bash
bash scripts/run_pipeline.sh
```

The implementation is in `src/` (`network.py`, `node2vec.py`, `algorithms.py`, `cadtp.py`, `go.py`, `evaluation.py`, `integration.py`, `plotting.py`, and `pipeline.py`), with parameters in `configs/default.yaml`.

**Step 3 — Supplementary analyses:** The individual notebooks in `notebooks/` cover baseline network diagnostics, betweenness deletion and ERAD-guided screening. The repository also includes `scripts/recompute_deletion_700.py` and `scripts/brandes_exact.cpp` for deletion/betweenness computations. These may be needed for group-member-specific figures or outputs not included in the main pipeline; run them using their corresponding source files and prepared data.

## 4. Outputs and reference results

Results are organised as follows:

| Directory | Contents |
| --- | --- |
| `outputs/rankings/` | Node2Vec and CADTP rankings |
| `outputs/evaluation/` | GO precision, matched randomisations and Louvain stability |
| `outputs/integrated/` | Cross-method candidate evidence |
| `outputs/figures/` | Figures presented in the report |
| `outputs/models/node2vec/` | Trained Node2Vec models |
| `outputs/alex/`, `outputs/jadon/`, `outputs/junxi/` | Supplementary group-member analyses |

Selected values reported in the final report (STRING v12.0; functional layer, score ≥ 700):

| Metric | Reference value |
| --- | ---: |
| Total network nodes / edges | 6,600 / 104,188 |
| HLJ1 component nodes / edges | 5,716 / 104,138 |
| Eligible candidate proteins | 4,439 |
| Node2Vec Cross-track GO Precision@20 | 55% |
| Node2Vec Cross-track matched fold enrichment | 1.84 |
| Node2Vec Cross-track BH-adjusted `q` | 0.012 |
| Held-out Louvain track agreement | 93.85% |

Differences may arise if GO files, seeds, packages or input data differ. The source code is provided for reproducibility checks; the analysis produces network-based *candidate hypotheses*, not confirmed biological compensation following HLJ1 deletion.

## 5. Tests

```bash
python -m pytest -q
```
