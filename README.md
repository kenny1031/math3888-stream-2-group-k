# Group K - HLJ1 Protein Interaction Network Analysis

This repository contains the Python code and Jupyter notebooks used for our MATH3888 group project, *Integrating Graph Theory and Representation Learning for HLJ1-Centred Protein Interaction Network Analysis*.

We analyse the *Saccharomyces cerevisiae* protein association network using Node2Vec, Random Walk with Restart, Louvain community detection, Markov Clustering (MCL), and in-silico deletion analysis. Candidate proteins are evaluated using Gene Ontology (GO) annotations and matched randomisation tests.

This README provides instructions for reproducing the computational results presented in our report.

## 1. Requirements

- Python 3.12 recommended
- Git
- Jupyter Notebook or JupyterLab
- A C++ compiler for the standalone exact betweenness implementation, if required

From the repository root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate` instead.

## 2. Required Datasets

Raw datasets are not included in the GitHub repository and must be downloaded separately.

### 2.1 STRING v12.0

Source: https://version-12-0.string-db.org/cgi/download

Select:

- Organism: *Saccharomyces cerevisiae*
- NCBI taxonomy ID: `4932`
- STRING version: **12.0**

Download the following four files:

| File | Purpose |
|---|---|
| `4932.protein.links.full.v12.0.txt.gz` | Functional protein association network |
| `4932.protein.physical.links.full.v12.0.txt.gz` | Physical interaction network |
| `4932.protein.info.v12.0.txt.gz` | Protein names and descriptions |
| `4932.protein.aliases.v12.0.txt.gz` | STRING ID to gene/ORF mapping |

Use the original unfiltered files rather than applying a confidence threshold during download.

Place all four files in:

```text
data/raw/string_v12_0/
```

The files can remain gzip-compressed (`.gz`).

### 2.2 Gene Ontology (GO)

Download the following files:

| File | Download source |
|---|---|
| `YEAST-mod.gaf.gz` | https://current.geneontology.org/annotations/gaf/YEAST-mod.gaf.gz |
| `go-basic.obo` | https://purl.obolibrary.org/obo/go/go-basic.obo |

Place them in `data/raw/go/` and decompress the annotation file:

```bash
mkdir -p data/raw/go

gzip -dk data/raw/go/YEAST-mod.gaf.gz
```

The expected files are:

```text
data/raw/go/
├── YEAST-mod.gaf
└── go-basic.obo
```

**Reproducibility note:** GO annotations and ontology files are periodically updated. Exact reproduction of the reported GO evaluation requires the same data releases used in the original analysis. Current downloads may produce slightly different results.

### 2.3 Essential proteins

The project also uses:

```text
data/raw/essential_proteins.csv
```

This is the yeast essential-protein reference dataset used for candidate eligibility filtering. Obtain the original dataset supplied for the project and place it at the path above. Preserve its original column structure.

### 2.4 Expected raw-data structure

After downloading, the directory should contain:

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

The metadata files are included in the repository.

## 3. Reproducing the Results

Run all commands from the repository root.

### Step 1: Data preparation

Open Jupyter:

```bash
jupyter lab
```

Run the following notebook from top to bottom:

```text
notebooks/01_string_data_setup.ipynb
```

This notebook prepares STRING data for subsequent network analysis.

The processed data are stored under:

```text
data/processed/string_v12_0/
```

The main report uses an undirected STRING functional-association network with a combined confidence-score threshold of 700.

### Step 2: Main computational pipeline

Run:

```bash
bash scripts/run_pipeline.sh
```

The core Python implementation is organised under `src/`, including:

- `network.py` — Network construction
- `algorithms.py` — Graph algorithms
- `node2vec.py` — Weighted Node2Vec
- `cadtp.py` — Candidate prioritisation
- `go.py` and `evaluation.py` — GO-based evaluation
- `integration.py` — Cross-method evidence integration
- `plotting.py` — Result visualisation
- `pipeline.py` — Pipeline implementation

Configuration parameters are stored in:

```text
configs/default.yaml
```

### Step 3: Additional analyses

Additional notebooks and scripts are provided for analyses developed by individual group members.

| Notebook / script | Analysis |
|---|---|
| `02_baseline_network_structure_and_candidate_features.ipynb` | Baseline network properties and candidate features |
| `Betweenness deletion comparison.ipynb` | Betweenness comparison before and after HLJ1 deletion |
| `Alex Project Week 8.ipynb` | Additional deletion analysis |
| `jadon_hlj1_full_analysis_v12.ipynb` | ERAD-guided screening and candidate analysis |
| `scripts/recompute_deletion_700.py` | Deletion betweenness recomputation |
| `scripts/brandes_exact.cpp` | C++ implementation of exact betweenness |

These provide the additional computational analyses used in the integrated results. They can be run separately where required, using the prepared data and relevant intermediate outputs.

## 4. Output Files

Generated results are organised under `outputs/`.

| Directory | Contents |
|---|---|
| `outputs/rankings/` | Node2Vec and CADTP candidate rankings |
| `outputs/evaluation/` | GO evaluation, matched randomisation and community stability |
| `outputs/integrated/` | Combined evidence across network methods |
| `outputs/figures/` | Figures included in the report |
| `outputs/models/node2vec/` | Trained Node2Vec models |
| `outputs/alex/` | Deletion betweenness analysis |
| `outputs/jadon/` | ERAD-guided analysis |
| `outputs/junxi/` | MCL results |

Key report figures include:

```text
outputs/figures/
├── cross_method_evidence_heatmap.pdf
├── go_precision_at_20.pdf
├── community_resolution_sensitivity.pdf
└── louvain_heldout_membership_heatmap.pdf
```

## 5. Reference Results

The reported analysis uses a confidence threshold of 700.

| Quantity | Reported result |
|---|---|
| Network vertices | 6,600 |
| Network edges | 104,188 |
| HLJ1 component vertices | 5,716 |
| HLJ1 component edges | 104,138 |
| Eligible candidate proteins | 4,439 |
| Node2Vec Cross-track GO Precision@20 | 55% |
| Node2Vec Cross-track matched fold enrichment | 1.84 |
| Node2Vec Cross-track BH-adjusted q-value | 0.012 |
| Held-out Louvain track agreement | 93.85% |

Fixed random seeds are used in the relevant experiments. Small numerical differences may arise from software versions, stochastic computation or updated annotation datasets.

## 6. Tests

Unit tests are located in `tests/` and can be run using:

```bash
python -m pytest -q
```

## 7. Notes

- The analysis uses **STRING v12.0**, not the latest STRING release.
- Raw datasets are excluded from GitHub. Intermediate files, trained models and generated figures can be regenerated or inspected as applicable.
- Reproduction of exact GO statistics requires the original annotation versions.