# 🌾 scGRAIN: Multilevel Gene Regulatory Network Inference from Single-Cell Multiomic Data in Rice

> **scGRAIN** is a computational framework for inferring multilevel regulatory networks from single-cell RNA-seq and chromatin accessibility data in rice.

This repository contains the core implementation associated with the manuscript:

**scGRAIN infers multilevel gene regulatory networks from single-cell multiomic data in rice**

---

## Overview

Cell-type-specific transcriptional regulation involves regulatory relationships at multiple levels, including transcription factor (TF)–gene interactions, coordinated gene programs, and dependencies among genes.

scGRAIN integrates single-cell transcriptomic and chromatin accessibility information to connect these complementary levels of regulatory organization. The framework:

1. infers **TF–program associations** using multiomic graph representation learning and causal subgraph learning;
2. derives **gene-level joint-mask scores** to prioritize program genes contributing strongly to predicted TF–program associations;
3. infers **TF–gene links** using an edge-aware bipartite graph neural network; and
4. reconstructs **program-internal gene–gene dependency structures** using Fast Causal Inference (FCI).

Together, these components provide a multilevel representation connecting TFs, program genes, and gene–gene dependency structures.

---

## Framework overview

scGRAIN integrates single-cell RNA sequencing and chromatin accessibility data to infer multilevel regulatory relationships.

<p align="center">
  <img src="docs/Fig1.png" width="95%">
</p>

<p align="center">
  <b>Figure 1. Overview of the scGRAIN framework.</b>
</p>

---

## Key features

- **Multilevel regulatory inference**  
  Connects TF–program, TF–gene, and gene–gene regulatory evidence within a common analytical framework.

- **Single-cell multiomic integration**  
  Incorporates RNA expression and chromatin accessibility information into regulatory network inference.

- **Gene-program representation**  
  Uses gene programs comprising coordinately expressed genes to characterize transcriptional organization beyond individual genes.

- **Joint-mask gene prioritization**  
  Derives gene-level joint-mask scores to prioritize program genes contributing to predicted TF–program associations.

- **TF–gene link inference**  
  Uses an edge-aware bipartite GNN with TF/gene RNA profiles, gene-level chromatin accessibility, and TF–gene joint-expression histograms.

- **Gene–gene dependency analysis**  
  Uses FCI with Fisher-Z conditional-independence testing to infer candidate dependency structures within fixed program gene sets.

- **Structural sensitivity assessment**  
  Supports resampling-based FCI stability analysis and PC-stable comparison.

---

## Repository structure

```text
scGRAIN/
├── TF-Program.py
├── TF_Gene.py
├── program50_fci_causal_discovery.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.md
│
└── docs/
    └── Fig1.png
```

### Main scripts

#### `TF-Program.py`

Core TF–program analysis.

This script implements the multiomic evidence-subgraph framework used for:

- TF–program association inference;
- graph-based representation learning;
- gene-level edge-mask estimation;
- joint-mask scoring of program genes; and
- reproduction of TF–program predictions from the trained model.

#### `TF_Gene.py`

TF–gene link inference using an edge-aware bipartite graph neural network.

The model integrates:

- TF RNA profiles;
- gene RNA profiles;
- TF and gene chromatin accessibility profiles; and
- TF–gene joint-expression histograms.

Motif/TSS evidence is used as the supervision source for TF–gene link recovery rather than as a predictor input.

#### `program50_fci_causal_discovery.py`

Program-specific gene–gene dependency analysis using FCI.

For a selected gene program, the script:

- uses the fixed program gene set defined by the TF–program workflow;
- constructs RNA pseudo-metacells using global cNMF usage profiles;
- performs FCI with Fisher-Z conditional-independence testing;
- evaluates sensitivity across conditional-independence thresholds;
- assesses adjacency and orientation stability by repeated subsampling; and
- optionally compares FCI structure with PC-stable.

FCI/PAG edges should be interpreted as candidate dependency or causal-structure relationships under the assumptions of FCI, rather than as experimental evidence of direct gene regulation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SummerY0502/scGRAIN.git
cd scGRAIN
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- NumPy
- pandas
- SciPy
- scikit-learn
- PyTorch
- joblib
- causal-learn

---

## Data

Large input and intermediate files are not stored directly in this GitHub repository because of file-size limits.

See:

```text
data/README.md
```

for the required directory structure.

The expected local organization is:

```text
data/
├── output_top50_bg_centered_Parenchyma_cell/
├── tf_program_ces_top50_bg_centered_cov_fix13/
├── TF_program_result50/
├── RNA/
├── atacRNA/
├── cell_metadata.csv
├── RNA_scRNA_Parenchyma_cell.spectra.k_100.dt_0_5.consensus.txt
├── RNA_scRNA_Parenchyma_cell.usages.k_100.dt_0_5.consensus.txt
└── trans.txt
```

A permanent archive and DOI for the required data files will be provided through Zenodo.

---

## Running the analyses

After placing the required files under `data/`, the three core analyses can be run independently.

### 1. TF–program association inference

```bash
python TF-Program.py
```

Infers TF–program association scores and derives gene-level joint-mask scores for prioritizing program genes.

### 2. TF–gene link inference

```bash
python TF_Gene.py
```

Infers TF–gene link scores and generates binary predictions for candidate TF–gene regulatory relationships.

### 3. Program-internal gene–gene dependency inference

```bash
python program50_fci_causal_discovery.py
```

Infers gene–gene dependency structures among fixed program genes using FCI, with subsampling-based stability analysis.
---

## Reproducibility

The repository is designed so that large data files can be downloaded separately and placed under `data/` without modifying the repository structure.

For reproducible use:

1. clone this repository;
2. install the dependencies in `requirements.txt`;
3. download the associated data archive;
4. extract the files under `data/`; and
5. run the corresponding analysis script.

A versioned GitHub release and permanent Zenodo archive will be provided for the manuscript-associated release.

---

## Citation

If you use scGRAIN in your research, please cite the associated manuscript.

Citation information will be updated upon publication.

---

## Contact

For questions, reproducibility issues, or bug reports, please open an issue in this repository.
