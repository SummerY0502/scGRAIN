# scGRAIN

scGRAIN is a multilevel gene regulatory network inference framework for single-cell multiomic data in rice.

This repository contains the code supporting the analyses in the manuscript:

**scGRAIN infers multilevel gene regulatory networks from single-cell multiomic data in rice**

## Main scripts

### `TF-Program.py`
TF-program association inference and gene-level joint-mask scoring.

### `TF_Gene.py`
TF-gene link inference using an edge-aware bipartite graph neural network.

### `program50_fci_causal_discovery.py`
FCI-based inference of gene-gene dependency structures within selected gene programs.

## Installation

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

## Data

Large input and intermediate files are not stored directly in this repository.

See `data/README.md` for the required directory structure and data placement.

## Running the analyses

```bash
python TF-Program.py
python TF_Gene.py
python program50_fci_causal_discovery.py
```

## Citation

Citation information will be updated upon publication.
