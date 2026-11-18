#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


# =============================================================================
# CONFIG -- edit paths/settings here
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    # Existing feature/edge output directory. The directory name can be legacy;
    # this script reads ONLY the pure TF/gene files listed in the docstring.
    "prepared_dir": str(BASE_DIR / "data/output_top50_bg_centered_Parenchyma_cell"),

    # Existing 0label output directory. Only TF_gene_matrix_lambda*.csv is read.
    "label_dir": str(BASE_DIR / "data/TF_program_result50"),
    "lambda_dist": 1000,

    # Same TF ID mapping used upstream. Set None if no mapping is required.
    "id_map_file": str(BASE_DIR / "data/trans.txt"),

    # Pure TF-gene classifier output.
    "out_dir": str(BASE_DIR / "tf_gene_binary_edge_gnn_logs_Parenchyma_cell"),

    # -------------------------------------------------------------------------
    # Split design
    # -------------------------------------------------------------------------
    # pair          : held-out TF-gene combinations; TF/gene identities may be
    #                 seen in training through other pairs.
    # leave_tf_out  : entire TF identities held out.
    # leave_gene_out: entire gene identities held out.
    "split_mode": "pair",
    "n_splits": 10,
    "split_id": 1,
    "train_fraction_within_trainval": 0.70,

    # -------------------------------------------------------------------------
    # Node-profile compression
    # -------------------------------------------------------------------------
    # PCA is unsupervised and fit only on node identities appearing in TRAIN.
    # Set 0 to disable. Recommended ablation later: 64 / 128 / 256 / 0.
    "pca_dim_per_modality": 64,

    # -------------------------------------------------------------------------
    # Model/training
    # -------------------------------------------------------------------------
    "hidden_dim": 128,
    "edge_dim": 64,
    "dropout": 0.20,

    # Edge-aware bipartite GNN. Setting use_gnn=False gives a clean MLP-only
    # ablation while keeping the same node/pair encoders and decoder.
    "use_gnn": True,
    "gnn_max_neighbors": 16,
    "gnn_attention_hidden_dim": 64,
    # During training, neighbors are randomly sampled when degree > max_neighbors;
    # evaluation uses a fixed, seed-shuffled adjacency order.
    "gnn_random_neighbor_sampling_train": True,

    # GNN batches carry query-pair histograms plus two endpoint neighborhoods,
    # so use a smaller batch than the pairwise MLP version.
    "batch_size": 512,
    "prediction_batch_size": 1024,
    "max_epochs": 500,
    "patience": 40,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "gradient_clip_val": 1.0,

    # Loss: "weighted_bce" (default/recommended first) or "focal".
    "loss_mode": "weighted_bce",
    "focal_gamma": 2.0,

    # Train-only class weighting:
    # base = N_negative / N_positive, then multiply by this factor.
    # Keep 1.0 for the first optimized-threshold run. If recall remains too low,
    # controlled ablation values such as 1.25 / 1.5 can be tried on validation.
    "use_pos_weight": True,
    "pos_weight_multiplier": 1.0,
    "max_pos_weight": 50.0,

    # Early stopping/checkpoint ranking metric: average_precision or roc_auc.
    "early_stop_metric": "average_precision",

    # -------------------------------------------------------------------------
    # Recall-oriented threshold selection -- VALIDATION ONLY
    # -------------------------------------------------------------------------
    # target_recall: among thresholds reaching target recall, maximize precision.
    # mcc / f1 / f2: maximize the named validation metric.
    "threshold_strategy": "target_recall",
    "target_recall": 0.70,
    "f_beta": 2.0,

    # A compact validation operating-point table will also be exported for these
    # recall targets, whether or not they are used as the primary threshold.
    "report_recall_targets": [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90],

    "device": "auto",  # auto / cuda / cpu
    "seed": 42,
}


# =============================================================================
# Reproducibility / I/O
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    if name == "cuda":
        torch.set_float32_matmul_precision("high")
        print("[DEVICE]", torch.cuda.get_device_name(0))
    else:
        print("[DEVICE] CPU")
    return torch.device(name)


def ensure_exists(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def read_names(path: str | Path) -> pd.Index:
    x = pd.read_csv(path, sep="\t", header=None, dtype=str)
    return pd.Index(x.iloc[:, 0].astype(str).values)


def read_table_auto(path: str | Path, index_col=0) -> pd.DataFrame:
    p = str(path)
    df = pd.read_csv(p, sep="\t", index_col=index_col)
    if df.shape[1] <= 1:
        df = pd.read_csv(p, sep=",", index_col=index_col)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


# =============================================================================
# TF ID mapping with binary-OR semantics
# =============================================================================


def load_id_map(path: Optional[str | Path]) -> Dict[str, List[str]]:
    if path is None or str(path).lower() in {"", "none", "null"}:
        return {}

    p = ensure_exists(path, "ID map")
    df = pd.read_csv(p, sep="\t", header=None, dtype=str)
    if df.shape[1] < 2:
        raise ValueError(f"ID map must have >=2 tab-separated columns: {p}")

    mapping: Dict[str, List[str]] = {}
    for old, new in zip(df.iloc[:, 0], df.iloc[:, 1]):
        old = str(old).strip()
        new = str(new).strip()
        if old == "" or old.lower() == "nan":
            continue
        if new == "" or new.lower() in {"none", "nan", "null", "na"}:
            mapping[old] = []
            continue

        vals: List[str] = []
        for x in new.split(","):
            x = x.strip()
            if x and x.lower() not in {"none", "nan", "null", "na"}:
                vals.append(x)
        mapping[old] = vals
    return mapping


def map_evidence_rows_binary_or(
    evidence: pd.DataFrame,
    id_map: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map old TF IDs to final TF IDs.

    If several old IDs map to one new TF, aggregate by MAX/OR. This is required
    for a presence/absence label: any positive old-ID evidence makes the mapped
    TF-gene pair positive. Continuous E is max-aggregated only for audit.
    """
    rows = []
    audit = []

    for old_id, row in evidence.iterrows():
        old = str(old_id)
        mapped_ids = id_map.get(old, [old])
        if len(mapped_ids) == 0:
            audit.append({"old_tf": old, "new_tf": "", "status": "dropped"})
            continue

        for new_id in mapped_ids:
            s = row.copy()
            s.name = str(new_id)
            rows.append(s)
            audit.append({
                "old_tf": old,
                "new_tf": str(new_id),
                "status": "mapped" if str(new_id) != old else "unchanged",
            })

    if not rows:
        raise ValueError("All TF rows were dropped during ID mapping.")

    mapped = pd.DataFrame(rows)
    mapped.index = mapped.index.astype(str)
    mapped = mapped.groupby(mapped.index, sort=False).max()

    audit_df = pd.DataFrame(audit)
    if len(audit_df):
        nonempty = audit_df[audit_df["new_tf"].astype(str) != ""]
        mult = nonempty.groupby("new_tf")["old_tf"].nunique()
        audit_df["n_old_ids_to_new_tf"] = (
            audit_df["new_tf"].map(mult).fillna(0).astype(int)
        )

    return mapped, audit_df


# =============================================================================
# Binary TF-gene supervision
# =============================================================================


def build_binary_tf_gene_labels(
    label_dir: Path,
    tfs: pd.Index,
    genes: pd.Index,
    out_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build Z_tg = 1[E_tg > 0] aligned to prepared TF/gene IDs."""
    lam = int(CONFIG["lambda_dist"])
    raw_e_path = ensure_exists(
        label_dir / f"TF_gene_matrix_lambda{lam}.csv",
        "raw TF-gene motif/TSS evidence matrix",
    )

    E_old = pd.read_csv(raw_e_path, index_col=0)
    E_old.index = E_old.index.astype(str)
    E_old.columns = E_old.columns.astype(str)
    E_old = E_old.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if np.nanmin(E_old.to_numpy(dtype=float)) < -1e-10:
        raise ValueError(
            "Raw TF-gene evidence contains negative values, but E_tg should be a "
            "sum of positive motif/TSS distance-decay contributions. Check label input."
        )
    # Numerical safety only.
    E_old = E_old.clip(lower=0.0)

    # A gene absent from the evidence matrix has zero observed valid hits.
    E_old = E_old.reindex(columns=genes, fill_value=0.0).astype(float)

    id_map = load_id_map(CONFIG.get("id_map_file"))
    E_mapped, mapping_audit = map_evidence_rows_binary_or(E_old, id_map)
    mapping_audit.to_csv(out_dir / "00_tf_id_mapping_audit.csv", index=False)

    missing_tfs = tfs.difference(E_mapped.index)
    if len(missing_tfs):
        raise ValueError(
            "Prepared TFs are absent from mapped motif/TSS evidence. To avoid "
            "silently treating an ID-mapping error as all-negative labels, this "
            f"pipeline stops. Examples: {missing_tfs[:20].tolist()}"
        )

    E_df = E_mapped.reindex(index=tfs, columns=genes)
    if E_df.isna().any().any():
        raise ValueError("NaN after aligning motif/TSS evidence to TF/gene IDs.")

    E = E_df.to_numpy(dtype=np.float32)
    Z = (E > 0).astype(np.int8)

    pd.DataFrame(Z, index=tfs, columns=genes).to_csv(
        out_dir / "01_TF_gene_binary_motif_TSS_label.tsv", sep="\t"
    )
    E_df.to_csv(out_dir / "02_TF_gene_raw_E_audit_only.tsv", sep="\t")

    pos = int(Z.sum())
    total = int(Z.size)
    summary = {
        "label_formula": "Z_tg = 1[E_tg > 0]",
        "positive_definition": ">=1 valid motif/TSS hit under the existing 0label filters",
        "negative_definition": "0 observed valid motif/TSS hits under the existing 0label filters",
        "raw_E_definition": "sum_valid_hits exp(-abs(distance_to_TSS)/lambda)",
        "raw_E_file": str(raw_e_path),
        "lambda_dist": lam,
        "distance_magnitude_used_as_binary_target_strength": False,
        "motif_TSS_distance_used_as_predictor": False,
        "binary_ID_mapping_aggregation": "OR/max",
        "n_tf": int(Z.shape[0]),
        "n_gene": int(Z.shape[1]),
        "n_pairs": total,
        "n_positive": pos,
        "n_negative": total - pos,
        "positive_fraction": float(pos / total) if total else float("nan"),
        "raw_E_positive_min": float(E[E > 0].min()) if pos else float("nan"),
        "raw_E_positive_median": float(np.median(E[E > 0])) if pos else float("nan"),
        "raw_E_positive_max": float(E.max()) if pos else float("nan"),
    }
    with open(out_dir / "03_binary_label_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return Z, E, summary


# =============================================================================
# Load PURE TF/gene predictors
# =============================================================================


def load_prepared_data(prepared_dir: Path, out_dir: Path):
    tfs = read_names(ensure_exists(prepared_dir / "tf_names.txt", "tf_names.txt"))
    genes = read_names(ensure_exists(prepared_dir / "gene_names.txt", "gene_names.txt"))

    node_rna = read_table_auto(
        ensure_exists(prepared_dir / "node_rna_feature.tsv", "node_rna_feature.tsv")
    )
    node_atac = read_table_auto(
        ensure_exists(prepared_dir / "node_atac_feature.tsv", "node_atac_feature.tsv")
    )

    tf_rows = [f"TF::{x}" for x in tfs]
    gene_rows = [f"GENE::{x}" for x in genes]

    for name, df in [("RNA", node_rna), ("ATAC", node_atac)]:
        missing = (set(tf_rows) | set(gene_rows)) - set(df.index.astype(str))
        if missing:
            raise ValueError(
                f"{name} node feature table is missing rows. Examples: {list(missing)[:20]}"
            )

    tf_rna = node_rna.loc[tf_rows].to_numpy(dtype=np.float32)
    gene_rna = node_rna.loc[gene_rows].to_numpy(dtype=np.float32)
    tf_atac = node_atac.loc[tf_rows].to_numpy(dtype=np.float32)
    gene_atac = node_atac.loc[gene_rows].to_numpy(dtype=np.float32)

    edge_df = pd.read_csv(
        ensure_exists(prepared_dir / "tf_gene_edges.csv", "tf_gene_edges.csv")
    )
    required = {"tf_id", "gene_id", "tf", "gene"}
    if not required.issubset(edge_df.columns):
        raise ValueError(
            f"tf_gene_edges.csv must contain {sorted(required)}; found {list(edge_df.columns)}"
        )

    edge_df["tf_id"] = pd.to_numeric(edge_df["tf_id"], errors="raise").astype(int)
    edge_df["gene_id"] = pd.to_numeric(edge_df["gene_id"], errors="raise").astype(int)
    edge_df["tf"] = edge_df["tf"].astype(str)
    edge_df["gene"] = edge_df["gene"].astype(str)

    T, G = len(tfs), len(genes)
    if len(edge_df) != T * G:
        raise ValueError(
            f"Expected complete TF x gene table with {T*G:,} rows; found {len(edge_df):,}."
        )
    if edge_df.duplicated(["tf_id", "gene_id"]).any():
        raise ValueError("Duplicate (tf_id, gene_id) pairs in tf_gene_edges.csv.")
    if ((edge_df["tf_id"] < 0) | (edge_df["tf_id"] >= T)).any():
        raise ValueError("Out-of-range tf_id in tf_gene_edges.csv.")
    if ((edge_df["gene_id"] < 0) | (edge_df["gene_id"] >= G)).any():
        raise ValueError("Out-of-range gene_id in tf_gene_edges.csv.")

    tf_expected = np.asarray(tfs, dtype=str)[edge_df["tf_id"].to_numpy()]
    gene_expected = np.asarray(genes, dtype=str)[edge_df["gene_id"].to_numpy()]
    if not np.array_equal(tf_expected, edge_df["tf"].to_numpy(dtype=str)):
        raise ValueError("tf_gene_edges.csv TF names do not match tf_names.txt IDs.")
    if not np.array_equal(gene_expected, edge_df["gene"].to_numpy(dtype=str)):
        raise ValueError("tf_gene_edges.csv gene names do not match gene_names.txt IDs.")

    edge_attr_path = ensure_exists(
        prepared_dir / "tf_gene_edge_attr.npy", "tf_gene_edge_attr.npy"
    )
    edge_attr = np.load(edge_attr_path, mmap_mode="r")
    if edge_attr.ndim != 3:
        raise ValueError(
            f"tf_gene_edge_attr.npy must have shape [edge,bins,bins], got {edge_attr.shape}."
        )
    if edge_attr.shape[0] != len(edge_df):
        raise ValueError(
            f"edge_attr rows {edge_attr.shape[0]} != edge table rows {len(edge_df)}."
        )
    if edge_attr.shape[1] != edge_attr.shape[2]:
        raise ValueError("TF-gene joint histogram must be square bins x bins.")
    bins = int(edge_attr.shape[1])

    # LUT means we never assume the CSV row order equals Cartesian-product order.
    edge_lut = np.full((T, G), -1, dtype=np.int64)
    edge_lut[
        edge_df["tf_id"].to_numpy(dtype=int),
        edge_df["gene_id"].to_numpy(dtype=int),
    ] = np.arange(len(edge_df), dtype=np.int64)
    if (edge_lut < 0).any():
        raise ValueError("tf_gene_edges.csv is not a complete TF x gene Cartesian product.")

    audit = {
        "n_tf": T,
        "n_gene": G,
        "n_tf_gene_pairs": T * G,
        "edge_hist_shape": list(map(int, edge_attr.shape)),
        "bins": bins,
        "tf_rna_shape": list(map(int, tf_rna.shape)),
        "gene_rna_shape": list(map(int, gene_rna.shape)),
        "tf_atac_shape": list(map(int, tf_atac.shape)),
        "gene_atac_shape": list(map(int, gene_atac.shape)),
        "predictor_tf": "TF RNA + TF gene-level ATAC",
        "predictor_gene": "gene RNA + gene gene-level ATAC",
        "predictor_pair": "TF-gene RNA joint histogram",
        "motif_TSS_distance_used_as_predictor": False,
    }
    with open(out_dir / "04_prepared_data_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    print(
        f"[PREPARED] TF={T}, gene={G}, pairs={T*G:,}, edge_hist={edge_attr.shape}"
    )
    print(
        f"[PREPARED] RNA TF/gene={tf_rna.shape}/{gene_rna.shape}; "
        f"ATAC TF/gene={tf_atac.shape}/{gene_atac.shape}"
    )

    return (
        tfs,
        genes,
        tf_rna,
        gene_rna,
        tf_atac,
        gene_atac,
        edge_df,
        edge_attr,
        edge_lut,
        bins,
    )


# =============================================================================
# Splits
# =============================================================================


def all_pairs_array(T: int, G: int) -> np.ndarray:
    return np.column_stack([
        np.repeat(np.arange(T, dtype=np.int64), G),
        np.tile(np.arange(G, dtype=np.int64), T),
    ])


def pair_labels(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return Z[pairs[:, 0], pairs[:, 1]].astype(np.int8)


def _split_label_stats(Z: np.ndarray, pairs: np.ndarray) -> dict:
    y = pair_labels(Z, pairs)
    return {
        "n": int(len(y)),
        "positive_n": int(y.sum()),
        "negative_n": int(len(y) - y.sum()),
        "positive_fraction": float(y.mean()) if len(y) else float("nan"),
    }


def build_split(
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    T, G = Z.shape
    mode = str(CONFIG["split_mode"])
    n_splits = int(CONFIG["n_splits"])
    split_id = int(CONFIG["split_id"])
    frac = float(CONFIG["train_fraction_within_trainval"])
    seed = int(CONFIG["seed"])

    if n_splits < 2:
        raise ValueError("n_splits must be >=2.")
    if not (1 <= split_id <= n_splits):
        raise ValueError("split_id must be in [1, n_splits].")
    if not (0.0 < frac < 1.0):
        raise ValueError("train_fraction_within_trainval must be in (0,1).")

    rng = np.random.default_rng(seed)
    all_pairs = all_pairs_array(T, G)

    if mode == "pair":
        y_all = pair_labels(Z, all_pairs)
        if len(np.unique(y_all)) < 2:
            raise ValueError("Binary label contains only one class.")
        class_counts = np.bincount(y_all, minlength=2)
        if np.min(class_counts) < n_splits:
            raise ValueError(
                f"The smaller class has only {np.min(class_counts)} samples, "
                f"less than n_splits={n_splits}."
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(all_pairs, y_all))
        trainval_idx, test_idx = folds[split_id - 1]
        trainval = all_pairs[trainval_idx]
        test = all_pairs[test_idx]
        y_trainval = y_all[trainval_idx]

        train, val = train_test_split(
            trainval,
            train_size=frac,
            random_state=seed + split_id,
            shuffle=True,
            stratify=y_trainval,
        )

    elif mode == "leave_tf_out":
        if T < n_splits:
            raise ValueError(f"n_tf={T} < n_splits={n_splits}.")
        ids = np.arange(T, dtype=np.int64)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        trainval_idx, test_idx = list(kf.split(ids))[split_id - 1]
        trainval_tfs = ids[trainval_idx]
        test_tfs = ids[test_idx]
        if len(trainval_tfs) < 2:
            raise ValueError(
                "leave_tf_out needs at least two train/validation TF identities; "
                "reduce n_splits or use more TFs."
            )
        rng.shuffle(trainval_tfs)
        n_train = int(round(len(trainval_tfs) * frac))
        n_train = min(max(n_train, 1), len(trainval_tfs) - 1)
        train_tfs = trainval_tfs[:n_train]
        val_tfs = trainval_tfs[n_train:]

        def pairs_for_tfs(tf_ids: np.ndarray) -> np.ndarray:
            return np.column_stack([
                np.repeat(tf_ids, G),
                np.tile(np.arange(G, dtype=np.int64), len(tf_ids)),
            ])

        train = pairs_for_tfs(train_tfs)
        val = pairs_for_tfs(val_tfs)
        test = pairs_for_tfs(test_tfs)

    elif mode == "leave_gene_out":
        if G < n_splits:
            raise ValueError(f"n_gene={G} < n_splits={n_splits}.")
        ids = np.arange(G, dtype=np.int64)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        trainval_idx, test_idx = list(kf.split(ids))[split_id - 1]
        trainval_genes = ids[trainval_idx]
        test_genes = ids[test_idx]
        if len(trainval_genes) < 2:
            raise ValueError(
                "leave_gene_out needs at least two train/validation gene identities; "
                "reduce n_splits or use more genes."
            )
        rng.shuffle(trainval_genes)
        n_train = int(round(len(trainval_genes) * frac))
        n_train = min(max(n_train, 1), len(trainval_genes) - 1)
        train_genes = trainval_genes[:n_train]
        val_genes = trainval_genes[n_train:]

        def pairs_for_genes(gene_ids: np.ndarray) -> np.ndarray:
            return np.column_stack([
                np.repeat(np.arange(T, dtype=np.int64), len(gene_ids)),
                np.tile(gene_ids, T),
            ])

        train = pairs_for_genes(train_genes)
        val = pairs_for_genes(val_genes)
        test = pairs_for_genes(test_genes)

    else:
        raise ValueError("split_mode must be pair, leave_tf_out, or leave_gene_out.")

    # Deterministic within-split order randomization.
    train = train.copy(); rng.shuffle(train)
    val = val.copy(); rng.shuffle(val)
    test = test.copy(); rng.shuffle(test)

    for split_name, pairs in [("train", train), ("val", val), ("test", test)]:
        y = pair_labels(Z, pairs)
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"{split_name} split contains only one class. Change split_id/n_splits."
            )

    summary = {
        "split_mode": mode,
        "n_splits": n_splits,
        "split_id": split_id,
        "train_fraction_within_trainval": frac,
        "train": _split_label_stats(Z, train),
        "val": _split_label_stats(Z, val),
        "test": _split_label_stats(Z, test),
        "n_train_tfs": int(len(np.unique(train[:, 0]))),
        "n_val_tfs": int(len(np.unique(val[:, 0]))),
        "n_test_tfs": int(len(np.unique(test[:, 0]))),
        "n_train_genes": int(len(np.unique(train[:, 1]))),
        "n_val_genes": int(len(np.unique(val[:, 1]))),
        "n_test_genes": int(len(np.unique(test[:, 1]))),
        "train_test_tf_overlap_n": int(
            len(set(train[:, 0]).intersection(set(test[:, 0])))
        ),
        "train_test_gene_overlap_n": int(
            len(set(train[:, 1]).intersection(set(test[:, 1])))
        ),
    }
    return train, val, test, all_pairs, summary


# =============================================================================
# PCA compression
# =============================================================================


def fit_pca_one(
    train_x: np.ndarray,
    all_x: np.ndarray,
    requested_dim: int,
    name: str,
    out_dir: Path,
) -> Tuple[np.ndarray, dict]:
    train_x = np.asarray(train_x, dtype=np.float32)
    all_x = np.asarray(all_x, dtype=np.float32)

    if train_x.ndim != 2 or all_x.ndim != 2:
        raise ValueError(f"{name}: expected 2-D matrices.")
    if train_x.shape[1] != all_x.shape[1]:
        raise ValueError(f"{name}: train/all feature dimensions differ.")

    if requested_dim <= 0:
        meta = {
            "name": name,
            "pca_used": False,
            "input_dim": int(all_x.shape[1]),
            "output_dim": int(all_x.shape[1]),
        }
        return all_x.astype(np.float32, copy=False), meta

    k = min(
        int(requested_dim),
        int(train_x.shape[0] - 1),
        int(train_x.shape[1]),
    )
    if k < 1:
        raise ValueError(f"{name}: too few training nodes for PCA: {train_x.shape}.")

    print(f"[PCA] {name}: {train_x.shape} -> {k}")
    pca = PCA(
        n_components=k,
        svd_solver="randomized",
        random_state=int(CONFIG["seed"]),
    )
    pca.fit(train_x)
    transformed = pca.transform(all_x).astype(np.float32)
    joblib.dump(pca, out_dir / f"pca_{name}.joblib")

    meta = {
        "name": name,
        "pca_used": True,
        "input_dim": int(all_x.shape[1]),
        "output_dim": int(k),
        "fit_n_training_nodes": int(train_x.shape[0]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return transformed, meta


def compress_node_features(
    tf_rna: np.ndarray,
    gene_rna: np.ndarray,
    tf_atac: np.ndarray,
    gene_atac: np.ndarray,
    train_pairs: np.ndarray,
    out_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    train_tf_ids = np.unique(train_pairs[:, 0])
    train_gene_ids = np.unique(train_pairs[:, 1])
    k = int(CONFIG["pca_dim_per_modality"])

    tf_rna_z, m1 = fit_pca_one(tf_rna[train_tf_ids], tf_rna, k, "tf_rna", out_dir)
    tf_atac_z, m2 = fit_pca_one(tf_atac[train_tf_ids], tf_atac, k, "tf_atac", out_dir)
    gene_rna_z, m3 = fit_pca_one(gene_rna[train_gene_ids], gene_rna, k, "gene_rna", out_dir)
    gene_atac_z, m4 = fit_pca_one(gene_atac[train_gene_ids], gene_atac, k, "gene_atac", out_dir)

    tf_x = np.concatenate([tf_rna_z, tf_atac_z], axis=1).astype(np.float32)
    gene_x = np.concatenate([gene_rna_z, gene_atac_z], axis=1).astype(np.float32)

    meta = {
        "pca_dim_per_modality_requested": k,
        "fit_only_on_training_node_ids": True,
        "tf_input_dim_after_concat": int(tf_x.shape[1]),
        "gene_input_dim_after_concat": int(gene_x.shape[1]),
        "components": [m1, m2, m3, m4],
    }
    with open(out_dir / "06_node_feature_compression.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return tf_x, gene_x, meta


# =============================================================================
# Training-positive bipartite scaffold
# =============================================================================


def build_train_positive_adjacency(
    Z: np.ndarray,
    train_pairs: np.ndarray,
    edge_lut: np.ndarray,
    out_dir: Path,
) -> Tuple[dict, dict]:
    """
    Build the GNN message-passing scaffold from TRAINING POSITIVES ONLY.

    The returned adjacency is seed-shuffled once, so evaluation can take the
    first K neighbors deterministically. During training, optional random
    subsampling is applied on top of this fixed adjacency.
    """
    T, G = Z.shape
    y = pair_labels(Z, train_pairs)
    pos_pairs = np.asarray(train_pairs[y == 1], dtype=np.int64)
    if len(pos_pairs) == 0:
        raise ValueError("No positive training pairs; cannot build GNN scaffold.")

    tf_gene_lists: List[List[int]] = [[] for _ in range(T)]
    tf_edge_lists: List[List[int]] = [[] for _ in range(T)]
    gene_tf_lists: List[List[int]] = [[] for _ in range(G)]
    gene_edge_lists: List[List[int]] = [[] for _ in range(G)]

    for t, g in pos_pairs:
        row = int(edge_lut[int(t), int(g)])
        if row < 0:
            raise RuntimeError(f"Missing edge row for positive training pair {(int(t), int(g))}.")
        tf_gene_lists[int(t)].append(int(g))
        tf_edge_lists[int(t)].append(row)
        gene_tf_lists[int(g)].append(int(t))
        gene_edge_lists[int(g)].append(row)

    rng = np.random.default_rng(int(CONFIG["seed"]) + 1729)

    def finalize(nbr_lists: List[List[int]], row_lists: List[List[int]]):
        nbr_out: List[np.ndarray] = []
        row_out: List[np.ndarray] = []
        for nbrs, rows in zip(nbr_lists, row_lists):
            a = np.asarray(nbrs, dtype=np.int64)
            r = np.asarray(rows, dtype=np.int64)
            if len(a) != len(r):
                raise RuntimeError("Adjacency neighbor/edge-row length mismatch.")
            if len(a) > 1:
                perm = rng.permutation(len(a))
                a = a[perm]
                r = r[perm]
            nbr_out.append(a)
            row_out.append(r)
        return nbr_out, row_out

    tf_gene, tf_edge = finalize(tf_gene_lists, tf_edge_lists)
    gene_tf, gene_edge = finalize(gene_tf_lists, gene_edge_lists)

    # Fixed K+1 candidate neighborhoods allow vectorized deterministic evaluation.
    # The extra slot provides a replacement when the queried edge itself is among
    # the first K scaffold neighbors and must be removed.
    K_eval = int(CONFIG["gnn_max_neighbors"]) + 1
    tf_eval_gene = np.full((T, K_eval), -1, dtype=np.int64)
    tf_eval_edge = np.full((T, K_eval), -1, dtype=np.int64)
    gene_eval_tf = np.full((G, K_eval), -1, dtype=np.int64)
    gene_eval_edge = np.full((G, K_eval), -1, dtype=np.int64)
    for i in range(T):
        n = min(K_eval, len(tf_gene[i]))
        if n:
            tf_eval_gene[i, :n] = tf_gene[i][:n]
            tf_eval_edge[i, :n] = tf_edge[i][:n]
    for i in range(G):
        n = min(K_eval, len(gene_tf[i]))
        if n:
            gene_eval_tf[i, :n] = gene_tf[i][:n]
            gene_eval_edge[i, :n] = gene_edge[i][:n]

    tf_deg = np.asarray([len(x) for x in tf_gene], dtype=int)
    gene_deg = np.asarray([len(x) for x in gene_tf], dtype=int)
    summary = {
        "scaffold_source": "positive training TF-gene labels only",
        "query_edge_excluded_from_own_context": True,
        "n_scaffold_edges": int(len(pos_pairs)),
        "n_tf": int(T),
        "n_gene": int(G),
        "tf_degree_min": int(tf_deg.min()) if len(tf_deg) else 0,
        "tf_degree_median": float(np.median(tf_deg)) if len(tf_deg) else 0.0,
        "tf_degree_mean": float(tf_deg.mean()) if len(tf_deg) else 0.0,
        "tf_degree_max": int(tf_deg.max()) if len(tf_deg) else 0,
        "gene_degree_min": int(gene_deg.min()) if len(gene_deg) else 0,
        "gene_degree_median": float(np.median(gene_deg)) if len(gene_deg) else 0.0,
        "gene_degree_mean": float(gene_deg.mean()) if len(gene_deg) else 0.0,
        "gene_degree_max": int(gene_deg.max()) if len(gene_deg) else 0,
        "isolated_tf_n": int(np.sum(tf_deg == 0)),
        "isolated_gene_n": int(np.sum(gene_deg == 0)),
        "gnn_max_neighbors": int(CONFIG["gnn_max_neighbors"]),
    }
    with open(out_dir / "06b_gnn_scaffold_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    graph = {
        "tf_gene": tf_gene,
        "tf_edge": tf_edge,
        "gene_tf": gene_tf,
        "gene_edge": gene_edge,
        "tf_eval_gene": tf_eval_gene,
        "tf_eval_edge": tf_eval_edge,
        "gene_eval_tf": gene_eval_tf,
        "gene_eval_edge": gene_eval_edge,
    }
    return graph, summary


def sample_context_neighbors(
    pairs: np.ndarray,
    graph: dict,
    max_neighbors: int,
    rng: np.random.Generator,
    training: bool,
) -> dict:
    """
    Sample one-hop bipartite neighborhoods for each query pair.

    Crucially, the exact queried edge is always removed from both endpoint
    neighborhoods if it is present in the training-positive scaffold.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    B = len(pairs)
    K = int(max_neighbors)
    if K < 1:
        raise ValueError("gnn_max_neighbors must be >=1.")

    tf_nbr_gene = np.full((B, K), -1, dtype=np.int64)
    tf_nbr_edge = np.full((B, K), -1, dtype=np.int64)
    gene_nbr_tf = np.full((B, K), -1, dtype=np.int64)
    gene_nbr_edge = np.full((B, K), -1, dtype=np.int64)
    tf_mask = np.zeros((B, K), dtype=bool)
    gene_mask = np.zeros((B, K), dtype=bool)

    random_train = bool(CONFIG.get("gnn_random_neighbor_sampling_train", True)) and training

    if not random_train:
        # Vectorized deterministic evaluation. The scaffold arrays were shuffled
        # once with a fixed seed during construction. We gather K+1 candidates,
        # remove the exact queried partner, then compact the first K valid entries.
        tid = pairs[:, 0]
        gid = pairs[:, 1]

        cand_g = graph["tf_eval_gene"][tid]
        cand_e = graph["tf_eval_edge"][tid]
        valid = (cand_g >= 0) & (cand_g != gid[:, None])
        order = np.argsort(~valid, axis=1, kind="stable")[:, :K]
        sel_g = np.take_along_axis(cand_g, order, axis=1)
        sel_e = np.take_along_axis(cand_e, order, axis=1)
        sel_m = np.take_along_axis(valid, order, axis=1)
        tf_nbr_gene[:] = np.where(sel_m, sel_g, -1)
        tf_nbr_edge[:] = np.where(sel_m, sel_e, -1)
        tf_mask[:] = sel_m

        cand_t = graph["gene_eval_tf"][gid]
        cand_e = graph["gene_eval_edge"][gid]
        valid = (cand_t >= 0) & (cand_t != tid[:, None])
        order = np.argsort(~valid, axis=1, kind="stable")[:, :K]
        sel_t = np.take_along_axis(cand_t, order, axis=1)
        sel_e = np.take_along_axis(cand_e, order, axis=1)
        sel_m = np.take_along_axis(valid, order, axis=1)
        gene_nbr_tf[:] = np.where(sel_m, sel_t, -1)
        gene_nbr_edge[:] = np.where(sel_m, sel_e, -1)
        gene_mask[:] = sel_m

    else:
        def choose(nbrs: np.ndarray, rows: np.ndarray, exclude_id: int):
            if len(nbrs) == 0:
                return nbrs, rows
            keep = nbrs != int(exclude_id)
            a = nbrs[keep]
            r = rows[keep]
            if len(a) <= K:
                return a, r
            sel = rng.choice(len(a), size=K, replace=False)
            return a[sel], r[sel]

        for i, (t, g) in enumerate(pairs):
            t = int(t); g = int(g)

            a, r = choose(graph["tf_gene"][t], graph["tf_edge"][t], g)
            n = min(K, len(a))
            if n:
                tf_nbr_gene[i, :n] = a[:n]
                tf_nbr_edge[i, :n] = r[:n]
                tf_mask[i, :n] = True

            a, r = choose(graph["gene_tf"][g], graph["gene_edge"][g], t)
            n = min(K, len(a))
            if n:
                gene_nbr_tf[i, :n] = a[:n]
                gene_nbr_edge[i, :n] = r[:n]
                gene_mask[i, :n] = True

    return {
        "tf_nbr_gene": tf_nbr_gene,
        "tf_nbr_edge": tf_nbr_edge,
        "tf_mask": tf_mask,
        "gene_nbr_tf": gene_nbr_tf,
        "gene_nbr_edge": gene_nbr_edge,
        "gene_mask": gene_mask,
    }


# =============================================================================
# Model
# =============================================================================


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeAttrEncoder(nn.Module):
    def __init__(self, bins: int, edge_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bins * bins, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeAwareNeighborAggregator(nn.Module):
    """One edge-aware attention message-passing layer for one endpoint type."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        attention_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, attention_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_h: torch.Tensor,       # [B,H]
        neighbor_h: torch.Tensor,    # [B,K,H]
        edge_h: torch.Tensor,        # [B,K,E]
        mask: torch.Tensor,          # [B,K] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, H = neighbor_h.shape
        q = query_h.unsqueeze(1).expand(-1, K, -1)

        msg = self.message(torch.cat([neighbor_h, edge_h], dim=-1))
        attn_logits = self.attention(torch.cat([q, neighbor_h, edge_h], dim=-1)).squeeze(-1)

        valid = mask.bool()
        attn_logits = attn_logits.masked_fill(~valid, -1e9)
        alpha = torch.softmax(attn_logits, dim=1)
        alpha = alpha * valid.to(alpha.dtype)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)

        agg = torch.sum(alpha.unsqueeze(-1) * msg, dim=1)
        delta = self.update(torch.cat([query_h, agg], dim=1))
        updated = self.norm(query_h + self.dropout(delta))

        # For isolated endpoints, preserve the node encoder output exactly.
        has_neighbor = valid.any(dim=1, keepdim=True)
        updated = torch.where(has_neighbor, updated, query_h)
        return updated, alpha


class TFgeneEdgeGNNClassifier(nn.Module):
    def __init__(
        self,
        tf_in_dim: int,
        gene_in_dim: int,
        bins: int,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        attention_hidden_dim: int = 64,
        dropout: float = 0.2,
        use_gnn: bool = True,
    ):
        super().__init__()
        self.use_gnn = bool(use_gnn)
        self.tf_encoder = NodeEncoder(tf_in_dim, hidden_dim, dropout)
        self.gene_encoder = NodeEncoder(gene_in_dim, hidden_dim, dropout)
        self.edge_encoder = EdgeAttrEncoder(bins, edge_dim, dropout)

        self.tf_context = EdgeAwareNeighborAggregator(
            hidden_dim, edge_dim, attention_hidden_dim, dropout
        )
        self.gene_context = EdgeAwareNeighborAggregator(
            hidden_dim, edge_dim, attention_hidden_dim, dropout
        )

        # Keep the original decoder form so the GNN contribution is isolated to
        # contextualized node representations rather than a changed link decoder.
        decoder_in = hidden_dim * 3 + edge_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _encode_neighbors(
        encoder: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, K, D = x.shape
        h = encoder(x.reshape(B * K, D))
        return h.reshape(B, K, -1)

    @staticmethod
    def _encode_neighbor_edges(
        encoder: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, K, b1, b2 = x.shape
        h = encoder(x.reshape(B * K, b1, b2))
        return h.reshape(B, K, -1)

    def forward(
        self,
        tf_x: torch.Tensor,
        gene_x: torch.Tensor,
        query_edge_hist: torch.Tensor,
        tf_neighbor_gene_x: Optional[torch.Tensor] = None,
        tf_neighbor_edge_hist: Optional[torch.Tensor] = None,
        tf_neighbor_mask: Optional[torch.Tensor] = None,
        gene_neighbor_tf_x: Optional[torch.Tensor] = None,
        gene_neighbor_edge_hist: Optional[torch.Tensor] = None,
        gene_neighbor_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        h_t = self.tf_encoder(tf_x)
        h_g = self.gene_encoder(gene_x)
        h_e = self.edge_encoder(query_edge_hist)

        attn_t = None
        attn_g = None
        if self.use_gnn:
            required = [
                tf_neighbor_gene_x,
                tf_neighbor_edge_hist,
                tf_neighbor_mask,
                gene_neighbor_tf_x,
                gene_neighbor_edge_hist,
                gene_neighbor_mask,
            ]
            if any(x is None for x in required):
                raise ValueError("GNN is enabled but neighborhood tensors are missing.")

            h_tf_nbr_gene = self._encode_neighbors(self.gene_encoder, tf_neighbor_gene_x)
            h_tf_nbr_edge = self._encode_neighbor_edges(self.edge_encoder, tf_neighbor_edge_hist)
            h_t, attn_t = self.tf_context(
                h_t, h_tf_nbr_gene, h_tf_nbr_edge, tf_neighbor_mask
            )

            h_gene_nbr_tf = self._encode_neighbors(self.tf_encoder, gene_neighbor_tf_x)
            h_gene_nbr_edge = self._encode_neighbor_edges(self.edge_encoder, gene_neighbor_edge_hist)
            h_g, attn_g = self.gene_context(
                h_g, h_gene_nbr_tf, h_gene_nbr_edge, gene_neighbor_mask
            )

        z = torch.cat([h_t, h_g, h_t * h_g, h_e], dim=1)
        logits = self.decoder(z).squeeze(1)
        if return_attention:
            return logits, attn_t, attn_g
        return logits


# =============================================================================
# Batch / prediction utilities
# =============================================================================


def pair_edge_hist_batch(
    pairs: np.ndarray,
    edge_lut: np.ndarray,
    edge_attr_mmap,
) -> np.ndarray:
    rows = edge_lut[pairs[:, 0], pairs[:, 1]]
    if (rows < 0).any():
        raise RuntimeError("Missing edge-attribute row for a TF-gene pair.")
    return np.asarray(edge_attr_mmap[rows], dtype=np.float32)


def _neighbor_hist_batch(rows: np.ndarray, mask: np.ndarray, edge_attr_mmap) -> np.ndarray:
    """Load padded neighbor edge histograms; invalid slots are returned as zeros."""
    rows = np.asarray(rows, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    B, K = rows.shape
    bins = int(edge_attr_mmap.shape[1])
    out = np.zeros((B, K, bins, bins), dtype=np.float32)
    flat_mask = mask.reshape(-1)
    if np.any(flat_mask):
        flat_rows = rows.reshape(-1)[flat_mask]
        vals = np.asarray(edge_attr_mmap[flat_rows], dtype=np.float32)
        out.reshape(B * K, bins, bins)[flat_mask] = vals
    return out


def make_model_batch(
    pairs: np.ndarray,
    tf_x_dev: torch.Tensor,
    gene_x_dev: torch.Tensor,
    edge_lut: np.ndarray,
    edge_attr_mmap,
    graph: dict,
    rng: np.random.Generator,
    device: torch.device,
    training: bool,
) -> dict:
    pairs = np.asarray(pairs, dtype=np.int64)
    tid = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device)
    gid = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device)

    query_hist_np = pair_edge_hist_batch(pairs, edge_lut, edge_attr_mmap)
    query_hist = torch.from_numpy(query_hist_np).to(device=device, dtype=torch.float32)

    batch = {
        "tf_x": tf_x_dev[tid],
        "gene_x": gene_x_dev[gid],
        "query_edge_hist": query_hist,
    }

    if not bool(CONFIG.get("use_gnn", True)):
        return batch

    ctx = sample_context_neighbors(
        pairs=pairs,
        graph=graph,
        max_neighbors=int(CONFIG["gnn_max_neighbors"]),
        rng=rng,
        training=training,
    )

    # Padded -1 IDs are replaced by 0 only for safe tensor indexing. Their masks
    # are False, so the corresponding representations never contribute.
    tf_nbr_gene_safe = np.where(ctx["tf_mask"], ctx["tf_nbr_gene"], 0)
    gene_nbr_tf_safe = np.where(ctx["gene_mask"], ctx["gene_nbr_tf"], 0)

    tf_nbr_gene_ids = torch.from_numpy(tf_nbr_gene_safe).to(device=device, dtype=torch.long)
    gene_nbr_tf_ids = torch.from_numpy(gene_nbr_tf_safe).to(device=device, dtype=torch.long)

    tf_nbr_hist_np = _neighbor_hist_batch(ctx["tf_nbr_edge"], ctx["tf_mask"], edge_attr_mmap)
    gene_nbr_hist_np = _neighbor_hist_batch(ctx["gene_nbr_edge"], ctx["gene_mask"], edge_attr_mmap)

    batch.update({
        "tf_neighbor_gene_x": gene_x_dev[tf_nbr_gene_ids],
        "tf_neighbor_edge_hist": torch.from_numpy(tf_nbr_hist_np).to(
            device=device, dtype=torch.float32
        ),
        "tf_neighbor_mask": torch.from_numpy(ctx["tf_mask"]).to(device=device),
        "gene_neighbor_tf_x": tf_x_dev[gene_nbr_tf_ids],
        "gene_neighbor_edge_hist": torch.from_numpy(gene_nbr_hist_np).to(
            device=device, dtype=torch.float32
        ),
        "gene_neighbor_mask": torch.from_numpy(ctx["gene_mask"]).to(device=device),
    })
    return batch


def predict_logits(
    model: nn.Module,
    pairs: np.ndarray,
    tf_x_dev: torch.Tensor,
    gene_x_dev: torch.Tensor,
    edge_lut: np.ndarray,
    edge_attr_mmap,
    graph: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out = np.empty(len(pairs), dtype=np.float32)
    # Evaluation neighbor selection is deterministic because training=False.
    rng = np.random.default_rng(int(CONFIG["seed"]) + 99991)
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            mb = make_model_batch(
                batch_pairs, tf_x_dev, gene_x_dev,
                edge_lut, edge_attr_mmap, graph, rng, device,
                training=False,
            )
            logits = model(**mb)
            out[start:start + len(batch_pairs)] = (
                logits.detach().cpu().numpy().astype(np.float32)
            )
    return out


def predict_scores(*args, **kwargs) -> np.ndarray:
    return expit(predict_logits(*args, **kwargs)).astype(np.float32)


# =============================================================================
# Loss
# =============================================================================


def binary_loss_vector(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor,
    mode: str,
    focal_gamma: float,
) -> torch.Tensor:
    """Weighted BCE, optionally modulated by focal factor for hard examples."""
    base = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    if mode == "weighted_bce":
        return base
    if mode == "focal":
        p = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, p, 1.0 - p)
        focal_factor = (1.0 - pt).clamp(min=0.0).pow(float(focal_gamma))
        return focal_factor * base
    raise ValueError("loss_mode must be 'weighted_bce' or 'focal'.")


# =============================================================================
# Metrics and VALIDATION-only threshold selection
# =============================================================================


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) >= 2 else float("nan")


def safe_average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.any(y == 1) else float("nan")


def threshold_curve(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """
    Exact metrics at every distinct score threshold using rule score >= threshold.
    Rows are ordered from high threshold (few positives) to low threshold.
    """
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(score, dtype=float).reshape(-1)
    finite = np.isfinite(p)
    y, p = y[finite], p[finite]
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Threshold selection requires both classes in validation data.")

    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order].astype(np.int64)

    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(1 - ys)
    total_pos = int(ys.sum())
    total_neg = int(len(ys) - total_pos)

    distinct_end = np.r_[np.where(ps[:-1] != ps[1:])[0], len(ps) - 1]
    threshold = ps[distinct_end]
    tp = tp_cum[distinct_end].astype(float)
    fp = fp_cum[distinct_end].astype(float)
    fn = total_pos - tp
    tn = total_neg - fp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, total_pos, out=np.zeros_like(tp), where=total_pos > 0)
    specificity = np.divide(tn, total_neg, out=np.zeros_like(tn), where=total_neg > 0)

    f1_denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp), where=f1_denom > 0)

    beta = float(CONFIG.get("f_beta", 2.0))
    b2 = beta * beta
    fb_denom = (1 + b2) * tp + b2 * fn + fp
    f_beta = np.divide(
        (1 + b2) * tp,
        fb_denom,
        out=np.zeros_like(tp),
        where=fb_denom > 0,
    )

    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(mcc_num, mcc_den, out=np.zeros_like(tp), where=mcc_den > 0)

    balanced_accuracy = 0.5 * (recall + specificity)

    return pd.DataFrame({
        "threshold": threshold.astype(float),
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "fn": fn.astype(int),
        "tn": tn.astype(int),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        f"f{beta:g}": f_beta,
        "mcc": mcc,
        "predicted_positive_fraction": (tp + fp) / len(y),
    })


def _row_to_dict(row: pd.Series, name: str) -> dict:
    out = {"strategy": name}
    for k, v in row.to_dict().items():
        if isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def choose_best_row(curve: pd.DataFrame, metric: str) -> pd.Series:
    beta = float(CONFIG.get("f_beta", 2.0))
    metric_col = {"mcc": "mcc", "f1": "f1", "f2": f"f{beta:g}"}.get(metric)
    if metric_col is None:
        raise ValueError("metric must be mcc, f1, or f2.")
    max_val = curve[metric_col].max()
    candidates = curve[np.isclose(curve[metric_col], max_val, rtol=0, atol=1e-15)]
    # Tie-break: higher precision, then higher recall, then higher threshold.
    candidates = candidates.sort_values(
        ["precision", "recall", "threshold"], ascending=[False, False, False]
    )
    return candidates.iloc[0]


def choose_target_recall_row(curve: pd.DataFrame, target_recall: float) -> pd.Series:
    if not (0 < target_recall <= 1):
        raise ValueError("target_recall must be in (0,1].")
    candidates = curve[curve["recall"] + 1e-12 >= float(target_recall)]
    if len(candidates) == 0:
        # With the minimum threshold recall should be 1, but retain a safe fallback.
        warnings.warn(
            f"No validation threshold reached recall={target_recall}; using maximum-recall point."
        )
        max_recall = curve["recall"].max()
        candidates = curve[np.isclose(curve["recall"], max_recall)]

    # Primary objective requested by the design: among points meeting recall,
    # maximize precision. Tie-break with MCC, then use the higher threshold.
    candidates = candidates.sort_values(
        ["precision", "mcc", "threshold"], ascending=[False, False, False]
    )
    return candidates.iloc[0]


def select_validation_threshold(
    y_val: np.ndarray,
    score_val: np.ndarray,
    out_dir: Path,
) -> Tuple[float, dict, Dict[str, float]]:
    """
    Select primary threshold on validation only and export compact alternatives.
    Returns primary threshold, primary info, and named alternative thresholds.
    """
    curve = threshold_curve(y_val, score_val)
    strategy = str(CONFIG["threshold_strategy"]).lower()
    target_recall = float(CONFIG["target_recall"])

    if strategy == "target_recall":
        primary_row = choose_target_recall_row(curve, target_recall)
        primary_name = f"target_recall_{target_recall:g}"
    elif strategy in {"mcc", "f1", "f2"}:
        primary_row = choose_best_row(curve, strategy)
        primary_name = strategy
    else:
        raise ValueError("threshold_strategy must be target_recall, mcc, f1, or f2.")

    alternatives: Dict[str, float] = {
        "primary": float(primary_row["threshold"]),
        "mcc": float(choose_best_row(curve, "mcc")["threshold"]),
        "f1": float(choose_best_row(curve, "f1")["threshold"]),
        "f2": float(choose_best_row(curve, "f2")["threshold"]),
        "fixed_0.5": 0.5,
    }

    compact_rows = []
    compact_rows.append(_row_to_dict(primary_row, f"PRIMARY:{primary_name}"))
    for metric in ["mcc", "f1", "f2"]:
        compact_rows.append(_row_to_dict(choose_best_row(curve, metric), f"best_{metric}"))
    for r in CONFIG.get("report_recall_targets", []):
        rr = float(r)
        compact_rows.append(
            _row_to_dict(choose_target_recall_row(curve, rr), f"target_recall_{rr:g}")
        )

    compact = pd.DataFrame(compact_rows).drop_duplicates(
        subset=["strategy", "threshold"], keep="first"
    )
    compact.to_csv(out_dir / "07b_validation_threshold_operating_points.csv", index=False)

    primary_info = _row_to_dict(primary_row, primary_name)
    primary_info.update({
        "selection_dataset": "validation_only",
        "configured_strategy": strategy,
        "configured_target_recall": target_recall,
        "validation_positive_fraction": float(np.mean(y_val)),
        "interpretation": (
            "Threshold was fixed before test evaluation. Under target_recall, "
            "precision is maximized among validation thresholds meeting the recall target."
        ),
    })
    return float(primary_row["threshold"]), primary_info, alternatives


def classification_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(score, dtype=float).reshape(-1)
    finite = np.isfinite(p)
    y, p = y[finite], p[finite]
    pred = (p >= float(threshold)).astype(np.int8)

    n = int(len(y))
    pos = int(y.sum())
    neg = n - pos
    out = {
        "n": n,
        "positive_n": pos,
        "negative_n": neg,
        "positive_fraction": float(pos / n) if n else float("nan"),
        "threshold": float(threshold),
        "roc_auc": safe_roc_auc(y, p) if n else float("nan"),
        "average_precision_pr_auc": safe_average_precision(y, p) if n else float("nan"),
    }
    if n == 0:
        return out

    p_clip = np.clip(p, 1e-7, 1 - 1e-7)
    out["log_loss"] = float(log_loss(y, p_clip, labels=[0, 1]))
    out["brier_score"] = float(brier_score_loss(y, p_clip))
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = (
        float(balanced_accuracy_score(y, pred))
        if len(np.unique(y)) >= 2 else float("nan")
    )
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall_sensitivity"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    beta = float(CONFIG.get("f_beta", 2.0))
    out[f"f{beta:g}"] = float(fbeta_score(y, pred, beta=beta, zero_division=0))
    out["mcc"] = (
        float(matthews_corrcoef(y, pred))
        if len(np.unique(y)) >= 2 else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "predicted_positive_fraction": float(pred.mean()),
    })
    return out


# =============================================================================
# Label-only baselines (evaluation only)
# =============================================================================


def build_prevalence_baselines(
    Z: np.ndarray,
    train_pairs: np.ndarray,
    query_pairs: np.ndarray,
    smoothing: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Detect whether TF/gene identity prevalence alone explains the task."""
    y_train = pair_labels(Z, train_pairs).astype(float)
    global_p = float(y_train.mean())
    T, G = Z.shape

    tf_pos = np.zeros(T, dtype=float)
    tf_n = np.zeros(T, dtype=float)
    gene_pos = np.zeros(G, dtype=float)
    gene_n = np.zeros(G, dtype=float)
    np.add.at(tf_pos, train_pairs[:, 0], y_train)
    np.add.at(tf_n, train_pairs[:, 0], 1.0)
    np.add.at(gene_pos, train_pairs[:, 1], y_train)
    np.add.at(gene_n, train_pairs[:, 1], 1.0)

    tf_p = (tf_pos + smoothing * global_p) / (tf_n + smoothing)
    gene_p = (gene_pos + smoothing * global_p) / (gene_n + smoothing)

    eps = 1e-5
    gp = np.clip(global_p, eps, 1 - eps)
    tf_p = np.clip(tf_p, eps, 1 - eps)
    gene_p = np.clip(gene_p, eps, 1 - eps)

    tid = query_pairs[:, 0]
    gid = query_pairs[:, 1]
    additive_logit = logit(tf_p[tid]) + logit(gene_p[gid]) - logit(gp)

    return {
        "global_prevalence": np.full(len(query_pairs), global_p, dtype=float),
        "tf_prevalence": tf_p[tid],
        "gene_prevalence": gene_p[gid],
        "additive_tf_gene_logit": expit(additive_logit),
    }


# =============================================================================
# Error diagnostics
# =============================================================================


def save_test_error_diagnostics(
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Summarize TP/FN/FP/TN, with raw E used only as label-source audit."""
    y = test_df["observed_motif_TSS_hit_label"].to_numpy(dtype=int)
    pred = test_df["predicted_class"].to_numpy(dtype=int)
    group = np.full(len(test_df), "", dtype=object)
    group[(y == 1) & (pred == 1)] = "TP"
    group[(y == 1) & (pred == 0)] = "FN"
    group[(y == 0) & (pred == 1)] = "FP"
    group[(y == 0) & (pred == 0)] = "TN"
    test_df = test_df.copy()
    test_df["error_group"] = group

    records = []
    for g in ["TP", "FN", "FP", "TN"]:
        d = test_df[test_df["error_group"] == g]
        score = d["predicted_motif_TSS_hit_score"].to_numpy(dtype=float)
        E = d["raw_distance_weighted_E_audit_only"].to_numpy(dtype=float)
        records.append({
            "group": g,
            "n": int(len(d)),
            "score_mean": float(np.mean(score)) if len(d) else float("nan"),
            "score_median": float(np.median(score)) if len(d) else float("nan"),
            "score_q10": float(np.quantile(score, 0.10)) if len(d) else float("nan"),
            "score_q90": float(np.quantile(score, 0.90)) if len(d) else float("nan"),
            "raw_E_mean": float(np.mean(E)) if len(d) else float("nan"),
            "raw_E_median": float(np.median(E)) if len(d) else float("nan"),
            "raw_E_q10": float(np.quantile(E, 0.10)) if len(d) else float("nan"),
            "raw_E_q90": float(np.quantile(E, 0.90)) if len(d) else float("nan"),
        })
    pd.DataFrame(records).to_csv(out_dir / "10c_test_error_group_summary.csv", index=False)


# =============================================================================
# Training
# =============================================================================


def train_model(
    tf_x: np.ndarray,
    gene_x: np.ndarray,
    Z: np.ndarray,
    E: np.ndarray,
    train_pairs: np.ndarray,
    val_pairs: np.ndarray,
    test_pairs: np.ndarray,
    all_pairs: np.ndarray,
    edge_lut: np.ndarray,
    edge_attr_mmap,
    bins: int,
    tfs: pd.Index,
    genes: pd.Index,
    out_dir: Path,
):
    device = setup_device(str(CONFIG["device"]))
    seed = int(CONFIG["seed"])
    rng = np.random.default_rng(seed)

    y_train = pair_labels(Z, train_pairs).astype(np.float32)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training split must contain both positive and negative labels.")

    raw_pos_weight = float(n_neg / n_pos)
    if bool(CONFIG["use_pos_weight"]):
        pos_weight = raw_pos_weight * float(CONFIG.get("pos_weight_multiplier", 1.0))
        cap = CONFIG.get("max_pos_weight")
        if cap is not None:
            pos_weight = min(pos_weight, float(cap))
    else:
        pos_weight = 1.0

    print(
        f"[CLASS] train positive={n_pos:,}, negative={n_neg:,}, "
        f"fraction={n_pos/len(y_train):.6f}, raw_pos_weight={raw_pos_weight:.4g}, "
        f"used_pos_weight={pos_weight:.4g}"
    )

    # Build graph strictly from training positives. This happens after splitting.
    graph, graph_summary = build_train_positive_adjacency(
        Z=Z,
        train_pairs=train_pairs,
        edge_lut=edge_lut,
        out_dir=out_dir,
    )
    print("[GNN SCAFFOLD]", json.dumps(graph_summary, ensure_ascii=False))

    tf_x_dev = torch.from_numpy(tf_x).to(device=device, dtype=torch.float32)
    gene_x_dev = torch.from_numpy(gene_x).to(device=device, dtype=torch.float32)

    model = TFgeneEdgeGNNClassifier(
        tf_in_dim=tf_x.shape[1],
        gene_in_dim=gene_x.shape[1],
        bins=bins,
        hidden_dim=int(CONFIG["hidden_dim"]),
        edge_dim=int(CONFIG["edge_dim"]),
        attention_hidden_dim=int(CONFIG["gnn_attention_hidden_dim"]),
        dropout=float(CONFIG["dropout"]),
        use_gnn=bool(CONFIG.get("use_gnn", True)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(CONFIG["lr"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )

    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    loss_mode = str(CONFIG["loss_mode"])
    focal_gamma = float(CONFIG["focal_gamma"])

    print(model)
    print(
        "[MODEL] trainable params:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    print(
        f"[GNN] enabled={bool(CONFIG.get('use_gnn', True))}, "
        f"max_neighbors={int(CONFIG['gnn_max_neighbors'])}, "
        "scaffold=train-positive-only, query-edge-exclusion=True"
    )
    print(f"[LOSS] mode={loss_mode}, focal_gamma={focal_gamma:g}")

    batch_size = int(CONFIG["batch_size"])
    pred_batch = int(CONFIG["prediction_batch_size"])
    max_epochs = int(CONFIG["max_epochs"])
    patience = int(CONFIG["patience"])
    early_metric = str(CONFIG["early_stop_metric"])

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(train_pairs))
        running_loss_sum = 0.0
        running_n = 0

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch_pairs = train_pairs[idx]
            y_np = Z[batch_pairs[:, 0], batch_pairs[:, 1]].astype(np.float32)
            target = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)

            mb = make_model_batch(
                batch_pairs, tf_x_dev, gene_x_dev,
                edge_lut, edge_attr_mmap, graph, rng, device,
                training=True,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(**mb)
            loss_vec = binary_loss_vector(
                logits, target, pos_weight_tensor, loss_mode, focal_gamma
            )
            loss = loss_vec.mean()
            loss.backward()

            clip = float(CONFIG["gradient_clip_val"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            running_loss_sum += float(loss_vec.detach().sum().cpu())
            running_n += len(batch_pairs)

        train_loss = running_loss_sum / max(1, running_n)

        # Validation uses the training-positive scaffold only, and every query
        # edge is removed from its own local context by sample_context_neighbors().
        val_logits = predict_logits(
            model, val_pairs, tf_x_dev, gene_x_dev,
            edge_lut, edge_attr_mmap, graph, device, pred_batch,
        )
        val_score = expit(val_logits)
        val_y = pair_labels(Z, val_pairs).astype(np.float32)

        with torch.no_grad():
            vl = binary_loss_vector(
                torch.from_numpy(val_logits).to(device=device, dtype=torch.float32),
                torch.from_numpy(val_y).to(device=device, dtype=torch.float32),
                pos_weight_tensor,
                loss_mode,
                focal_gamma,
            )
            val_loss = float(vl.mean().cpu())

        val_ap = safe_average_precision(val_y.astype(np.int8), val_score)
        val_roc = safe_roc_auc(val_y.astype(np.int8), val_score)

        if early_metric == "average_precision" and np.isfinite(val_ap):
            monitor = val_ap
        elif early_metric == "roc_auc" and np.isfinite(val_roc):
            monitor = val_roc
        else:
            monitor = -val_loss

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_average_precision": val_ap,
            "val_roc_auc": val_roc,
            "monitor_score": monitor,
        })
        pd.DataFrame(history).to_csv(out_dir / "07_training_history.csv", index=False)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch:04d} | train_loss={train_loss:.6g} | "
                f"val_loss={val_loss:.6g} | val_AP={val_ap:.6g} | "
                f"val_AUC={val_roc:.6g}"
            )

        if monitor > best_score + 1e-12:
            best_score = float(monitor)
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save({
                "model_state_dict": best_state,
                "tf_input_dim": int(tf_x.shape[1]),
                "gene_input_dim": int(gene_x.shape[1]),
                "bins": bins,
                "train_positive_fraction": float(y_train.mean()),
                "raw_pos_weight": raw_pos_weight,
                "used_pos_weight": pos_weight,
                "gnn_scaffold_summary": graph_summary,
                "config": CONFIG,
            }, out_dir / "best_model.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(
                f"[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}, "
                f"best_monitor={best_score:.6g}"
            )
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Scores from best checkpoint. Query-edge exclusion is applied even to
    # training positives, preventing direct-edge self-leakage in train metrics.
    score_train = predict_scores(
        model, train_pairs, tf_x_dev, gene_x_dev,
        edge_lut, edge_attr_mmap, graph, device, pred_batch,
    )
    score_val = predict_scores(
        model, val_pairs, tf_x_dev, gene_x_dev,
        edge_lut, edge_attr_mmap, graph, device, pred_batch,
    )
    score_test = predict_scores(
        model, test_pairs, tf_x_dev, gene_x_dev,
        edge_lut, edge_attr_mmap, graph, device, pred_batch,
    )
    score_all = predict_scores(
        model, all_pairs, tf_x_dev, gene_x_dev,
        edge_lut, edge_attr_mmap, graph, device, pred_batch,
    )

    y_val = pair_labels(Z, val_pairs)
    primary_threshold, threshold_info, alternative_thresholds = select_validation_threshold(
        y_val, score_val, out_dir
    )

    y_train_i = pair_labels(Z, train_pairs)
    y_test = pair_labels(Z, test_pairs)

    test_by_threshold = {}
    val_by_threshold = {}
    for name, th in alternative_thresholds.items():
        val_by_threshold[name] = classification_metrics(y_val, score_val, th)
        test_by_threshold[name] = classification_metrics(y_test, score_test, th)

    metrics = {
        "best_epoch": best_epoch,
        "best_monitor_score": best_score,
        "best_val_loss": best_val_loss,
        "model_family": "edge-aware bipartite one-hop GNN link classifier",
        "gnn": {
            "enabled": bool(CONFIG.get("use_gnn", True)),
            "scaffold_source": "training positives only",
            "query_edge_excluded_from_own_context": True,
            "max_neighbors": int(CONFIG["gnn_max_neighbors"]),
            "aggregation": "edge-aware attention",
            "scaffold_summary": graph_summary,
        },
        "loss": {
            "mode": loss_mode,
            "focal_gamma": focal_gamma,
        },
        "class_weight": {
            "train_positive_fraction": float(y_train.mean()),
            "raw_neg_over_pos": raw_pos_weight,
            "pos_weight_multiplier": float(CONFIG.get("pos_weight_multiplier", 1.0)),
            "used_pos_weight": pos_weight,
            "note": (
                "With class-weighted BCE/focal loss, sigmoid output is best treated "
                "as a classification/ranking score unless separately calibrated."
            ),
        },
        "primary_threshold_selection_validation_only": threshold_info,
        "train_primary": classification_metrics(y_train_i, score_train, primary_threshold),
        "val_primary": classification_metrics(y_val, score_val, primary_threshold),
        "test_primary": classification_metrics(y_test, score_test, primary_threshold),
        "validation_metrics_by_validation_derived_threshold": val_by_threshold,
        "test_metrics_by_validation_derived_threshold": test_by_threshold,
    }

    # Label-only prevalence baselines are diagnostics, never model inputs.
    baseline_scores = build_prevalence_baselines(Z, train_pairs, test_pairs)
    metrics["test_label_only_baselines"] = {
        name: {
            "roc_auc": safe_roc_auc(y_test, s),
            "average_precision_pr_auc": safe_average_precision(y_test, s),
        }
        for name, s in baseline_scores.items()
    }

    with open(out_dir / "08_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    pred_test = (score_test >= primary_threshold).astype(np.int8)
    test_df = pd.DataFrame({
        "TF": np.asarray(tfs, dtype=str)[test_pairs[:, 0]],
        "gene": np.asarray(genes, dtype=str)[test_pairs[:, 1]],
        "observed_motif_TSS_hit_label": y_test.astype(np.int8),
        "predicted_motif_TSS_hit_score": score_test,
        "predicted_class": pred_test,
        "validation_selected_threshold": float(primary_threshold),
        "raw_distance_weighted_E_audit_only": E[test_pairs[:, 0], test_pairs[:, 1]],
    })
    test_df["is_false_negative"] = (
        (test_df["observed_motif_TSS_hit_label"] == 1)
        & (test_df["predicted_class"] == 0)
    )
    test_df["is_false_positive"] = (
        (test_df["observed_motif_TSS_hit_label"] == 0)
        & (test_df["predicted_class"] == 1)
    )
    test_df.to_csv(out_dir / "09_test_pair_predictions.csv", index=False)
    save_test_error_diagnostics(test_df, out_dir)

    score_matrix = np.empty(Z.shape, dtype=np.float32)
    score_matrix[all_pairs[:, 0], all_pairs[:, 1]] = score_all
    pd.DataFrame(score_matrix, index=tfs, columns=genes).to_csv(
        out_dir / "10_best_all_TF_gene_motif_hit_score.csv"
    )

    binary_matrix = (score_matrix >= primary_threshold).astype(np.int8)
    pd.DataFrame(binary_matrix, index=tfs, columns=genes).to_csv(
        out_dir / "10b_best_all_TF_gene_motif_hit_binary.csv"
    )

    print("[THRESHOLD]", json.dumps(threshold_info, ensure_ascii=False))
    print("[TEST]", json.dumps(metrics["test_primary"], ensure_ascii=False))
    return model, score_matrix, binary_matrix, metrics, float(primary_threshold), graph_summary


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    seed_everything(int(CONFIG["seed"]))

    prepared_dir = ensure_exists(CONFIG["prepared_dir"], "prepared_dir")
    label_dir = ensure_exists(CONFIG["label_dir"], "label_dir")
    out_dir = Path(CONFIG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("EDGE-AWARE BIPARTITE GNN TF-GENE MOTIF/TSS-HIT CLASSIFIER")
    print("=" * 88)
    print("Predictors: TF/gene RNA + gene-level ATAC + TF-gene RNA joint histogram")
    print("GNN scaffold: positive TRAIN pairs only; exact query edge excluded from its own context")
    print("Target: 1 iff >=1 valid motif/TSS hit (equivalently E_tg > 0)")
    print("Motif/TSS/distance are supervision only; never predictor inputs.")
    print(
        f"Threshold strategy: {CONFIG['threshold_strategy']} "
        f"(target_recall={CONFIG['target_recall']}) on VALIDATION only"
    )
    print("")

    # 1. Pure TF/gene features and pair covariates.
    (
        tfs,
        genes,
        tf_rna,
        gene_rna,
        tf_atac,
        gene_atac,
        edge_df,
        edge_attr,
        edge_lut,
        bins,
    ) = load_prepared_data(prepared_dir, out_dir)

    # 2. Pure TF-gene binary supervision.
    Z, E, label_summary = build_binary_tf_gene_labels(
        label_dir=label_dir,
        tfs=tfs,
        genes=genes,
        out_dir=out_dir,
    )
    print(
        f"[LABEL] shape={Z.shape}, positives={int(Z.sum()):,}/{Z.size:,} "
        f"({float(Z.mean()):.6f})"
    )

    # 3. Split.
    train_pairs, val_pairs, test_pairs, all_pairs, split_summary = build_split(Z)
    with open(out_dir / "05_split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, indent=2, ensure_ascii=False)
    print("[SPLIT]", json.dumps(split_summary, ensure_ascii=False))

    # 4. Node-profile compression fit on TRAIN node identities only.
    tf_x, gene_x, compression_meta = compress_node_features(
        tf_rna=tf_rna,
        gene_rna=gene_rna,
        tf_atac=tf_atac,
        gene_atac=gene_atac,
        train_pairs=train_pairs,
        out_dir=out_dir,
    )

    # 5. Train and evaluate edge-aware bipartite GNN classifier.
    model, score_matrix, binary_matrix, metrics, threshold, graph_summary = train_model(
        tf_x=tf_x,
        gene_x=gene_x,
        Z=Z,
        E=E,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        all_pairs=all_pairs,
        edge_lut=edge_lut,
        edge_attr_mmap=edge_attr,
        bins=bins,
        tfs=tfs,
        genes=genes,
        out_dir=out_dir,
    )

    run_meta = {
        "model": "edge-aware bipartite GNN TF-gene binary classifier",
        "target": "Z_tg = 1[E_tg > 0]",
        "positive_semantics": ">=1 valid motif/TSS hit under existing label filters",
        "negative_semantics": "0 observed valid motif/TSS hits under existing label filters",
        "raw_E": "audit only; not target magnitude and not predictor",
        "predictor_tf": "[TF RNA, TF gene-level ATAC]",
        "predictor_gene": "[gene RNA, gene gene-level ATAC]",
        "predictor_pair": "TF-gene RNA joint histogram",
        "gnn": "one-hop edge-aware attention over positive-training bipartite neighbors with query-edge exclusion",
        "decoder": "MLP([contextual_h_t, contextual_h_g, contextual_h_t * contextual_h_g, h_edge]) -> logit",
        "motif_TSS_distance_used_as_predictor": False,
        "gnn_scaffold_uses_validation_or_test_labels": False,
        "query_edge_excluded_from_own_gnn_context": True,
        "gnn_scaffold_summary": graph_summary,
        "threshold_selected_on": "validation only",
        "primary_threshold": threshold,
        "split_mode": CONFIG["split_mode"],
        "config": CONFIG,
    }
    with open(out_dir / "11_run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, ensure_ascii=False)

    report_lines = [
        "Edge-aware bipartite GNN TF-gene motif/TSS-hit classifier completed.",
        "",
        f"TFs: {len(tfs)}",
        f"Genes: {len(genes)}",
        f"TF-gene pairs: {len(tfs) * len(genes):,}",
        f"Positive fraction: {label_summary['positive_fraction']:.6f}",
        f"Split mode: {CONFIG['split_mode']}",
        f"Threshold strategy: {CONFIG['threshold_strategy']}",
        f"Validation-selected threshold: {threshold:.6g}",
        "",
        "Primary test metrics:",
        json.dumps(metrics["test_primary"], ensure_ascii=False),
        "",
        "Key outputs:",
        "  06b_gnn_scaffold_summary.json",
        "  07b_validation_threshold_operating_points.csv",
        "  08_metrics.json",
        "  09_test_pair_predictions.csv",
        "  10_best_all_TF_gene_motif_hit_score.csv",
        "  10b_best_all_TF_gene_motif_hit_binary.csv",
        "  10c_test_error_group_summary.csv",
        "",
        "Interpretation:",
        "  The model predicts motif/TSS-hit presence for TF-gene pairs.",
        "  GNN message passing uses positive TRAIN edges only and excludes each queried edge from its own context.",
        "  It does not predict or reconstruct any higher-level program label.",
        "  Raw E is label-source audit information only.",
    ]
    (out_dir / "12_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("")
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print("Output directory:", out_dir.resolve())
    print("Key files:")
    print("  06b_gnn_scaffold_summary.json")
    print("  07b_validation_threshold_operating_points.csv")
    print("  08_metrics.json")
    print("  09_test_pair_predictions.csv")
    print("  10_best_all_TF_gene_motif_hit_score.csv")
    print("  10b_best_all_TF_gene_motif_hit_binary.csv")
    print("  10c_test_error_group_summary.csv")
    print("  12_report.txt")


if __name__ == "__main__":
    main()
