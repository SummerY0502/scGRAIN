#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Utilities
# =============================================================================


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_names(path: Path) -> List[str]:
    return pd.read_csv(path, header=None)[0].astype(str).tolist()


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def read_table_auto(path: Path, index_col: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=index_col)
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep=",", index_col=index_col)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_id_map(path: Optional[str]) -> Dict[str, List[str]]:

    if path is None or str(path).lower() in ["", "none", "null"]:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[WARN] ID map file not found: {path}. Continue without ID mapping.")
        return {}
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 2:
        raise ValueError(f"ID map file must have at least two tab-separated columns: {path}")

    mapping: Dict[str, List[str]] = {}
    for old, new in zip(df.iloc[:, 0], df.iloc[:, 1]):
        old = str(old).strip()
        new = str(new).strip()
        if old == "" or old.lower() in ["nan", "none", "null"]:
            continue
        if new == "" or new.lower() in ["none", "nan", "null", "na"]:
            mapping[old] = []
            continue
        vals = []
        for x in new.split(","):
            x = x.strip()
            if x and x.lower() not in ["none", "nan", "null", "na"]:
                vals.append(x)
        mapping[old] = vals
    print(f"Loaded ID map entries: {len(mapping)} from {path}")
    return mapping


def map_axis_labels(labels, id_map: Dict[str, List[str]]):
    """Expand labels according to id_map; labels not in id_map are kept as-is."""
    out = []
    for x in labels:
        x = str(x)
        if x in id_map:
            mapped = id_map[x]
            # None / unmapped IDs are dropped.
            out.append(mapped)
        else:
            out.append([x])
    return out


def map_tf_gene_matrix_ids(mat: pd.DataFrame, id_map: Dict[str, List[str]], row_agg: str = "max", col_agg: str = "max") -> pd.DataFrame:

    if not id_map:
        return mat

    mat = mat.copy()
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)

    # Map rows.
    row_blocks = []
    for old_id, mapped_ids in zip(mat.index, map_axis_labels(mat.index, id_map)):
        if len(mapped_ids) == 0:
            continue
        row = mat.loc[[old_id]]
        for new_id in mapped_ids:
            tmp = row.copy()
            tmp.index = [new_id]
            row_blocks.append(tmp)
    if len(row_blocks) == 0:
        mapped = mat.iloc[0:0].copy()
    else:
        mapped = pd.concat(row_blocks, axis=0)
        if row_agg == "max":
            mapped = mapped.groupby(mapped.index).max()
        elif row_agg == "sum":
            mapped = mapped.groupby(mapped.index).sum()
        elif row_agg == "mean":
            mapped = mapped.groupby(mapped.index).mean()
        else:
            raise ValueError("row_agg must be max, sum, or mean")

    # Map columns.
    col_blocks = []
    for old_id, mapped_ids in zip(mapped.columns, map_axis_labels(mapped.columns, id_map)):
        if len(mapped_ids) == 0:
            continue
        col = mapped.loc[:, [old_id]]
        for new_id in mapped_ids:
            tmp = col.copy()
            tmp.columns = [new_id]
            col_blocks.append(tmp)
    if len(col_blocks) == 0:
        mapped2 = mapped.iloc[:, 0:0].copy()
    else:
        mapped2 = pd.concat(col_blocks, axis=1)
        if col_agg == "max":
            mapped2 = mapped2.T.groupby(level=0).max().T
        elif col_agg == "sum":
            mapped2 = mapped2.T.groupby(level=0).sum().T
        elif col_agg == "mean":
            mapped2 = mapped2.T.groupby(level=0).mean().T
        else:
            raise ValueError("col_agg must be max, sum, or mean")

    return mapped2


def transform_label(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if mode == "log1p":
        return np.log1p(np.maximum(y, 0.0)).astype(np.float32)
    if mode == "none":
        return y.astype(np.float32)
    raise ValueError("label_transform must be log1p or none")


def inverse_label_np(y: np.ndarray, mode: str, mean: float = 0.0, std: float = 1.0, standardized: bool = True) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if standardized:
        y = y * float(std) + float(mean)
    if mode == "log1p":
        return np.expm1(y).clip(min=0.0)
    return y


def regression_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    out = {
        prefix + "mse": float(mean_squared_error(y_true, y_pred)),
        prefix + "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        prefix + "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    try:
        out[prefix + "r2"] = float(r2_score(y_true, y_pred))
    except Exception:
        out[prefix + "r2"] = float("nan")
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        out[prefix + "pearson"] = float(pearsonr(y_true, y_pred)[0])
        out[prefix + "spearman"] = float(spearmanr(y_true, y_pred)[0])
    else:
        out[prefix + "pearson"] = float("nan")
        out[prefix + "spearman"] = float("nan")
    return out


def _values_at_pairs(mat: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    return mat[pairs[:, 0], pairs[:, 1]].astype(np.float64)


def fit_additive_baseline(label: np.ndarray, train_pairs: np.ndarray, n_iter: int = 50) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit y[t,p] ≈ mu + row_effect[t] + col_effect[p] using train pairs only."""
    y = np.asarray(label, dtype=np.float64)
    train_pairs = np.asarray(train_pairs, dtype=np.int64)
    obs = y[train_pairs[:, 0], train_pairs[:, 1]]
    mu = float(np.mean(obs)) if obs.size else 0.0
    row_eff = np.zeros(y.shape[0], dtype=np.float64)
    col_eff = np.zeros(y.shape[1], dtype=np.float64)

    rows_by_t = [[] for _ in range(y.shape[0])]
    cols_by_p = [[] for _ in range(y.shape[1])]
    for k, (t, p) in enumerate(train_pairs):
        rows_by_t[int(t)].append(k)
        cols_by_p[int(p)].append(k)

    tr_t = train_pairs[:, 0]
    tr_p = train_pairs[:, 1]
    for _ in range(int(n_iter)):
        for t, idx in enumerate(rows_by_t):
            if idx:
                idx = np.asarray(idx, dtype=np.int64)
                row_eff[t] = np.mean(obs[idx] - mu - col_eff[tr_p[idx]])
        # keep the decomposition identifiable
        row_center = row_eff.mean()
        row_eff -= row_center
        mu += row_center
        for p, idx in enumerate(cols_by_p):
            if idx:
                idx = np.asarray(idx, dtype=np.int64)
                col_eff[p] = np.mean(obs[idx] - mu - row_eff[tr_t[idx]])
        col_center = col_eff.mean()
        col_eff -= col_center
        mu += col_center
    return mu, row_eff, col_eff


def predict_additive_baseline(mu: float, row_eff: np.ndarray, col_eff: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    return (float(mu) + row_eff[pairs[:, 0]] + col_eff[pairs[:, 1]]).astype(np.float64)


def fit_rank1_svd_imputed_baseline(label: np.ndarray, train_pairs: np.ndarray, n_iter: int = 30) -> np.ndarray:
    y = np.asarray(label, dtype=np.float64)
    train_pairs = np.asarray(train_pairs, dtype=np.int64)
    mu, row_eff, col_eff = fit_additive_baseline(y, train_pairs, n_iter=30)
    filled = mu + row_eff[:, None] + col_eff[None, :]
    filled[train_pairs[:, 0], train_pairs[:, 1]] = y[train_pairs[:, 0], train_pairs[:, 1]]

    for _ in range(int(n_iter)):
        base = filled.mean()
        centered = filled - base
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            recon = base + s[0] * np.outer(u[:, 0], vt[0, :])
        except np.linalg.LinAlgError:
            recon = filled.copy()
        recon[train_pairs[:, 0], train_pairs[:, 1]] = y[train_pairs[:, 0], train_pairs[:, 1]]
        filled = recon
    return filled.astype(np.float64)


def compute_train_only_baselines(label: np.ndarray, train_pairs: np.ndarray, val_pairs: np.ndarray, test_pairs: np.ndarray) -> pd.DataFrame:
    """Return global/additive/rank1 baseline metrics using train pairs only."""
    label = np.asarray(label, dtype=np.float64)
    train_pairs = np.asarray(train_pairs, dtype=np.int64)
    records = []
    train_y = _values_at_pairs(label, train_pairs)
    global_mu = float(np.mean(train_y)) if train_y.size else 0.0

    mu, row_eff, col_eff = fit_additive_baseline(label, train_pairs)
    rank1_mat = fit_rank1_svd_imputed_baseline(label, train_pairs)

    eval_sets = {"val": val_pairs, "test": test_pairs}
    for split_name, pairs in eval_sets.items():
        y_true = _values_at_pairs(label, pairs)
        preds = {
            "global_mean": np.full_like(y_true, global_mu, dtype=np.float64),
            "additive_train": predict_additive_baseline(mu, row_eff, col_eff, pairs),
            "rank1_svd_train_imputed": _values_at_pairs(rank1_mat, pairs),
        }
        for baseline, y_pred in preds.items():
            rec = {"eval_split": split_name, "baseline": baseline}
            rec.update(regression_metrics_np(y_true, y_pred, prefix=""))
            rec["y_true_std"] = float(np.std(y_true))
            records.append(rec)
    return pd.DataFrame(records)


def setup_device(device_arg: str, matmul_precision: str = "high") -> torch.device:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if device == "cuda":
        torch.set_float32_matmul_precision(matmul_precision)
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    return torch.device(device)


def align_dataframe(df: pd.DataFrame, row_names: List[str], col_names: List[str], name: str) -> pd.DataFrame:
    row_names = list(map(str, row_names))
    col_names = list(map(str, col_names))
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if set(row_names).issubset(df.index) and set(col_names).issubset(df.columns):
        return df.loc[row_names, col_names]

    if df.shape == (len(row_names), len(col_names)):
        print(f"[WARN] {name}: label mismatch but shape matches; aligning by position.")
        out = df.copy()
        out.index = row_names
        out.columns = col_names
        return out

    missing_rows = list(set(row_names) - set(df.index))[:10]
    missing_cols = list(set(col_names) - set(df.columns))[:10]
    raise KeyError(
        f"{name} cannot align. shape={df.shape}, expected=({len(row_names)}, {len(col_names)}), "
        f"missing rows examples={missing_rows}, missing cols examples={missing_cols}"
    )


def orient_usages_program_by_cell(usages: pd.DataFrame, program_names: List[str], cell_names: List[str]) -> pd.DataFrame:

    program_names = list(map(str, program_names))
    cell_names = list(map(str, cell_names))

    usages = usages.copy()
    usages.index = usages.index.astype(str)
    usages.columns = usages.columns.astype(str)

    pset = set(program_names)
    cset = set(cell_names)

    score_cell_by_program = len(set(usages.index).intersection(cset)) + len(set(usages.columns).intersection(pset))
    score_program_by_cell = len(set(usages.index).intersection(pset)) + len(set(usages.columns).intersection(cset))

    if score_cell_by_program >= score_program_by_cell:
        u = usages.T
    else:
        u = usages.copy()

    u.index = u.index.astype(str)
    u.columns = u.columns.astype(str)

    if set(program_names).issubset(set(u.index)):
        u = u.loc[program_names]
    elif u.shape[0] == len(program_names):
        print("[WARN] usages_file: program labels do not match; aligning programs by position.")
        u = u.copy()
        u.index = program_names
    else:
        missing = list(set(program_names) - set(u.index))[:10]
        raise ValueError(f"usages_file cannot align programs. Missing examples: {missing}")

    if set(cell_names).issubset(set(u.columns)):
        u = u.loc[:, cell_names]
    elif u.shape[1] == len(cell_names):
        print("[WARN] usages_file: cell labels do not match; aligning cells by position.")
        u = u.copy()
        u.columns = cell_names
    else:
        common = [c for c in cell_names if c in set(u.columns)]
        if len(common) == 0:
            raise ValueError("usages_file has no overlapping cells with node feature columns.")
        print(f"[WARN] usages_file: using {len(common)} overlapping cells out of {len(cell_names)}.")
        out = pd.DataFrame(0.0, index=program_names, columns=cell_names, dtype=np.float32)
        out.loc[:, common] = u.loc[program_names, common].astype(np.float32)
        u = out

    return u.astype(np.float32)


def resolve_optional_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve an optional user path; first try as provided, then relative to base_dir."""
    if path_value is None or str(path_value).lower() in ["", "none", "null", "auto"]:
        return None
    p = Path(path_value)
    if p.exists():
        return p
    p2 = base_dir / str(path_value)
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Optional input file not found: {path_value} or {p2}")


# =============================================================================
# Data preparation inside training script
# =============================================================================


class EvidenceGraphData:
    def __init__(
        self,
        prepared_dir: str,
        split_id: int = 1,
        n_splits: int = 10,
        train_split_rate: float = 0.8,
        split_mode: str = "tf_holdout",
        seed: int = 42,
        label_transform: str = "none",
        label_standardize: bool = True,
        top_genes_per_program: int = 50,
        min_alpha: float = 0.0,
        bins: int = 16,
        tf_gene_edges_file: str = "auto",
        tf_gene_edge_attr_file: str = "auto",
        load_tf_gene_attr_to_memory: bool = False,
        usages_file: Optional[str] = None,
        use_program_activity_weight: bool = True,
        activity_weight_mode: str = "mean",
        activity_weight_power: float = 1.0,
        min_activity_weight: float = 0.1,
        min_pair_coverage: float = 0.2,
    ):
        self.prepared_dir = Path(prepared_dir)
        self.split_id = int(split_id)
        self.n_splits = int(n_splits)
        self.train_split_rate = float(train_split_rate)
        self.split_mode = str(split_mode)
        self.seed = int(seed)
        self.label_transform = str(label_transform)
        self.label_standardize = bool(label_standardize)
        self.top_genes_per_program = int(top_genes_per_program)
        self.min_alpha = float(min_alpha)
        self.bins = int(bins)
        self.tf_gene_edges_file = str(tf_gene_edges_file)
        self.tf_gene_edge_attr_file = str(tf_gene_edge_attr_file)
        self.load_tf_gene_attr_to_memory = bool(load_tf_gene_attr_to_memory)
        self.usages_file = usages_file
        self.use_program_activity_weight = bool(use_program_activity_weight)
        self.activity_weight_mode = str(activity_weight_mode)
        self.activity_weight_power = float(activity_weight_power)
        self.min_activity_weight = float(min_activity_weight)
        self.min_pair_coverage = float(min_pair_coverage)
        if not (0.0 <= self.min_pair_coverage <= 1.0):
            raise ValueError("min_pair_coverage must be within [0,1].")
        self.edge_dim = self.bins * self.bins + 3  # flattened histogram + edge_type_tf_gene + edge_type_gene_program + scalar_weight
        self.scalar_col = self.edge_dim - 1
        self.load()

    def load(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.tfs = load_names(self.prepared_dir / "tf_names.txt")
        self.genes = load_names(self.prepared_dir / "gene_names.txt")
        self.programs = load_names(self.prepared_dir / "program_names.txt")
        self.n_tf = len(self.tfs)
        self.n_gene = len(self.genes)
        self.n_program = len(self.programs)

        self.tf_to_idx = {x: i for i, x in enumerate(self.tfs)}
        self.gene_to_idx = {x: i for i, x in enumerate(self.genes)}
        self.program_to_idx = {x: i for i, x in enumerate(self.programs)}

        atac = read_tsv(self.prepared_dir / "node_atac_feature.tsv")
        rna = read_tsv(self.prepared_dir / "node_rna_feature.tsv")
        tf_node_names = [f"TF::{x}" for x in self.tfs]
        gene_node_names = [f"GENE::{x}" for x in self.genes]

        tf_atac = atac.loc[tf_node_names].values.astype(np.float32)
        tf_rna = rna.loc[tf_node_names].values.astype(np.float32)
        gene_atac = atac.loc[gene_node_names].values.astype(np.float32)
        gene_rna = rna.loc[gene_node_names].values.astype(np.float32)

        self.tf_feature = np.concatenate([tf_rna, tf_atac], axis=1).astype(np.float32)
        self.gene_feature = np.concatenate([gene_rna, gene_atac], axis=1).astype(np.float32)
        self.feature_dim = self.tf_feature.shape[1]

        alpha_df = read_tsv(self.prepared_dir / "alpha_program_gene.tsv")
        alpha_df = align_dataframe(alpha_df, self.programs, self.genes, "alpha_program_gene.tsv")
        alpha = alpha_df.values.astype(np.float32)
        alpha = np.maximum(alpha, 0.0)
        denom = alpha.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        alpha = alpha / denom
        self.alpha = alpha
        self.program_feature = (alpha @ self.gene_feature).astype(np.float32)

        # Canonical program-gene mask from prepare stage.
        # This mask defines the exact mediator gene set for every program.
        program_gene_mask_path = self.prepared_dir / "program_label_gene_mask.tsv"
        if not program_gene_mask_path.exists():
            raise FileNotFoundError(
                f"Missing canonical program gene mask: {program_gene_mask_path}. "
                "Please rerun the prepare script with --program_gene_mask_file."
            )

        program_gene_mask_df = read_tsv(program_gene_mask_path)
        program_gene_mask_df = align_dataframe(
            program_gene_mask_df,
            self.programs,
            self.genes,
            "program_label_gene_mask.tsv",
        )
        program_gene_mask = (
            program_gene_mask_df
            .fillna(0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values
            .astype(np.float32)
        )
        self.program_gene_mask = (program_gene_mask > 0).astype(np.float32)

        mask_gene_counts = self.program_gene_mask.sum(axis=1).astype(int)
        print(
            "Loaded canonical program gene mask:",
            program_gene_mask_path,
            f"gene count range={mask_gene_counts.min()}..{mask_gene_counts.max()}",
            f"mean={mask_gene_counts.mean():.2f}",
        )
        if mask_gene_counts.min() == 0:
            bad = [self.programs[i] for i, c in enumerate(mask_gene_counts) if c == 0][:20]
            raise ValueError(
                "Some programs have zero genes in program_label_gene_mask.tsv. "
                f"Examples: {bad}"
            )
        if mask_gene_counts.min() != mask_gene_counts.max():
            raise ValueError(
                "program_label_gene_mask.tsv has inconsistent gene counts across programs: "
                f"min={mask_gene_counts.min()}, max={mask_gene_counts.max()}."
            )
        if mask_gene_counts.min() != self.top_genes_per_program:
            print(
                f"[WARN] top_genes_per_program={self.top_genes_per_program}, "
                f"but program_label_gene_mask.tsv has {mask_gene_counts.min()} genes per program. "
                "Training will follow the mask, not alpha topK."
            )

        # Optional cNMF cell-program activity / usages.
        #
        # Current TF-program labels are aggregate pair-level labels rather than
        # cell-specific labels. Therefore the usages matrix is converted into a
        # program-level activity/reliability weight A(P), and each TF-program
        # subgraph sample receives w(TF, P) = A(P). This does not change the
        # evidence subgraph structure; it changes only how strongly each sample
        # contributes to the supervised and consistency losses.
        rna_cells = list(rna.columns.astype(str))
        self.program_activity_weight = np.ones(self.n_program, dtype=np.float32)
        self.usage_program_by_cell = None

        usage_path = resolve_optional_path(self.usages_file, self.prepared_dir)
        if usage_path is not None:
            usages = read_table_auto(usage_path)
            usage_pxc = orient_usages_program_by_cell(usages, self.programs, rna_cells)
            usage_arr = np.maximum(usage_pxc.values.astype(np.float32), 0.0)
            self.usage_program_by_cell = usage_arr

            if self.activity_weight_mode == "mean":
                act = usage_arr.mean(axis=1)
            elif self.activity_weight_mode == "max":
                act = usage_arr.max(axis=1)
            elif self.activity_weight_mode == "nonzero_fraction":
                act = (usage_arr > 0).mean(axis=1).astype(np.float32)
            else:
                raise ValueError("activity_weight_mode must be mean, max, or nonzero_fraction")

            act = act.astype(np.float32)
            if np.nanmean(act) > 0:
                act = act / np.nanmean(act)
            act = np.nan_to_num(act, nan=1.0, posinf=1.0, neginf=1.0)
            act = np.power(np.maximum(act, 0.0), self.activity_weight_power)
            act = np.maximum(act, self.min_activity_weight)

            if self.use_program_activity_weight:
                self.program_activity_weight = act.astype(np.float32)

            print(
                "Loaded cNMF program activity usages: "
                f"file={usage_path}, mode={self.activity_weight_mode}, "
                f"use_weight={self.use_program_activity_weight}, "
                f"weight range={self.program_activity_weight.min():.6g}..{self.program_activity_weight.max():.6g}, "
                f"mean={self.program_activity_weight.mean():.6g}"
            )
        else:
            print("No cNMF usages_file provided; program activity sample weights are all 1.")

        # Same program activity weight is applied to all TFs for a given program.
        self.sample_weight = np.tile(
            self.program_activity_weight.reshape(1, -1),
            (self.n_tf, 1)
        ).astype(np.float32)

        label_df = read_tsv(self.prepared_dir / "tf_program_score.tsv")
        label_df = align_dataframe(label_df, self.tfs, self.programs, "tf_program_score.tsv")
        self.label_raw = label_df.fillna(0).values.astype(np.float32)
        label_base = transform_label(self.label_raw, self.label_transform)

        coverage_path = self.prepared_dir / "tf_program_coverage_weight.tsv"
        if not coverage_path.exists():
            raise FileNotFoundError(
                f"Missing TF-program coverage file: {coverage_path}. "
                "Rerun the coverage-enabled prepare script."
            )
        coverage_df = read_tsv(coverage_path)
        coverage_df = align_dataframe(
            coverage_df, self.tfs, self.programs, "tf_program_coverage_weight.tsv"
        )
        coverage_df = coverage_df.apply(pd.to_numeric, errors="coerce")
        if coverage_df.isna().any().any():
            raise ValueError("tf_program_coverage_weight.tsv contains NA/non-numeric values.")
        self.tf_program_coverage = coverage_df.values.astype(np.float32)
        if not np.isfinite(self.tf_program_coverage).all():
            raise ValueError("tf_program_coverage_weight.tsv contains non-finite values.")
        if self.tf_program_coverage.min() < -1e-6 or self.tf_program_coverage.max() > 1.0 + 1e-6:
            raise ValueError(
                "TF-program coverage must be within [0,1]. "
                f"Observed range={self.tf_program_coverage.min()}..{self.tf_program_coverage.max()}"
            )
        self.tf_program_coverage = np.clip(self.tf_program_coverage, 0.0, 1.0)

        all_pairs = np.array([(i, j) for i in range(self.n_tf) for j in range(self.n_program)], dtype=np.int64)
        self.all_pairs = all_pairs

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (1 <= self.split_id <= self.n_splits):
            raise ValueError("split_id must be in [1, n_splits]")
        if not (0.0 < self.train_split_rate < 1.0):
            raise ValueError("train_split_rate must be between 0 and 1")

        if self.split_mode == "random_pair":
            shuffled_pairs = all_pairs.copy()
            rng.shuffle(shuffled_pairs)
            folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(shuffled_pairs))
            trainval_idx, test_idx = folds[self.split_id - 1]
            trainval = shuffled_pairs[trainval_idx]
            test = shuffled_pairs[test_idx]
            rng.shuffle(trainval)
            n_train = int(len(trainval) * self.train_split_rate)
            train = trainval[:n_train]
            val = trainval[n_train:]

        elif self.split_mode == "tf_holdout":
            tf_ids = np.arange(self.n_tf, dtype=np.int64)
            folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(tf_ids))
            trainval_tf_idx, test_tf_idx = folds[self.split_id - 1]
            trainval_tfs = tf_ids[trainval_tf_idx]
            test_tfs = tf_ids[test_tf_idx]
            rng.shuffle(trainval_tfs)
            n_train_tf = max(1, int(len(trainval_tfs) * self.train_split_rate))
            train_tfs = trainval_tfs[:n_train_tf]
            val_tfs = trainval_tfs[n_train_tf:]
            if len(val_tfs) == 0:
                val_tfs = train_tfs[-1:]
                train_tfs = train_tfs[:-1]
            train = np.array([(i, j) for i in train_tfs for j in range(self.n_program)], dtype=np.int64)
            val = np.array([(i, j) for i in val_tfs for j in range(self.n_program)], dtype=np.int64)
            test = np.array([(i, j) for i in test_tfs for j in range(self.n_program)], dtype=np.int64)
            rng.shuffle(train)
            rng.shuffle(val)
            rng.shuffle(test)

        elif self.split_mode == "program_holdout":
            program_ids = np.arange(self.n_program, dtype=np.int64)
            folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(program_ids))
            trainval_program_idx, test_program_idx = folds[self.split_id - 1]
            trainval_programs = program_ids[trainval_program_idx]
            test_programs = program_ids[test_program_idx]
            rng.shuffle(trainval_programs)
            n_train_program = max(1, int(len(trainval_programs) * self.train_split_rate))
            train_programs = trainval_programs[:n_train_program]
            val_programs = trainval_programs[n_train_program:]
            if len(val_programs) == 0:
                val_programs = train_programs[-1:]
                train_programs = train_programs[:-1]
            train = np.array([(i, j) for i in range(self.n_tf) for j in train_programs], dtype=np.int64)
            val = np.array([(i, j) for i in range(self.n_tf) for j in val_programs], dtype=np.int64)
            test = np.array([(i, j) for i in range(self.n_tf) for j in test_programs], dtype=np.int64)
            rng.shuffle(train)
            rng.shuffle(val)
            rng.shuffle(test)

        else:
            raise ValueError("split_mode must be random_pair, tf_holdout, or program_holdout")

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(f"Empty split generated: train={len(train)}, val={len(val)}, test={len(test)}")

        # Preserve the original split first, then apply the pair-specific coverage
        # threshold independently inside train/val/test. This avoids changing split
        # membership and ensures low-coverage pairs do not enter standardization,
        # supervised training, validation, or test metrics.
        self.train_pairs_before_coverage_filter = train.copy()
        self.val_pairs_before_coverage_filter = val.copy()
        self.test_pairs_before_coverage_filter = test.copy()

        def _coverage_filter(pairs: np.ndarray) -> np.ndarray:
            if self.min_pair_coverage <= 0:
                return pairs.copy()
            cov = self.tf_program_coverage[pairs[:, 0], pairs[:, 1]]
            return pairs[cov >= self.min_pair_coverage].copy()

        train = _coverage_filter(train)
        val = _coverage_filter(val)
        test = _coverage_filter(test)

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(
                "Coverage filtering produced an empty split: "
                f"threshold={self.min_pair_coverage}, "
                f"train={len(train)}, val={len(val)}, test={len(test)}. "
                "Lower --min_pair_coverage or change the split seed."
            )

        self.train_pairs = train
        self.val_pairs = val
        self.test_pairs = test
        eligible_mask = (
            self.tf_program_coverage[all_pairs[:, 0], all_pairs[:, 1]]
            >= self.min_pair_coverage
        )
        self.eligible_pairs = all_pairs[eligible_mask].copy()

        self.coverage_filter_stats = {
            "min_pair_coverage": float(self.min_pair_coverage),
            "full_pair_count": int(len(all_pairs)),
            "eligible_pair_count": int(len(self.eligible_pairs)),
            "eligible_fraction": float(len(self.eligible_pairs) / max(len(all_pairs), 1)),
            "split_counts_before_filter": {
                "train": int(len(self.train_pairs_before_coverage_filter)),
                "val": int(len(self.val_pairs_before_coverage_filter)),
                "test": int(len(self.test_pairs_before_coverage_filter)),
            },
            "split_counts_after_filter": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            },
        }

        if self.label_standardize:
            train_values = label_base[train[:, 0], train[:, 1]]
            self.label_mean = float(train_values.mean())
            self.label_std = float(train_values.std())
            if self.label_std < 1e-8:
                self.label_std = 1.0
            self.label_train = ((label_base - self.label_mean) / self.label_std).astype(np.float32)
        else:
            self.label_mean = 0.0
            self.label_std = 1.0
            self.label_train = label_base.astype(np.float32)

        self.tf_gene_edge_attr, self.tf_gene_edge_row = self._load_tf_gene_joint_histograms()
        self.program_gene_lists = self._build_program_gene_lists()
        self.program_templates = self._build_program_templates()

        print("Prepared causal evidence subgraph data")
        print(f"  prepared_dir: {self.prepared_dir}")
        print(f"  TF={self.n_tf}, genes={self.n_gene}, programs={self.n_program}")
        mask_gene_counts = self.program_gene_mask.sum(axis=1).astype(int)
        print(
            f"  feature_dim={self.feature_dim}, edge_dim={self.edge_dim}, "
            f"program_gene_mask_gene_count={mask_gene_counts.min()}..{mask_gene_counts.max()}, "
            f"top_genes_per_program_arg={self.top_genes_per_program}"
        )
        print(f"  split_mode={self.split_mode}")
        print(
            f"  pair coverage filter: coverage >= {self.min_pair_coverage}; "
            f"eligible={len(self.eligible_pairs)}/{len(all_pairs)} "
            f"({len(self.eligible_pairs) / max(len(all_pairs), 1):.3f})"
        )
        print(
            "  split counts before coverage filter: "
            f"train={len(self.train_pairs_before_coverage_filter)}, "
            f"val={len(self.val_pairs_before_coverage_filter)}, "
            f"test={len(self.test_pairs_before_coverage_filter)}"
        )
        print(f"  train={len(train)}, val={len(val)}, test={len(test)}, all={len(all_pairs)}")
        print(
            f"  coverage range={self.tf_program_coverage.min():.6g}.."
            f"{self.tf_program_coverage.max():.6g}, "
            f"mean={self.tf_program_coverage.mean():.6g}"
        )
        print(f"  label_raw range={self.label_raw.min():.6g}..{self.label_raw.max():.6g}")
        print(f"  label transform={self.label_transform}, standardize={self.label_standardize}, mean={self.label_mean:.6g}, std={self.label_std:.6g}")
        print(
            f"  program_activity_weight: use={self.use_program_activity_weight}, "
            f"mode={self.activity_weight_mode}, range={self.program_activity_weight.min():.6g}..{self.program_activity_weight.max():.6g}, "
            f"mean={self.program_activity_weight.mean():.6g}"
        )

    def _resolve_prepared_file(self, user_value: str, default_name: str) -> Path:
        if user_value and str(user_value).lower() not in ["", "none", "null", "auto"]:
            path = Path(user_value)
        else:
            path = self.prepared_dir / default_name
        if not path.exists():
            raise FileNotFoundError(f"Required prepared file not found: {path}")
        return path

    def _load_tf_gene_joint_histograms(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load TF-gene joint-histogram edge attributes from the prepare stage.

        The prepare script builds a complete TF x gene edge list and stores:
            tf_gene_edges.csv      columns: tf_id, gene_id, tf, gene
            tf_gene_edge_attr.npy  shape: [n_tf * n_gene, bins, bins]

        We build a row-index matrix row[t, g] -> row position in the npy file, so each
        TF-program subgraph can fetch the correct TF-gene histogram for its mediator genes.
        """
        edge_file = self._resolve_prepared_file(self.tf_gene_edges_file, "tf_gene_edges.csv")
        attr_file = self._resolve_prepared_file(self.tf_gene_edge_attr_file, "tf_gene_edge_attr.npy")

        edges = pd.read_csv(edge_file)
        edges.columns = edges.columns.astype(str)
        if self.load_tf_gene_attr_to_memory:
            arr = np.load(attr_file).astype(np.float32, copy=False)
            print("Loaded tf_gene_edge_attr.npy into RAM. This is faster but uses more host memory.")
        else:
            arr = np.load(attr_file, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[1] != self.bins or arr.shape[2] != self.bins:
            raise ValueError(
                f"tf_gene_edge_attr.npy shape {arr.shape} does not match expected [n_edges,{self.bins},{self.bins}]. "
                f"Use --bins to match the prepare-stage bins."
            )
        if len(edges) != arr.shape[0]:
            raise ValueError(f"tf_gene_edges.csv rows={len(edges)} but tf_gene_edge_attr.npy n_edges={arr.shape[0]}")

        row_index = np.full((self.n_tf, self.n_gene), -1, dtype=np.int64)
        matched = 0

        if {"tf_id", "gene_id"}.issubset(set(edges.columns)):
            for row_id, row in edges.iterrows():
                try:
                    tid = int(row["tf_id"])
                    gid = int(row["gene_id"])
                except Exception:
                    continue
                if 0 <= tid < self.n_tf and 0 <= gid < self.n_gene:
                    row_index[tid, gid] = int(row_id)
                    matched += 1
        else:
            if {"tf", "gene"}.issubset(set(edges.columns)):
                tf_col, gene_col = "tf", "gene"
            elif {"tf_gene", "gene_name"}.issubset(set(edges.columns)):
                tf_col, gene_col = "tf_gene", "gene_name"
            else:
                tf_col, gene_col = edges.columns[0], edges.columns[1]
            for row_id, row in edges.iterrows():
                tf = str(row[tf_col])
                gene = str(row[gene_col])
                if tf in self.tf_to_idx and gene in self.gene_to_idx:
                    row_index[self.tf_to_idx[tf], self.gene_to_idx[gene]] = int(row_id)
                    matched += 1

        missing = int((row_index < 0).sum())
        print(
            f"Loaded TF-gene joint-histogram edge attributes: edge_file={edge_file}, attr_file={attr_file}, "
            f"shape={arr.shape}, matched={matched}/{self.n_tf * self.n_gene}, missing={missing}"
        )
        if matched == 0:
            raise ValueError("No TF-gene joint-histogram edges matched current TF/gene names. Check prepared_dir consistency.")
        if missing > 0:
            print("[WARN] Some TF-gene histograms are missing; missing pairs will use zero histograms.")
        return arr, row_index

    def get_tf_gene_hist(self, tf_id: int, gene_id: int) -> np.ndarray:
        row = int(self.tf_gene_edge_row[tf_id, gene_id])
        if row < 0:
            return np.zeros((self.bins, self.bins), dtype=np.float32)
        h = np.asarray(self.tf_gene_edge_attr[row], dtype=np.float32)
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h

    def _build_program_gene_lists(self) -> List[np.ndarray]:
        """
        Build mediator gene lists strictly from prepared_dir/program_label_gene_mask.tsv.

        Do not re-select top genes from alpha here. Alpha is used only to order
        genes and to provide gene-program edge weights. The gene set itself is
        fixed by the label-stage mask.
        """
        if not hasattr(self, "program_gene_mask"):
            raise AttributeError(
                "self.program_gene_mask is missing. "
                "Load prepared_dir/program_label_gene_mask.tsv before building program gene lists."
            )

        lists = []
        for p in range(self.n_program):
            idx = np.where(self.program_gene_mask[p] > 0)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"Program {self.programs[p]} has no genes in program_label_gene_mask.tsv."
                )

            # Keep the exact mask gene set; sort only for deterministic order.
            idx = idx[np.argsort(-self.alpha[p, idx])]
            lists.append(idx.astype(np.int64))
        return lists

    def _build_program_templates(self) -> List[Dict[str, np.ndarray]]:
        """Precompute fixed graph structure and fixed gene-program attributes per program.

        The selected mediator genes, node layout, edge_index, edge_kind,
        edge_gene_global, and gene-program edge attributes depend only on program p.
        For each TF-program sample, __getitem__ only needs to fill the TF-specific
        TF-gene joint-histogram edge attributes.
        """
        templates: List[Dict[str, np.ndarray]] = []
        hist_dim = self.bins * self.bins
        edge_dim = self.edge_dim

        for p, gene_idx in enumerate(self.program_gene_lists):
            gene_idx = gene_idx.astype(np.int64)
            k = len(gene_idx)
            n_nodes = k + 2
            tf_node = 0
            program_node = k + 1
            gene_nodes = np.arange(1, k + 1, dtype=np.int64)

            edge_src = np.empty(4 * k, dtype=np.int64)
            edge_dst = np.empty(4 * k, dtype=np.int64)
            edge_kind = np.empty(4 * k, dtype=np.int64)
            edge_gene_global = np.empty(4 * k, dtype=np.int64)
            edge_attr_template = np.zeros((4 * k, edge_dim), dtype=np.float32)
            tf_gene_edge_pos = np.empty(2 * k, dtype=np.int64)

            for local_i, g in enumerate(gene_idx):
                gn = 1 + local_i
                base = 4 * local_i

                # Directed TF-gene edges.
                edge_src[base + 0] = tf_node
                edge_dst[base + 0] = gn
                edge_kind[base + 0] = 0
                edge_gene_global[base + 0] = g

                edge_src[base + 1] = gn
                edge_dst[base + 1] = tf_node
                edge_kind[base + 1] = 0
                edge_gene_global[base + 1] = g

                # Directed gene-program edges.
                edge_src[base + 2] = gn
                edge_dst[base + 2] = program_node
                edge_kind[base + 2] = 1
                edge_gene_global[base + 2] = g

                edge_src[base + 3] = program_node
                edge_dst[base + 3] = gn
                edge_kind[base + 3] = 1
                edge_gene_global[base + 3] = g

                tf_gene_edge_pos[2 * local_i + 0] = base + 0
                tf_gene_edge_pos[2 * local_i + 1] = base + 1

                # gene-program edge_attr:
                # histogram = 0, edge type = [0, 1], scalar = alpha(p,g).
                a_pg = float(self.alpha[p, g])
                edge_attr_template[base + 2, hist_dim + 0] = 0.0
                edge_attr_template[base + 2, hist_dim + 1] = 1.0
                edge_attr_template[base + 2, hist_dim + 2] = a_pg
                edge_attr_template[base + 3, hist_dim + 0] = 0.0
                edge_attr_template[base + 3, hist_dim + 1] = 1.0
                edge_attr_template[base + 3, hist_dim + 2] = a_pg

            templates.append({
                "gene_idx": gene_idx,
                "k": np.asarray(k, dtype=np.int64),
                "n_nodes": np.asarray(n_nodes, dtype=np.int64),
                "gene_nodes": gene_nodes,
                "program_node": np.asarray(program_node, dtype=np.int64),
                "edge_index": np.stack([edge_src, edge_dst], axis=0).astype(np.int64),
                "edge_attr_template": edge_attr_template.astype(np.float32),
                "edge_kind": edge_kind.astype(np.int64),
                "edge_gene_global": edge_gene_global.astype(np.int64),
                "tf_gene_edge_pos": tf_gene_edge_pos.astype(np.int64),
            })

        print(f"Precomputed program graph templates: {len(templates)} programs")
        return templates


class TFProgramEvidenceDataset(Dataset):
    def __init__(self, data: EvidenceGraphData, pairs: np.ndarray, augment: bool = True, edge_drop_rate: float = 0.25):
        self.data = data
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.augment = bool(augment)
        self.edge_drop_rate = float(edge_drop_rate)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        t, p = self.pairs[idx]
        t = int(t)
        p = int(p)

        tpl = self.data.program_templates[p]
        gene_idx = tpl["gene_idx"]
        n_nodes = int(tpl["n_nodes"])
        gene_nodes = tpl["gene_nodes"]
        program_node = int(tpl["program_node"])
        hist_dim = self.data.bins * self.data.bins

        # Node features still depend on the current TF and program.
        x = np.empty((n_nodes, self.data.feature_dim), dtype=np.float32)
        x[0] = self.data.tf_feature[t]
        x[gene_nodes] = self.data.gene_feature[gene_idx]
        x[program_node] = self.data.program_feature[p]

        # Edge structure is fixed for each program.
        edge_index = tpl["edge_index"]

        # gene-program edge_attr is already in the template.
        # Only TF-gene histogram attributes depend on current TF t.
        edge_attr = tpl["edge_attr_template"].copy()

        row_ids = self.data.tf_gene_edge_row[t, gene_idx]
        valid = row_ids >= 0
        hist_mat = np.zeros((len(gene_idx), hist_dim), dtype=np.float32)

        if np.any(valid):
            hist_valid = np.asarray(
                self.data.tf_gene_edge_attr[row_ids[valid]],
                dtype=np.float32,
            ).reshape(-1, hist_dim)
            hist_valid = np.nan_to_num(hist_valid, nan=0.0, posinf=0.0, neginf=0.0)
            hist_mat[valid] = hist_valid

        hist_scalar = np.clip(hist_mat.mean(axis=1), 0.0, 1.0).astype(np.float32)
        tf_gene_pos = tpl["tf_gene_edge_pos"]

        # Each mediator gene has two directed TF-gene edges: TF->gene and gene->TF.
        edge_attr[tf_gene_pos, :hist_dim] = np.repeat(hist_mat, 2, axis=0)
        edge_attr[tf_gene_pos, hist_dim + 0] = 1.0
        edge_attr[tf_gene_pos, hist_dim + 1] = 0.0
        edge_attr[tf_gene_pos, hist_dim + 2] = np.repeat(hist_scalar, 2, axis=0)

        edge_keep_aug = np.ones((edge_index.shape[1],), dtype=np.float32)
        if self.augment and self.edge_drop_rate > 0:
            keep = np.random.rand(edge_index.shape[1]) >= self.edge_drop_rate
            if keep.sum() < 2:
                keep[:] = True
            edge_keep_aug = keep.astype(np.float32)

        y = self.data.label_train[t, p].astype(np.float32)
        y_raw = self.data.label_raw[t, p].astype(np.float32)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_keep_aug": edge_keep_aug,
            "edge_kind": tpl["edge_kind"],
            "edge_gene_global": tpl["edge_gene_global"],
            "tf_idx": t,
            "program_idx": p,
            "gene_idx": gene_idx,
            "y": float(y),
            "y_raw": float(y_raw),
            "sample_weight": float(self.data.sample_weight[t, p]),
            "program_activity_weight": float(self.data.program_activity_weight[p]),
        }

def collate_graphs(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    xs = []
    eis = []
    eas = []
    keep_augs = []
    graph_ids = []
    edge_graph_ids = []
    edge_kinds = []
    edge_gene_globals = []
    tf_nodes = []
    program_nodes = []
    tf_idx = []
    program_idx = []
    y = []
    y_raw = []
    sample_weight = []
    program_activity_weight = []

    node_offset = 0
    for gi, item in enumerate(batch):
        x = item["x"]
        ei = item["edge_index"].copy()
        ea = item["edge_attr"]
        n = x.shape[0]
        e = ei.shape[1]
        ei = ei + node_offset
        xs.append(x)
        eis.append(ei)
        eas.append(ea)
        keep_augs.append(item["edge_keep_aug"])
        graph_ids.append(np.full(n, gi, dtype=np.int64))
        edge_graph_ids.append(np.full(e, gi, dtype=np.int64))
        edge_kinds.append(item["edge_kind"])
        edge_gene_globals.append(item["edge_gene_global"])
        tf_nodes.append(node_offset)
        program_nodes.append(node_offset + n - 1)
        tf_idx.append(item["tf_idx"])
        program_idx.append(item["program_idx"])
        y.append(item["y"])
        y_raw.append(item["y_raw"])
        sample_weight.append(item["sample_weight"])
        program_activity_weight.append(item["program_activity_weight"])
        node_offset += n

    out = {
        "x": torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32),
        "edge_index": torch.tensor(np.concatenate(eis, axis=1), dtype=torch.long),
        "edge_attr": torch.tensor(np.concatenate(eas, axis=0), dtype=torch.float32),
        "edge_keep_aug": torch.tensor(np.concatenate(keep_augs, axis=0), dtype=torch.float32),
        "graph_id": torch.tensor(np.concatenate(graph_ids, axis=0), dtype=torch.long),
        "edge_graph_id": torch.tensor(np.concatenate(edge_graph_ids, axis=0), dtype=torch.long),
        "edge_kind": torch.tensor(np.concatenate(edge_kinds, axis=0), dtype=torch.long),
        "edge_gene_global": torch.tensor(np.concatenate(edge_gene_globals, axis=0), dtype=torch.long),
        "tf_node": torch.tensor(tf_nodes, dtype=torch.long),
        "program_node": torch.tensor(program_nodes, dtype=torch.long),
        "tf_idx": torch.as_tensor(np.asarray(tf_idx, dtype=np.int64), dtype=torch.long),
        "program_idx": torch.as_tensor(np.asarray(program_idx, dtype=np.int64), dtype=torch.long),
        "y": torch.as_tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32),
        "y_raw": torch.as_tensor(np.asarray(y_raw, dtype=np.float32), dtype=torch.float32),
        "sample_weight": torch.as_tensor(np.asarray(sample_weight, dtype=np.float32), dtype=torch.float32),
        "program_activity_weight": torch.as_tensor(np.asarray(program_activity_weight, dtype=np.float32), dtype=torch.float32),
        "num_graphs": len(batch),
    }
    return out


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def batch_num_graphs(batch: Dict[str, torch.Tensor]) -> int:
    n = batch["num_graphs"]
    return int(n.item()) if torch.is_tensor(n) else int(n)


# =============================================================================
# Model
# =============================================================================


class EdgeAwareMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        e = self.edge_encoder(edge_attr)
        msg = self.msg_mlp(torch.cat([h[src], e], dim=-1))
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        if edge_weight is None:
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        else:
            deg.index_add_(0, dst, edge_weight.to(dtype=h.dtype))
        agg = agg / deg.clamp(min=1.0).view(-1, 1)
        out = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + out)


class EvidenceSubgraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            EdgeAwareMessageLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, edge_weight=edge_weight)
        return h


class EdgeMaskDecoder(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        # Clamp logits before sigmoid to avoid numerical overflow/NaN in CUDA losses.
        logits = self.net(torch.cat([z[src], z[dst], edge_attr], dim=-1)).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=15.0, neginf=-15.0).clamp(min=-15.0, max=15.0)
        mask = torch.sigmoid(logits)
        return mask.clamp(min=1e-6, max=1.0 - 1e-6)


class CausalEvidenceSubgraphRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 3,
        num_layers: int = 2,
        predictor_layers: int = 2,
        dropout: float = 0.1,
        mixup_lambda: float = 0.5,
        use_reparameterization: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.mixup_lambda = float(mixup_lambda)
        self.use_reparameterization = bool(use_reparameterization)

        self.encoder = EvidenceSubgraphEncoder(input_dim, hidden_dim, edge_dim, num_layers, dropout)
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)
        self.mask_decoder = EdgeMaskDecoder(hidden_dim, edge_dim, dropout)

        self.predictor_gnn = nn.ModuleList([
            EdgeAwareMessageLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(predictor_layers)
        ])
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def graph_mean_pool(h: torch.Tensor, graph_id: torch.Tensor, num_graphs: int) -> torch.Tensor:
        out = torch.zeros((num_graphs, h.shape[1]), dtype=h.dtype, device=h.device)
        out.index_add_(0, graph_id, h)
        cnt = torch.zeros(num_graphs, dtype=h.dtype, device=h.device)
        cnt.index_add_(0, graph_id, torch.ones(h.shape[0], dtype=h.dtype, device=h.device))
        return out / cnt.clamp(min=1.0).view(-1, 1)

    def encode_fused(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        keep_aug = batch["edge_keep_aug"]

        z = self.encoder(x, edge_index, edge_attr, edge_weight=None)
        z_aug = self.encoder(x, edge_index, edge_attr, edge_weight=keep_aug)
        z_mix = self.mixup_lambda * z + (1.0 - self.mixup_lambda) * z_aug

        mu = self.mu(z_mix)
        logvar = self.logvar(z_mix).clamp(min=-8.0, max=4.0)
        if self.training and self.use_reparameterization:
            eps = torch.randn_like(mu)
            z_latent = mu + torch.exp(0.5 * logvar) * eps
        else:
            z_latent = mu
        return z, z_aug, z_latent, mu, logvar

    def predict_from_masked_graph(self, z_latent: torch.Tensor, batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        h = z_latent
        for layer in self.predictor_gnn:
            h = layer(h, edge_index, edge_attr, edge_weight=mask)

        num_graphs = batch_num_graphs(batch)
        graph_pool = self.graph_mean_pool(h, batch["graph_id"], num_graphs)
        tf_h = h[batch["tf_node"]]
        program_h = h[batch["program_node"]]
        prod = tf_h * program_h
        pred = self.score_mlp(torch.cat([graph_pool, tf_h, program_h, prod], dim=-1)).squeeze(-1)
        return pred

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z, z_aug, z_latent, mu, logvar = self.encode_fused(batch)
        mask = self.mask_decoder(z_latent, batch["edge_index"], batch["edge_attr"])
        pred = self.predict_from_masked_graph(z_latent, batch, mask)

        pred_aug = self.predict_from_masked_graph(z_latent, batch, mask * batch["edge_keep_aug"])
        return {
            "pred": pred,
            "pred_aug": pred_aug,
            "mask": mask,
            "z": z,
            "z_aug": z_aug,
            "z_latent": z_latent,
            "mu": mu,
            "logvar": logvar,
        }


# =============================================================================
# Loss, train, evaluate
# =============================================================================


def mask_entropy(mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = mask.clamp(min=eps, max=1.0 - eps)
    return -(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m)).mean()


def compute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|G) || N(0,I)); averaged over all nodes and latent dimensions."""
    mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0)
    logvar = torch.nan_to_num(logvar, nan=0.0, posinf=4.0, neginf=-8.0).clamp(min=-8.0, max=4.0)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def select_val_key(metrics: Dict[str, float], metric_name: str) -> float:
    """Convert validation metric to a minimization key for early stopping."""
    v = float(metrics.get(metric_name, float("nan")))
    if not np.isfinite(v):
        return float("inf")
    if metric_name in ["ces_r2", "ces_pearson", "ces_spearman"]:
        return -v
    return v


def compute_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    return_parts: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    y = batch["y"]
    pred = out["pred"]
    pred_aug = out["pred_aug"]
    mask = out["mask"]

    sample_weight = batch.get("sample_weight", None)

    if args.loss == "mse":
        loss_vec = (pred - y).pow(2)
    elif args.loss == "smooth_l1":
        loss_vec = F.smooth_l1_loss(pred, y, reduction="none")
    else:
        raise ValueError("loss must be mse or smooth_l1")

    use_weight = bool(getattr(args, "use_program_activity_weight", False)) and sample_weight is not None
    if use_weight:
        w = torch.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0).to(dtype=loss_vec.dtype)
        w = w.clamp(min=1e-8)
        loss_score = (loss_vec * w).sum() / w.sum().clamp(min=1e-8)
    else:
        w = None
        loss_score = loss_vec.mean()

    mask_safe = torch.nan_to_num(
        mask,
        nan=0.5,
        posinf=1.0 - 1e-6,
        neginf=1e-6,
    ).clamp(1e-6, 1.0 - 1e-6)

    cons_vec = (pred_aug - pred.detach()).pow(2)
    if use_weight:
        loss_consistency = (cons_vec * w).sum() / w.sum().clamp(min=1e-8)
    else:
        loss_consistency = cons_vec.mean()

    loss_sparsity = mask_safe.mean()
    loss_entropy = mask_entropy(mask_safe)

    edge_weight_scalar = torch.nan_to_num(
        batch["edge_attr"][:, -1],
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(min=0.0, max=1.0)
    loss_prior = F.mse_loss(mask_safe, edge_weight_scalar)

    mu = out.get("mu", None)
    logvar = out.get("logvar", None)
    if mu is not None and logvar is not None and getattr(args, "lambda_kl", 0.0) > 0:
        loss_kl = compute_kl_loss(mu, logvar)
    else:
        loss_kl = torch.zeros((), dtype=pred.dtype, device=pred.device)

    kl_scale = 1.0
    warmup_epochs = int(getattr(args, "kl_warmup_epochs", 0))
    if warmup_epochs > 0:
        current_epoch = int(getattr(args, "current_epoch", 0))
        kl_scale = min(1.0, max(0.0, float(current_epoch) / float(warmup_epochs)))

    loss = (
        args.lambda_score * loss_score
        + args.lambda_consistency * loss_consistency
        + args.lambda_sparsity * loss_sparsity
        + args.lambda_entropy * loss_entropy
        + args.lambda_prior * loss_prior
        + args.lambda_kl * kl_scale * loss_kl
    )

    if not return_parts:
        return loss, None

    # Keep these on GPU during training. Converting to CPU/float per batch
    # forces CUDA synchronization and lowers GPU utilization.
    parts: Dict[str, torch.Tensor] = {
        "loss_total": loss.detach(),
        "loss_score": loss_score.detach(),
        "loss_consistency": loss_consistency.detach(),
        "loss_sparsity": loss_sparsity.detach(),
        "loss_entropy": loss_entropy.detach(),
        "loss_prior": loss_prior.detach(),
        "loss_kl": loss_kl.detach(),
        "kl_scale": torch.as_tensor(kl_scale, dtype=loss.dtype, device=loss.device).detach(),
        "mask_mean": mask_safe.mean().detach(),
    }
    if sample_weight is not None:
        sw = torch.nan_to_num(sample_weight.detach(), nan=1.0, posinf=1.0, neginf=1.0)
        parts["sample_weight_mean"] = sw.mean().detach()
        parts["sample_weight_min"] = sw.min().detach()
        parts["sample_weight_max"] = sw.max().detach()

    return loss, parts

def run_epoch(model, loader, optimizer, device, args) -> Dict[str, float]:
    model.train()
    sums: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss, parts = compute_loss(out, batch, args, return_parts=True)

        # Do not check torch.isfinite(loss) every batch in normal training,
        # because a Python if on a CUDA tensor forces synchronization.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        assert parts is not None
        for k, v in parts.items():
            if k not in sums:
                sums[k] = torch.zeros((), dtype=v.dtype, device=v.device)
            sums[k] += v
        n_batches += 1

    return {k: (v / max(n_batches, 1)).item() for k, v in sums.items()}

@torch.no_grad()
def evaluate(model, loader, data: EvidenceGraphData, device, args) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    # 保存原始模式，并在退出时恢复（防止副作用）
    training = model.training
    model.eval()   # 强制进入评估模式

    pred_mat_train_space = np.full((data.n_tf, data.n_program), np.nan, dtype=np.float32)
    raw_label_mat = data.label_raw.copy()
    mask_sum = np.zeros((data.n_tf, data.n_program), dtype=np.float32)
    mask_count = np.zeros((data.n_tf, data.n_program), dtype=np.float32)

    y_true_raw_all = []
    y_pred_raw_all = []
    loss_parts_sum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        _, parts = compute_loss(out, batch, args, return_parts=True)
        assert parts is not None
        pred = out["pred"].detach().cpu().numpy()
        tf_idx = batch["tf_idx"].detach().cpu().numpy()
        program_idx = batch["program_idx"].detach().cpu().numpy()
        for val, t, p in zip(pred, tf_idx, program_idx):
            pred_mat_train_space[t, p] = val

        pred_raw = inverse_label_np(pred, data.label_transform, data.label_mean, data.label_std, data.label_standardize)
        y_raw = batch["y_raw"].detach().cpu().numpy()
        y_true_raw_all.append(y_raw)
        y_pred_raw_all.append(pred_raw)

        mask = out["mask"].detach()
        edge_graph = batch["edge_graph_id"]
        for gi in range(batch_num_graphs(batch)):
            m = mask[edge_graph == gi]
            if m.numel() == 0:
                continue
            t = int(tf_idx[gi])
            p = int(program_idx[gi])
            mask_sum[t, p] += float(m.mean().cpu())
            mask_count[t, p] += 1.0

        for k, v in parts.items():
            loss_parts_sum[k] = loss_parts_sum.get(k, 0.0) + float(v.detach().cpu())
        n_batches += 1

    y_true_raw_all = np.concatenate(y_true_raw_all)
    y_pred_raw_all = np.concatenate(y_pred_raw_all)
    metrics = regression_metrics_np(y_true_raw_all, y_pred_raw_all, prefix="ces_")
    for k, v in loss_parts_sum.items():
        metrics[k] = v / max(n_batches, 1)

    mask_mean_mat = mask_sum / np.maximum(mask_count, 1.0)

    if training:
        model.train()

    return metrics, pred_mat_train_space, raw_label_mat, mask_mean_mat


@torch.no_grad()
def predict_all_matrix(model, data: EvidenceGraphData, device, args) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ds = TFProgramEvidenceDataset(data, data.all_pairs, augment=False, edge_drop_rate=0.0)
    loader_kwargs = {
        "num_workers": getattr(args, "num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    if getattr(args, "num_workers", 0) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 4)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graphs, **loader_kwargs)
    model.eval()
    pred_train_space = np.zeros((data.n_tf, data.n_program), dtype=np.float32)
    mask_mean = np.zeros((data.n_tf, data.n_program), dtype=np.float32)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        pred = out["pred"].detach().cpu().numpy()
        tf_idx = batch["tf_idx"].detach().cpu().numpy()
        program_idx = batch["program_idx"].detach().cpu().numpy()
        mask = out["mask"].detach()
        edge_graph = batch["edge_graph_id"]
        for gi, (val, t, p) in enumerate(zip(pred, tf_idx, program_idx)):
            pred_train_space[t, p] = val
            m = mask[edge_graph == gi]
            mask_mean[t, p] = float(m.mean().cpu()) if m.numel() > 0 else 0.0
    pred_raw = inverse_label_np(pred_train_space, data.label_transform, data.label_mean, data.label_std, data.label_standardize)
    return pred_raw, mask_mean


@torch.no_grad()
def collect_top_evidence_paths(model, data: EvidenceGraphData, device, args, pred_raw: np.ndarray) -> pd.DataFrame:
    # Collect masks for top predicted TF-program pairs by program.
    records = []
    pairs = []
    for p in range(data.n_program):
        top_tfs = np.argsort(-pred_raw[:, p])[: args.export_top_tfs_per_program]
        for t in top_tfs:
            pairs.append((int(t), int(p)))
    pairs = np.array(pairs, dtype=np.int64)
    ds = TFProgramEvidenceDataset(data, pairs, augment=False, edge_drop_rate=0.0)
    loader_kwargs = {
        "num_workers": getattr(args, "num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    if getattr(args, "num_workers", 0) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 4)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graphs, **loader_kwargs)
    model.eval()

    # Because each gene has four directed edges, combine masks by undirected evidence type and gene.
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        mask = out["mask"].detach().cpu().numpy()
        edge_graph = batch["edge_graph_id"].detach().cpu().numpy()
        edge_kind = batch["edge_kind"].detach().cpu().numpy()
        edge_gene_global = batch["edge_gene_global"].detach().cpu().numpy()
        tf_idx = batch["tf_idx"].detach().cpu().numpy()
        program_idx = batch["program_idx"].detach().cpu().numpy()
        edge_attr = batch["edge_attr"].detach().cpu().numpy()

        for gi in range(batch_num_graphs(batch)):
            t = int(tf_idx[gi])
            p = int(program_idx[gi])
            sel = np.where(edge_graph == gi)[0]
            tmp = []
            for eidx in sel:
                g = int(edge_gene_global[eidx])
                kind = int(edge_kind[eidx])
                tmp.append((g, kind, float(mask[eidx]), float(edge_attr[eidx, -1])))
            # Aggregate by gene and edge kind.
            agg: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
            for g, kind, m, w in tmp:
                agg.setdefault((g, kind), []).append((m, w))
            by_gene: Dict[int, Dict[str, float]] = {}
            for (g, kind), vals in agg.items():
                ms = [v[0] for v in vals]
                ws = [v[1] for v in vals]
                if g not in by_gene:
                    by_gene[g] = {}
                if kind == 0:
                    by_gene[g]["tf_gene_mask"] = float(np.mean(ms))
                    by_gene[g]["tf_gene_weight"] = float(np.mean(ws))
                else:
                    by_gene[g]["gene_program_mask"] = float(np.mean(ms))
                    by_gene[g]["alpha_weight"] = float(np.mean(ws))
            rows = []
            for g, d in by_gene.items():
                tfm = d.get("tf_gene_mask", 0.0)
                gpm = d.get("gene_program_mask", 0.0)
                rows.append((g, tfm, gpm, tfm * gpm, d.get("tf_gene_weight", 0.0), d.get("alpha_weight", 0.0)))
            rows = sorted(rows, key=lambda x: x[3], reverse=True)[: args.export_top_genes_per_pair]
            for rank, (g, tfm, gpm, joint, tfw, alpha_w) in enumerate(rows, start=1):
                records.append({
                    "program": data.programs[p],
                    "tf": data.tfs[t],
                    "tf_program_pred_score": float(pred_raw[t, p]),
                    "rank_gene_support": rank,
                    "gene": data.genes[g],
                    "tf_gene_mask": tfm,
                    "gene_program_mask": gpm,
                    "joint_path_mask": joint,
                    "tf_gene_joint_hist_mean": tfw,
                    "alpha_program_gene_weight": alpha_w,
                })
    return pd.DataFrame(records)


# =============================================================================
# Output helpers
# =============================================================================


def save_rankings(out_dir: Path, data: EvidenceGraphData, pred_raw: np.ndarray, label_raw: np.ndarray, mask_mean: np.ndarray, top_n: int = 30) -> None:
    pred_df = pd.DataFrame(pred_raw, index=data.tfs, columns=data.programs)
    pred_df.to_csv(out_dir / "best_all_TF_program_score_ces.csv")

    mask_df = pd.DataFrame(mask_mean, index=data.tfs, columns=data.programs)
    mask_df.to_csv(out_dir / "best_all_TF_program_mean_edge_mask.csv")

    records = []
    for p, program in enumerate(data.programs):
        s = pred_df[program].sort_values(ascending=False).head(top_n)
        for rank, (tf, score) in enumerate(s.items(), start=1):
            records.append({"program": program, "rank": rank, "tf": tf, "ces_score": float(score), "mean_edge_mask": float(mask_df.loc[tf, program])})
    pd.DataFrame(records).to_csv(out_dir / f"top{top_n}_TF_per_program_by_ces_score.csv", index=False)

    # Compare ranking to label/raw prior matrix, not to old model.
    label_df = pd.DataFrame(label_raw, index=data.tfs, columns=data.programs)
    comp = []
    for program in data.programs:
        pred_rank = pred_df[program].rank(ascending=False, method="average")
        label_rank = label_df[program].rank(ascending=False, method="average")
        pred_order = pred_df[program].sort_values(ascending=False).index.tolist()
        label_order = label_df[program].sort_values(ascending=False).index.tolist()
        rec = {
            "program": program,
            "pred_vs_label_exactly_same_order": pred_order == label_order,
            "pred_top1": pred_order[0],
            "label_top1": label_order[0],
            "top1_same": pred_order[0] == label_order[0],
            "spearman_rank_corr": float(spearmanr(pred_rank, label_rank).correlation),
        }
        for k in [10, 30, 50]:
            kk = min(k, len(pred_order))
            a = set(pred_order[:kk])
            b = set(label_order[:kk])
            rec[f"top{k}_overlap_fraction"] = len(a & b) / kk
            rec[f"top{k}_jaccard"] = len(a & b) / len(a | b)
        comp.append(rec)
    pd.DataFrame(comp).to_csv(out_dir / "ces_prediction_vs_label_ranking_comparison_by_program.csv", index=False)


def save_pair_coverage_outputs(
    out_dir: Path,
    data: EvidenceGraphData,
    pred_raw: np.ndarray,
) -> None:
    """Save included and excluded pairs with coverage, labels, and predictions."""
    included_records = []
    for split_name, pairs in [
        ("train", data.train_pairs),
        ("val", data.val_pairs),
        ("test", data.test_pairs),
    ]:
        for t, p in np.asarray(pairs, dtype=np.int64):
            included_records.append({
                "split": split_name,
                "TF": data.tfs[int(t)],
                "program": data.programs[int(p)],
                "label": float(data.label_raw[int(t), int(p)]),
                "prediction": float(pred_raw[int(t), int(p)]),
                "coverage": float(data.tf_program_coverage[int(t), int(p)]),
                "included": True,
            })
    pd.DataFrame(included_records).to_csv(
        out_dir / "pair_predictions.csv", index=False
    )

    excluded_records = []
    for split_name, pairs in [
        ("train", data.train_pairs_before_coverage_filter),
        ("val", data.val_pairs_before_coverage_filter),
        ("test", data.test_pairs_before_coverage_filter),
    ]:
        for t, p in np.asarray(pairs, dtype=np.int64):
            cov = float(data.tf_program_coverage[int(t), int(p)])
            if cov >= data.min_pair_coverage:
                continue
            excluded_records.append({
                "split": split_name,
                "TF": data.tfs[int(t)],
                "program": data.programs[int(p)],
                "label": float(data.label_raw[int(t), int(p)]),
                "prediction": float(pred_raw[int(t), int(p)]),
                "coverage": cov,
                "exclusion_reason": f"coverage < {data.min_pair_coverage}",
            })
    pd.DataFrame(excluded_records).to_csv(
        out_dir / "excluded_low_coverage_pairs.csv", index=False
    )

    coverage_df = pd.DataFrame(
        data.tf_program_coverage, index=data.tfs, columns=data.programs
    )
    coverage_df.to_csv(out_dir / "tf_program_coverage_matrix.tsv", sep="\t")

    confident_pred = pd.DataFrame(
        pred_raw.copy(), index=data.tfs, columns=data.programs
    )
    confident_pred.values[data.tf_program_coverage < data.min_pair_coverage] = np.nan
    confident_pred.to_csv(
        out_dir / "best_TF_program_score_coverage_filtered.tsv", sep="\t"
    )


# =============================================================================
# Main
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TFPG causal evidence subgraph training")

    # One-click defaults matching your current workflow.
    p.add_argument("--prepared_dir", type=str, default="data/output_top50_bg_centered_Parenchyma_cell")   #"output_top20_program_genes"   output_top20_program_genes_Parenchyma_cell
    p.add_argument("--out_dir", type=str, default="data/tf_program_ces_top50_bg_centered_cov_fix13")   #tf_program_causal_evidence_subgraph_joint_hist_logs20_Parenchyma_cell
    p.add_argument("--split_id", type=int, default=1)
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--train_split_rate", type=float, default=0.8)
    p.add_argument("--split_mode", type=str, default="random_pair", choices=["random_pair", "tf_holdout", "program_holdout"],
                   help="Evaluation split. tf_holdout is recommended as the main setting after bg-centering; random_pair is only an interpolation control.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min_pair_coverage",
        type=float,
        default=0.2,
        help=(
            "After generating the original train/val/test split, keep only TF-program "
            "pairs with coverage >= this threshold in each split. Use 0 to disable."
        ),
    )

    p.add_argument("--top_genes_per_program", type=int, default=50)
    p.add_argument("--min_alpha", type=float, default=0.0)
    p.add_argument("--bins", type=int, default=16,
                   help="Histogram bin number used in prepared_dir/tf_gene_edge_attr.npy.")
    p.add_argument("--tf_gene_edges_file", type=str, default="auto",
                   help="Path to tf_gene_edges.csv. Default: prepared_dir/tf_gene_edges.csv")
    p.add_argument("--tf_gene_edge_attr_file", type=str, default="auto",
                   help="Path to tf_gene_edge_attr.npy. Default: prepared_dir/tf_gene_edge_attr.npy")

    p.add_argument("--usages_file", type=str, default="RNA_scRNA_Parenchyma_cell.usages.k_100.dt_0_5.consensus.txt",
                   help="Optional cNMF usages matrix, usually cell × program. If provided, program activity weights each TF-program subgraph loss.")
    p.add_argument("--use_program_activity_weight", action="store_true", default=True,
                   help="Use cNMF program activity as TF-program sample weight when --usages_file is provided.")
    p.add_argument("--no_program_activity_weight", dest="use_program_activity_weight", action="store_false",
                   help="Load usages_file if provided but do not apply activity weights to losses.")
    p.add_argument("--activity_weight_mode", type=str, default="mean",
                   choices=["mean", "max", "nonzero_fraction"],
                   help="How to summarize program × cell usages into one activity weight per program.")
    p.add_argument("--activity_weight_power", type=float, default=1.0,
                   help="Power transform applied after mean-normalizing program activity weights.")
    p.add_argument("--min_activity_weight", type=float, default=0.1,
                   help="Lower bound for program activity sample weights.")

    p.add_argument("--label_transform", type=str, default="none", choices=["log1p", "none"],
                   help="Use none for signed bg-centered labels. Do not use log1p unless labels are non-negative.")
    p.add_argument("--label_standardize", action="store_true", default=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--matmul_precision", type=str, default="high")
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader workers. Use 0 if Windows multiprocessing causes issues.")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="DataLoader prefetch factor when num_workers > 0.")
    p.add_argument("--load_hist_to_memory", action="store_true", default=False,
                   help="Load tf_gene_edge_attr.npy fully into RAM instead of mmap. Faster if host RAM is enough.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--predictor_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--mixup_lambda", type=float, default=0.5)
    p.add_argument("--use_reparameterization", action="store_true", default=True)
    p.add_argument("--no_reparameterization", dest="use_reparameterization", action="store_false",
                   help="Disable stochastic latent sampling and use mu directly.")
    p.add_argument("--edge_drop_rate", type=float, default=0.15)

    p.add_argument("--loss", type=str, default="mse", choices=["mse", "smooth_l1"],
                   help="smooth_l1 is the default for small-range signed bg-centered residual labels.")
    p.add_argument("--lambda_score", type=float, default=1.0)
    p.add_argument("--lambda_consistency", type=float, default=0.0)
    p.add_argument("--lambda_sparsity", type=float, default=0.0)
    p.add_argument("--lambda_entropy", type=float, default=0.0)
    p.add_argument("--lambda_prior", type=float, default=0.0)
    p.add_argument("--lambda_kl", type=float, default=0,
                   help="KL regularization weight for the variational latent state. Start with 1e-5 to 1e-4.")
    p.add_argument("--kl_warmup_epochs", type=int, default=50,
                   help="Linearly warm up KL weight over this many epochs. Use 0 to disable warmup.")
    p.add_argument("--val_metric", type=str, default="ces_r2",
                   choices=["loss_total", "ces_rmse", "ces_mae", "ces_r2", "ces_pearson", "ces_spearman"],
                   help="Validation metric used for best-model selection / early stopping.")

    p.add_argument("--export_top_tfs_per_program", type=int, default=50)
    p.add_argument("--export_top_genes_per_pair", type=int, default=10)

    return p



# =============================================================================
# Standalone best-model inference configuration
# =============================================================================
#
# This file is SELF-CONTAINED:
#   - it DOES NOT import the original training .py file;
#   - model classes, graph construction, preprocessing and output functions
#     are already included above in this file.
#
# Before right-click Run, only edit the paths below.
# =============================================================================

# Best checkpoint produced by training.
CHECKPOINT = r"data\tf_program_ces_top50_bg_centered_cov_fix13\best_model.pt"

# Same prepared_dir used by the training run.
PREPARED_DIR = r"data\output_top50_bg_centered_Parenchyma_cell"

# New directory for the reproduced inference outputs.
OUT_DIR = r"tf_program_inference"

# "cuda", "cpu", or "auto".
DEVICE = "cuda"

# Runtime-only settings. These do not change the model prediction definition.
NUM_WORKERS = 0

# None -> use eval_batch_size saved in checkpoint args.
EVAL_BATCH_SIZE = None

# True may be faster but consumes more host RAM.
LOAD_HIST_TO_MEMORY = False

# Optional original training-output directory.
# If present, the script compares the newly reproduced score/mask matrices
# against the original training outputs and also reads history.csv when needed
# to reproduce KL-warmup-scaled final loss_total.
#
# Set to None if you do not want comparison.
REFERENCE_OUT_DIR = r"tf_program_ces_top50_bg_centered_cov_fix13"

# Usually None.
# Only relevant to loss_total when lambda_kl > 0 and KL warmup was not saturated.
FINAL_EPOCH = None

# Whether to copy best_model.pt into OUT_DIR.
COPY_CHECKPOINT = False


def _compare_matrix_csv(produced_file: Path, reference_file: Path) -> Dict[str, object]:
    """Numerically compare two TF x Program matrix CSV files."""
    if not reference_file.exists():
        return {"reference_exists": False}

    a = pd.read_csv(produced_file, index_col=0)
    b = pd.read_csv(reference_file, index_col=0)

    result: Dict[str, object] = {
        "reference_exists": True,
        "same_shape": a.shape == b.shape,
        "same_index": list(map(str, a.index)) == list(map(str, b.index)),
        "same_columns": list(map(str, a.columns)) == list(map(str, b.columns)),
    }

    if not (
        result["same_shape"]
        and result["same_index"]
        and result["same_columns"]
    ):
        result["allclose"] = False
        return result

    av = a.to_numpy(dtype=np.float64)
    bv = b.to_numpy(dtype=np.float64)

    result["allclose"] = bool(
        np.allclose(av, bv, rtol=1e-6, atol=1e-7, equal_nan=True)
    )
    diff = np.abs(av - bv)
    finite = np.isfinite(diff)
    result["max_abs_diff"] = float(diff[finite].max()) if finite.any() else 0.0
    return result


def _get_saved_arg(saved: Dict[str, object], key: str, default):
    return saved[key] if key in saved else default


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Resolve paths
    # -------------------------------------------------------------------------
    checkpoint_path = Path(CHECKPOINT).expanduser().resolve()
    prepared_dir = Path(PREPARED_DIR).expanduser().resolve()
    out_dir = Path(OUT_DIR).expanduser().resolve()

    reference_out_dir = (
        None
        if REFERENCE_OUT_DIR is None
        else Path(REFERENCE_OUT_DIR).expanduser().resolve()
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nCannot find best_model.pt：\n"
            f"{checkpoint_path}\n\n"
            "Please modify the CHECKPOINT at the top of this file."
        )

    if not prepared_dir.exists():
        raise FileNotFoundError(
            "\n找不到 prepared_dir：\n"
            f"{prepared_dir}\n\n"
            "请修改本文件顶部 PREPARED_DIR。"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Load checkpoint
    # -------------------------------------------------------------------------
    print("CHECKPOINT:", checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint：{type(ckpt)}")
    if "model" not in ckpt:
        raise KeyError("checkpoint")
    if "args" not in ckpt or not isinstance(ckpt["args"], dict):
        raise KeyError(
            "checkpoint"
            "{'model': ..., 'args': vars(args), ...}。"
        )

    saved_args = dict(ckpt["args"])

    # Build Namespace directly from the parameters actually saved with best_model.pt.
    args = argparse.Namespace(**saved_args)

    # Compatibility defaults for older checkpoints.
    defaults = {
        "split_id": 1,
        "n_splits": 10,
        "train_split_rate": 0.8,
        "split_mode": "random_pair",
        "seed": 42,
        "min_pair_coverage": 0.2,
        "top_genes_per_program": 50,
        "min_alpha": 0.0,
        "bins": 16,
        "tf_gene_edges_file": "auto",
        "tf_gene_edge_attr_file": "auto",
        "usages_file": None,
        "use_program_activity_weight": True,
        "activity_weight_mode": "mean",
        "activity_weight_power": 1.0,
        "min_activity_weight": 0.1,
        "label_transform": "none",
        "label_standardize": True,
        "matmul_precision": "high",
        "eval_batch_size": 256,
        "num_workers": 0,
        "prefetch_factor": 4,
        "load_hist_to_memory": False,
        "hidden_dim": 256,
        "num_layers": 2,
        "predictor_layers": 2,
        "dropout": 0.3,
        "mixup_lambda": 0.5,
        "use_reparameterization": True,
        "loss": "mse",
        "lambda_score": 1.0,
        "lambda_consistency": 0.0,
        "lambda_sparsity": 0.0,
        "lambda_entropy": 0.0,
        "lambda_prior": 0.0,
        "lambda_kl": 0.0,
        "kl_warmup_epochs": 50,
        "export_top_tfs_per_program": 50,
        "export_top_genes_per_pair": 10,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Runtime/path overrides only.
    args.prepared_dir = str(prepared_dir)
    args.out_dir = str(out_dir)
    args.device = DEVICE
    args.num_workers = int(NUM_WORKERS)
    args.load_hist_to_memory = bool(LOAD_HIST_TO_MEMORY)

    if EVAL_BATCH_SIZE is not None:
        args.eval_batch_size = int(EVAL_BATCH_SIZE)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动改为 CPU。")
        args.device = "cpu"

    seed_everything(int(args.seed))
    device = setup_device(args.device, args.matmul_precision)

    # -------------------------------------------------------------------------
    # 3. Reconstruct EXACT training-time data/preprocessing
    # -------------------------------------------------------------------------

    data = EvidenceGraphData(
        prepared_dir=args.prepared_dir,
        split_id=args.split_id,
        n_splits=args.n_splits,
        train_split_rate=args.train_split_rate,
        split_mode=args.split_mode,
        seed=args.seed,
        label_transform=args.label_transform,
        label_standardize=args.label_standardize,
        top_genes_per_program=args.top_genes_per_program,
        min_alpha=args.min_alpha,
        bins=args.bins,
        tf_gene_edges_file=args.tf_gene_edges_file,
        tf_gene_edge_attr_file=args.tf_gene_edge_attr_file,
        load_tf_gene_attr_to_memory=args.load_hist_to_memory,
        usages_file=args.usages_file,
        use_program_activity_weight=args.use_program_activity_weight,
        activity_weight_mode=args.activity_weight_mode,
        activity_weight_power=args.activity_weight_power,
        min_activity_weight=args.min_activity_weight,
        min_pair_coverage=args.min_pair_coverage,
    )

    print()
    print(f"TF={data.n_tf}, genes={data.n_gene}, programs={data.n_program}")
    print(f"feature_dim={data.feature_dim}, edge_dim={data.edge_dim}")
    print(f"label_mean={data.label_mean:.12g}")
    print(f"label_std={data.label_std:.12g}")

    # Save inference args snapshot.
    args_json = dict(vars(args))
    args_json["_checkpoint"] = str(checkpoint_path)
    args_json["_standalone_inference"] = True
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_json, f, indent=2, ensure_ascii=False, default=str)

    # -------------------------------------------------------------------------
    # 4. Reproduce deterministic preprocessing-side outputs
    # -------------------------------------------------------------------------
    pd.DataFrame({
        "program": data.programs,
        "program_activity_weight": data.program_activity_weight,
    }).to_csv(
        out_dir / "program_activity_weight.tsv",
        sep="\t",
        index=False,
    )

    baseline_df = compute_train_only_baselines(
        data.label_raw,
        data.train_pairs,
        data.val_pairs,
        data.test_pairs,
    )
    baseline_df.insert(0, "split_mode", args.split_mode)
    baseline_df.to_csv(
        out_dir / "train_only_baseline_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 5. Build model architecture locally and load best weights
    # -------------------------------------------------------------------------

    model = CausalEvidenceSubgraphRegressor(
        input_dim=data.feature_dim,
        hidden_dim=args.hidden_dim,
        edge_dim=data.edge_dim,
        num_layers=args.num_layers,
        predictor_layers=args.predictor_layers,
        dropout=args.dropout,
        mixup_lambda=args.mixup_lambda,
        use_reparameterization=args.use_reparameterization,
    ).to(device)

    # strict=True: architecture must match exactly.
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 6. Determine current_epoch only for KL-warmup-scaled final loss metrics
    # -------------------------------------------------------------------------
    final_epoch_for_metrics = None

    if FINAL_EPOCH is not None:
        final_epoch_for_metrics = int(FINAL_EPOCH)

    elif reference_out_dir is not None:
        history_path = reference_out_dir / "history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            if "epoch" in history_df.columns and len(history_df) > 0:
                final_epoch_for_metrics = int(history_df["epoch"].iloc[-1])
                print(
                    "从原训练 history.csv 读取 final epoch =",
                    final_epoch_for_metrics,
                )

    if final_epoch_for_metrics is None:
        final_epoch_for_metrics = int(
            getattr(args, "current_epoch", ckpt.get("epoch", 0))
        )

    args.current_epoch = int(final_epoch_for_metrics)

    # -------------------------------------------------------------------------
    # 7. Same final train / val / test evaluation as training script
    # -------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("train / val / test")
    print("=" * 90)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_eval_ds = TFProgramEvidenceDataset(
        data,
        data.train_pairs,
        augment=False,
        edge_drop_rate=0.0,
    )
    val_ds = TFProgramEvidenceDataset(
        data,
        data.val_pairs,
        augment=False,
        edge_drop_rate=0.0,
    )
    test_ds = TFProgramEvidenceDataset(
        data,
        data.test_pairs,
        augment=False,
        edge_drop_rate=0.0,
    )

    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )

    with torch.inference_mode():
        train_metrics, _, _, _ = evaluate(
            model, train_eval_loader, data, device, args
        )
        val_metrics, _, _, _ = evaluate(
            model, val_loader, data, device, args
        )
        test_metrics, _, _, _ = evaluate(
            model, test_loader, data, device, args
        )

    print("Train metrics:", train_metrics)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    # -------------------------------------------------------------------------
    # 8. EXACT same full TF x Program inference/output functions
    # -------------------------------------------------------------------------

    pred_raw, mask_mean = predict_all_matrix(
        model,
        data,
        device,
        args,
    )

    save_rankings(
        out_dir,
        data,
        pred_raw,
        data.label_raw,
        mask_mean,
        top_n=args.export_top_tfs_per_program,
    )

    save_pair_coverage_outputs(
        out_dir,
        data,
        pred_raw,
    )


    evidence_df = collect_top_evidence_paths(
        model,
        data,
        device,
        args,
        pred_raw,
    )
    evidence_df.to_csv(
        out_dir / "top_evidence_paths_by_mask.csv",
        index=False,
    )

    # Same structure as original training metrics.json.
    metrics_all = {
        "best_epoch": int(ckpt.get("epoch", -1)),
        "pair_filter": data.coverage_filter_stats,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            metrics_all,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if COPY_CHECKPOINT:
        import shutil
        target = out_dir / "best_model.pt"
        if checkpoint_path != target:
            shutil.copy2(checkpoint_path, target)

    # -------------------------------------------------------------------------
    # 9. Optional exact comparison to original training outputs
    # -------------------------------------------------------------------------
    meta = {
        "standalone_inference": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_epoch": ckpt.get("epoch"),
        "checkpoint_best_val": ckpt.get("best_val"),
        "prepared_dir": str(prepared_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "feature_dim": int(data.feature_dim),
        "edge_dim": int(data.edge_dim),
        "label_mean": float(data.label_mean),
        "label_std": float(data.label_std),
        "label_transform": str(data.label_transform),
        "label_standardize": bool(data.label_standardize),
        "current_epoch_used_for_metrics": int(args.current_epoch),
    }

    # if reference_out_dir is not None:
    #
    #     comparisons = {
    #         "best_all_TF_program_score_ces.csv": _compare_matrix_csv(
    #             out_dir / "best_all_TF_program_score_ces.csv",
    #             reference_out_dir / "best_all_TF_program_score_ces.csv",
    #         ),
    #         "best_all_TF_program_mean_edge_mask.csv": _compare_matrix_csv(
    #             out_dir / "best_all_TF_program_mean_edge_mask.csv",
    #             reference_out_dir / "best_all_TF_program_mean_edge_mask.csv",
    #         ),
    #     }
    #
    #     meta["reference_out_dir"] = str(reference_out_dir)
    #     meta["comparisons"] = comparisons
    #
    #     for filename, result in comparisons.items():
    #         print(filename)
    #         print(json.dumps(result, indent=2, ensure_ascii=False))
    #
    #     if comparisons[
    #         "best_all_TF_program_score_ces.csv"
    #     ].get("allclose") is True:
    #         print("[OK]")
    #
    #     if comparisons[
    #         "best_all_TF_program_mean_edge_mask.csv"
    #     ].get("allclose") is True:
    #         print("[OK]")

    with open(
        out_dir / "inference_reproduction_meta.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print("out_dir：", out_dir)
    print(
        "TF_program_score：",
        out_dir / "best_all_TF_program_score_ces.csv",
    )
    print(
        "mean edge mask：",
        out_dir / "best_all_TF_program_mean_edge_mask.csv",
    )
    print(
        "coverage filtered：",
        out_dir / "best_TF_program_score_coverage_filtered.tsv",
    )
    print(
        "evidence paths：",
        out_dir / "top_evidence_paths_by_mask.csv",
    )
    print(
        "metrics：",
        out_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
