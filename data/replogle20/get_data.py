"""
Prepare the Replogle20 dataset for CellSimBench.

This script mirrors the Norman19 combo-perturbation pipeline:
- Load raw matrix and metadata
- Build perturbation conditions
- QC, normalization, and HVG selection
- Technical duplicate splits
- Synthetic controls
- 2-fold combo splits
- DEGs and baselines
- Save processed h5ad
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm


DATASET_NAME = "replogle20"
CELL_TYPE = "k562"
DATA_CACHE_DIR = Path("data/replogle20")
MTX_PATH = DATA_CACHE_DIR / "GSM4367984_exp6.matrix.mtx"
FEATURES_PATH = DATA_CACHE_DIR / "GSM4367984_exp6.features.tsv"
BARCODES_PATH = DATA_CACHE_DIR / "GSM4367984_exp6.barcodes.tsv"
CELL_IDENTITIES_PATH = DATA_CACHE_DIR / "GSM4367984_exp6.cell_identities.csv"

MAX_CELLS_CONTROL = 8192
N_SYNTHETIC_CONTROLS = 500
MIN_CELLS_DEGS = 4
RANDOM_SEED = 42


def read_barcodes(path: Path) -> List[str]:
    """Read a barcodes file with one barcode per line.

    Args:
        path: Path to a barcodes TSV file.

    Returns:
        List of barcode strings in file order.
    """
    with path.open("r", encoding="utf-8") as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    return barcodes


def read_features(path: Path) -> pd.DataFrame:
    """Read a 10x-style features file.

    Args:
        path: Path to a features TSV file (gene_id, gene_name, feature_type).

    Returns:
        DataFrame with columns: gene_id, gene_name, feature_type.
    """
    features = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )
    return features


def read_cell_identities(path: Path) -> pd.DataFrame:
    """Read the cell identities metadata CSV.

    Args:
        path: Path to the cell identities CSV.

    Returns:
        DataFrame with per-cell metadata.
    """
    cell_identities = pd.read_csv(path)
    return cell_identities


def _is_true(value: object) -> bool:
    """Convert a metadata value to boolean.

    Args:
        value: Value to interpret (0/1, bool, or float-like).

    Returns:
        Boolean interpretation of the value.
    """
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def build_condition(row: pd.Series) -> str:
    """Build a perturbation condition string from a metadata row.

    Mapping:
    - control == 1 -> "control"
    - double == 1 -> "{gene_A}+{gene_B}"
    - single == 1 -> gene without "NegCtrl"

    Args:
        row: Metadata row with control/double/single and gene_A/gene_B.

    Returns:
        Condition string.

    Raises:
        ValueError: If the row cannot be mapped to a condition.
    """
    if _is_true(row.get("control")):
        return "control"
    if _is_true(row.get("double")):
        return f"{row['gene_A']}+{row['gene_B']}"
    if _is_true(row.get("single")):
        gene_a = str(row["gene_A"])
        gene_b = str(row["gene_B"])
        a_is_neg = "NegCtrl" in gene_a
        b_is_neg = "NegCtrl" in gene_b
        if a_is_neg and not b_is_neg:
            return gene_b
        if b_is_neg and not a_is_neg:
            return gene_a
        raise ValueError(
            f"Ambiguous single-pert row (gene_A={gene_a}, gene_B={gene_b})."
        )
    raise ValueError("Row is not labeled as control, single, or double.")


def align_matrix_to_metadata(
    matrix: csc_matrix,
    barcodes: Sequence[str],
    cell_metadata: pd.DataFrame,
) -> csr_matrix:
    """Align matrix columns to cell metadata order.

    Args:
        matrix: Gene-by-cell sparse matrix in barcode order.
        barcodes: Barcode list defining matrix column order.
        cell_metadata: Metadata with a cell_barcode column.

    Returns:
        Cell-by-gene CSR matrix aligned to cell_metadata order.

    Raises:
        ValueError: If barcodes are missing or duplicated.
    """
    if "cell_barcode" not in cell_metadata.columns:
        raise ValueError("cell_identities.csv must contain a 'cell_barcode' column.")

    barcode_to_index: Dict[str, int] = {bc: i for i, bc in enumerate(barcodes)}
    missing = set(cell_metadata["cell_barcode"]) - set(barcode_to_index)
    if missing:
        missing_preview = sorted(list(missing))[:10]
        raise ValueError(f"Missing barcodes in matrix: {missing_preview}")

    column_indices = [barcode_to_index[bc] for bc in cell_metadata["cell_barcode"]]
    if len(set(column_indices)) != len(column_indices):
        raise ValueError("Duplicate barcodes detected in cell_identities.csv.")

    aligned = matrix[:, column_indices].T.tocsr()
    return aligned


def load_raw_adata() -> sc.AnnData:
    """Load and assemble raw AnnData from mtx and metadata files.

    Returns:
        AnnData with aligned X, obs, and var.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If matrix dimensions are inconsistent.
    """
    if not MTX_PATH.exists():
        raise FileNotFoundError(f"Missing matrix file: {MTX_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    if not BARCODES_PATH.exists():
        raise FileNotFoundError(f"Missing barcodes file: {BARCODES_PATH}")
    if not CELL_IDENTITIES_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {CELL_IDENTITIES_PATH}")

    barcodes = read_barcodes(BARCODES_PATH)
    features = read_features(FEATURES_PATH)
    cell_metadata = read_cell_identities(CELL_IDENTITIES_PATH)

    raw_matrix = mmread(MTX_PATH).tocsc()
    n_features = features.shape[0]
    n_barcodes = len(barcodes)

    if raw_matrix.shape == (n_features, n_barcodes):
        gene_by_cell = raw_matrix
    elif raw_matrix.shape == (n_barcodes, n_features):
        gene_by_cell = raw_matrix.T.tocsc()
    else:
        raise ValueError(
            "Matrix dimensions do not match features/barcodes: "
            f"{raw_matrix.shape} vs ({n_features}, {n_barcodes})."
        )

    aligned_matrix = align_matrix_to_metadata(gene_by_cell, barcodes, cell_metadata)

    var = features.copy()
    var["ensemble_id"] = var["gene_id"]
    var["gene_name"] = var["gene_name"].astype(str)
    var.index = var["gene_name"]
    var.index.name = None

    adata = sc.AnnData(X=aligned_matrix, obs=cell_metadata.copy(), var=var)
    return adata


def add_conditions(adata: sc.AnnData) -> sc.AnnData:
    """Add perturbation condition columns to AnnData.

    Args:
        adata: AnnData with cell metadata.

    Returns:
        Updated AnnData with condition, perturbation, donor_id, and cell_type.
    """
    adata.obs["condition"] = adata.obs.apply(build_condition, axis=1).astype(str)
    adata.obs["perturbation"] = adata.obs["condition"].astype("category")
    adata.obs["donor_id"] = DATASET_NAME
    adata.obs["cell_type"] = CELL_TYPE
    return adata


def remove_duplicate_genes(adata: sc.AnnData) -> sc.AnnData:
    """Remove duplicated gene names from AnnData.

    Args:
        adata: AnnData with gene names as var_names.

    Returns:
        Filtered AnnData with unique gene names.
    """
    non_dup_mask = ~adata.var_names.duplicated(keep=False)
    return adata[:, non_dup_mask].copy()


def downsample_perturbations(adata: sc.AnnData) -> sc.AnnData:
    """Downsample perturbations to a balanced cell count.

    Args:
        adata: AnnData with condition column.

    Returns:
        Downsampled AnnData.
    """
    pert_counts = adata.obs["condition"].value_counts()
    perts_to_keep = pert_counts[pert_counts >= 12].index
    adata = adata[adata.obs["condition"].isin(perts_to_keep)]
    pert_counts = adata.obs["condition"].value_counts()
    mean_cells = pert_counts.mean()
    max_cells = round(mean_cells)

    cells_to_keep: List[str] = []
    for pert in tqdm(pert_counts.index, desc="Downsampling perturbations"):
        pert_cells = adata.obs[adata.obs["condition"] == pert].index.tolist()
        if pert == "control":
            if len(pert_cells) > MAX_CELLS_CONTROL:
                pert_cells = np.random.choice(
                    pert_cells, size=MAX_CELLS_CONTROL, replace=False
                )
        else:
            if len(pert_cells) > max_cells:
                pert_cells = np.random.choice(pert_cells, size=max_cells, replace=False)
        cells_to_keep.extend(pert_cells)

    return adata[cells_to_keep].copy()


def filter_min_cells_for_baselines(adata: sc.AnnData) -> sc.AnnData:
    """Filter perturbations with too few cells for baselines.

    Args:
        adata: AnnData with condition column.

    Returns:
        Filtered AnnData.
    """
    pert_counts = adata.obs["condition"].value_counts()
    valid_perts = pert_counts[pert_counts >= 4].index
    control_conditions = ["control", "ctrl"]
    for ctrl in control_conditions:
        if ctrl in adata.obs["condition"].unique() and ctrl not in valid_perts:
            valid_perts = valid_perts.append(pd.Index([ctrl]))
    return adata[adata.obs["condition"].isin(valid_perts)].copy()


def select_hvg_and_perturbation_genes(adata: sc.AnnData) -> sc.AnnData:
    """Keep highly variable genes plus perturbation genes.

    Args:
        adata: AnnData with condition column.

    Returns:
        Filtered AnnData.
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=8192, subset=False)
    perts = adata.obs["condition"].unique()
    genes = [gene for pert in perts for gene in pert.split("+") if gene != "control"]
    genes = list(set(genes))
    hvg_genes = adata.var_names[adata.var.highly_variable]
    genes_to_keep = list(set(hvg_genes) | set(genes))
    genes_to_keep = [gene for gene in genes_to_keep if gene in adata.var_names]
    return adata[:, genes_to_keep].copy()


def assign_technical_duplicate_splits(adata: sc.AnnData) -> sc.AnnData:
    """Assign technical duplicate splits per condition.

    Args:
        adata: AnnData with condition column.

    Returns:
        AnnData with tech_dup_split column.
    """
    adata.obs["tech_dup_split"] = pd.NA
    unique_conditions = adata.obs["condition"].unique()

    for condition in tqdm(unique_conditions, desc="Assigning technical duplicate splits"):
        condition_cells = adata.obs[adata.obs["condition"] == condition].index
        if len(condition_cells) >= 2:
            cell_indices = np.random.permutation(condition_cells)
            split_idx = len(cell_indices) // 2
            adata.obs.loc[cell_indices[:split_idx], "tech_dup_split"] = "first_half"
            adata.obs.loc[cell_indices[split_idx:], "tech_dup_split"] = "second_half"
    return adata


def add_synthetic_controls(adata: sc.AnnData, n_controls: int) -> sc.AnnData:
    """Add synthetic mean control cells.

    Args:
        adata: AnnData with condition column.
        n_controls: Number of synthetic controls to generate.

    Returns:
        AnnData with synthetic controls appended.
    """
    non_ctrl_cells = adata[~adata.obs["condition"].str.contains("control", case=False)]
    n_mean_controls = min(n_controls // 4, 100)
    mean_controls: List[sc.AnnData] = []

    for i in range(n_mean_controls):
        sampled_for_mean = np.random.choice(non_ctrl_cells.obs_names, size=100, replace=False)
        mean_expr = adata[sampled_for_mean].X.mean(axis=0)
        obs_df = pd.DataFrame(
            {
                "condition": "ctrl_synthetic_mean",
                "cell_type": adata.obs["cell_type"].iloc[0],
                "donor_id": DATASET_NAME,
                "ncounts": float(np.sum(mean_expr)),
                "ngenes": int(np.sum(mean_expr > 0)),
            },
            index=[f"synthetic_mean_{i}"],
        )
        mean_adata = sc.AnnData(X=mean_expr.reshape(1, -1), obs=obs_df, var=adata.var)
        mean_controls.append(mean_adata)

    if mean_controls:
        mean_adata_combined = sc.concat(mean_controls, join="outer", index_unique="_")
        adata = sc.concat([adata, mean_adata_combined], join="outer", index_unique="_")
        adata.var = adata.var.copy()
    return adata


def create_combo_splits(adata: sc.AnnData) -> sc.AnnData:
    """Create 2-fold combo perturbation splits.

    Args:
        adata: AnnData with condition column.

    Returns:
        AnnData with split_fold_0 and split_fold_1 columns.
    """
    adata.obs["split_fold_0"] = ""
    adata.obs["split_fold_1"] = ""

    all_conditions = adata.obs["condition"].unique()
    single_perts = [cond for cond in all_conditions if "+" not in cond and "control" not in cond]
    combo_perts = [cond for cond in all_conditions if "+" in cond and "control" not in cond]
    control_conditions = [cond for cond in all_conditions if "control" in cond]

    for pert in single_perts:
        adata.obs.loc[adata.obs["condition"] == pert, "split_fold_0"] = "train"
        adata.obs.loc[adata.obs["condition"] == pert, "split_fold_1"] = "train"

    if combo_perts:
        np.random.shuffle(combo_perts)
        n_test = len(combo_perts) // 2
        n_val = (len(combo_perts) - n_test) // 2

        fold_0_test = combo_perts[:n_test]
        fold_1_test = combo_perts[n_test:]
        fold_0_val = fold_1_test[:n_val]
        fold_1_val = fold_0_test[:n_val]
        fold_0_train = fold_1_test[n_val:]
        fold_1_train = fold_0_test[n_val:]

        for pert in combo_perts:
            if pert in fold_0_test:
                adata.obs.loc[adata.obs["condition"] == pert, "split_fold_0"] = "test"
            if pert in fold_0_val:
                adata.obs.loc[adata.obs["condition"] == pert, "split_fold_0"] = "val"
            if pert in fold_0_train:
                adata.obs.loc[adata.obs["condition"] == pert, "split_fold_0"] = "train"
            if pert in fold_1_test:
                adata.obs.loc[adata.obs["condition"] == pert, "split_fold_1"] = "test"
            if pert in fold_1_val:
                adata.obs.loc[adata.obs["condition"] == pert, "split_fold_1"] = "val"
            if pert in fold_1_train:
                adata.obs.loc[adata.obs["condition"] == pert, "split_fold_1"] = "train"

    for ctrl_cond in control_conditions:
        ctrl_cells = adata.obs[adata.obs["condition"] == ctrl_cond].index
        n_cells = len(ctrl_cells)
        ctrl_cells_shuffled = np.random.permutation(ctrl_cells)
        n_train = int(0.5 * n_cells)
        n_val = int(0.25 * n_cells)

        train_cells_fold_0 = ctrl_cells_shuffled[:n_train]
        val_cells_fold_0 = ctrl_cells_shuffled[n_train : n_train + n_val]
        test_cells_fold_0 = ctrl_cells_shuffled[n_train + n_val :]

        train_cells_fold_1 = ctrl_cells_shuffled[n_train:]
        val_cells_fold_1 = ctrl_cells_shuffled[:n_val]
        test_cells_fold_1 = ctrl_cells_shuffled[n_val:n_train]

        adata.obs.loc[train_cells_fold_0, "split_fold_0"] = "train"
        adata.obs.loc[val_cells_fold_0, "split_fold_0"] = "val"
        adata.obs.loc[test_cells_fold_0, "split_fold_0"] = "test"

        adata.obs.loc[train_cells_fold_1, "split_fold_1"] = "train"
        adata.obs.loc[val_cells_fold_1, "split_fold_1"] = "val"
        adata.obs.loc[test_cells_fold_1, "split_fold_1"] = "test"

    return adata


def calculate_degs(adata: sc.AnnData) -> sc.AnnData:
    """Calculate DEGs using second half of technical duplicates.

    Args:
        adata: AnnData with tech_dup_split and condition columns.

    Returns:
        AnnData with DEG results stored in .uns.
    """
    adata_second_half = adata[adata.obs["tech_dup_split"] == "second_half"].copy()
    pert_counts = adata_second_half.obs["condition"].value_counts()
    valid_perts = pert_counts[pert_counts >= MIN_CELLS_DEGS].index
    adata_deg = adata_second_half[adata_second_half.obs["condition"].isin(valid_perts)].copy()

    sc.tl.rank_genes_groups(
        adata_deg, "condition", method="t-test_overestim_var", reference="rest"
    )

    names_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals_adj"])
    pvals_unadj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals"])
    scores_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["scores"])

    deg_dict: Dict[str, List[str]] = {}
    names_df_dict_final: Dict[str, List[str]] = {}
    pvals_adj_df_dict_final: Dict[str, List[float]] = {}
    pvals_unadj_df_dict_final: Dict[str, List[float]] = {}
    scores_df_dict_final: Dict[str, List[float]] = {}

    for pert in names_df.columns:
        if pert == "control" or "ctrl" in pert:
            continue
        names_df_dict_final[f"{DATASET_NAME}_{pert}"] = names_df[pert].tolist()
        pvals_adj_df_dict_final[f"{DATASET_NAME}_{pert}"] = pvals_adj_df[pert].tolist()
        pvals_unadj_df_dict_final[f"{DATASET_NAME}_{pert}"] = pvals_unadj_df[pert].tolist()
        scores_df_dict_final[f"{DATASET_NAME}_{pert}"] = scores_df[pert].tolist()

        pert_degs = names_df[pert]
        pert_degs_sig = pert_degs[pvals_adj_df[pert] < 0.05]
        deg_dict[f"{DATASET_NAME}_{pert}"] = pert_degs_sig.tolist()

    adata.uns["deg_gene_dict"] = deg_dict
    adata.uns["names_df_dict"] = names_df_dict_final
    adata.uns["pvals_adj_df_dict"] = pvals_adj_df_dict_final
    adata.uns["pvals_unadj_df_dict"] = pvals_unadj_df_dict_final
    adata.uns["scores_df_dict"] = scores_df_dict_final

    return adata


def add_baselines(adata: sc.AnnData, split_col: str) -> sc.AnnData:
    """Add per-fold mean baselines.

    Args:
        adata: AnnData with split column.
        split_col: Split column name.

    Returns:
        AnnData with fold-specific baselines stored in .uns.
    """
    train_cells = adata[adata.obs[split_col] == "train"]
    train_non_ctrl = train_cells[~train_cells.obs["condition"].str.contains("control")]
    unique_donor_condition = train_non_ctrl.obs[["donor_id", "condition"]].drop_duplicates()

    condition_means = pd.DataFrame(index=range(len(unique_donor_condition)), columns=adata.var_names)
    for idx, (_, row) in enumerate(
        tqdm(unique_donor_condition.iterrows(), desc="Computing condition means")
    ):
        donor_id = row["donor_id"]
        condition = row["condition"]
        combo_cells = train_non_ctrl[
            (train_non_ctrl.obs["donor_id"] == donor_id)
            & (train_non_ctrl.obs["condition"] == condition)
        ]
        if len(combo_cells) > 0:
            combo_mean = combo_cells.X.mean(axis=0)
            if hasattr(combo_mean, "A1"):
                combo_mean = combo_mean.A1
            condition_means.iloc[idx] = combo_mean

    mean_baseline_df = pd.DataFrame(index=[DATASET_NAME], columns=adata.var_names)
    mean_baseline_df.loc[DATASET_NAME] = condition_means.mean(axis=0)
    mean_baseline_df = mean_baseline_df.astype(float)
    adata.uns[f"{split_col}_split_mean_baseline"] = mean_baseline_df
    return adata


def add_universal_baselines(adata: sc.AnnData) -> sc.AnnData:
    """Add universal baselines (control, technical duplicates, additive).

    Args:
        adata: AnnData with condition and split columns.

    Returns:
        AnnData with baselines stored in .uns.
    """
    ctrl_cells = adata[adata.obs["condition"] == "control"]
    if len(ctrl_cells) > 0:
        ctrl_mean = ctrl_cells.X.mean(axis=0)
        if hasattr(ctrl_mean, "A1"):
            ctrl_mean = ctrl_mean.A1
        ctrl_baseline_df = pd.DataFrame(index=[DATASET_NAME], columns=adata.var_names)
        ctrl_baseline_df.loc[DATASET_NAME] = ctrl_mean
        adata.uns["ctrl_baseline"] = ctrl_baseline_df.astype(float)

    unique_conditions = adata.obs["condition"].unique()
    tech_dup_first_half = pd.DataFrame(index=unique_conditions, columns=adata.var_names)
    tech_dup_second_half = pd.DataFrame(index=unique_conditions, columns=adata.var_names)

    for condition in tqdm(unique_conditions, desc="Computing technical duplicates"):
        condition_cells = adata[adata.obs["condition"] == condition]
        first_half_cells = condition_cells[condition_cells.obs["tech_dup_split"] == "first_half"]
        second_half_cells = condition_cells[condition_cells.obs["tech_dup_split"] == "second_half"]

        if len(first_half_cells) > 0 and len(second_half_cells) > 0:
            first_half_mean = first_half_cells.X.mean(axis=0)
            second_half_mean = second_half_cells.X.mean(axis=0)
            if hasattr(first_half_mean, "A1"):
                first_half_mean = first_half_mean.A1
            if hasattr(second_half_mean, "A1"):
                second_half_mean = second_half_mean.A1
            tech_dup_first_half.loc[condition] = first_half_mean
            tech_dup_second_half.loc[condition] = second_half_mean

    tech_dup_first_half = tech_dup_first_half.dropna().astype(float)
    tech_dup_second_half = tech_dup_second_half.dropna().astype(float)
    tech_dup_first_half.index = [f"{DATASET_NAME}_{cond}" for cond in tech_dup_first_half.index]
    tech_dup_second_half.index = [f"{DATASET_NAME}_{cond}" for cond in tech_dup_second_half.index]

    adata.uns["technical_duplicate_first_half_baseline"] = tech_dup_first_half
    adata.uns["technical_duplicate_second_half_baseline"] = tech_dup_second_half

    all_conditions = adata.obs["condition"].unique()
    control_conditions = [cond for cond in all_conditions if "control" in cond or "ctrl" in cond]
    single_conditions = [cond for cond in all_conditions if cond not in control_conditions and "+" not in cond]
    combo_conditions = [cond for cond in all_conditions if cond not in control_conditions and "+" in cond]

    train_adata = adata[adata.obs["split_fold_0"] == "train"]
    train_single_conditions = [
        cond for cond in train_adata.obs["condition"].unique() if cond in single_conditions
    ]

    ctrl_cells = train_adata[train_adata.obs["condition"] == "control"]
    if len(ctrl_cells) > 0:
        ctrl_mean = ctrl_cells.X.mean(axis=0)
        if hasattr(ctrl_mean, "A1"):
            ctrl_mean = ctrl_mean.A1
    else:
        ctrl_mean = np.zeros(adata.shape[1])

    single_effects: Dict[str, np.ndarray] = {}
    for condition in tqdm(train_single_conditions, desc="Computing single effects"):
        condition_cells = train_adata[train_adata.obs["condition"] == condition]
        if len(condition_cells) > 0:
            condition_mean = condition_cells.X.mean(axis=0)
            if hasattr(condition_mean, "A1"):
                condition_mean = condition_mean.A1
            single_effects[condition] = condition_mean - ctrl_mean

    additive_predictions: List[Dict[str, np.ndarray]] = []
    for combo_condition in combo_conditions:
        genes_in_combo = combo_condition.split("+")
        predicted_expression = ctrl_mean.copy()
        for gene in genes_in_combo:
            if gene in single_effects:
                predicted_expression += single_effects[gene]
        additive_predictions.append(
            {
                "condition": f"{DATASET_NAME}_{combo_condition}",
                "expression": predicted_expression,
            }
        )

    if additive_predictions:
        additive_df = pd.DataFrame(
            index=[pred["condition"] for pred in additive_predictions],
            columns=adata.var_names,
        )
        for i, pred in enumerate(additive_predictions):
            additive_df.iloc[i] = pred["expression"]
        adata.uns["additive_baseline"] = additive_df.astype(float)

    return adata


def add_gene_metadata(adata: sc.AnnData) -> sc.AnnData:
    """Add symbol_scgpt mapping based on ensemble IDs.

    Args:
        adata: AnnData with ensemble_id column.

    Returns:
        AnnData with symbol_scgpt column in var.
    """
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.copy()
    adata.var.index.name = None

    converter_path = Path("data/ref/gene_info_scgpt.csv")
    converter = pd.read_csv(converter_path)
    converter = converter[["feature_name", "feature_id"]]
    converter.set_index("feature_id", inplace=True)

    if "ensemble_id" not in adata.var.columns:
        adata.var["ensemble_id"] = adata.var.get("gene_id", adata.var_names)
    mapped_symbols = adata.var["ensemble_id"].map(
        lambda x: converter.loc[x]["feature_name"] if x in converter.index else pd.NA
    )
    fallback_symbols = adata.var["gene_name"].astype(str)
    symbol_scgpt = mapped_symbols.where(
        ~pd.isna(mapped_symbols) & (mapped_symbols.astype(str).str.len() > 0),
        fallback_symbols,
    )
    adata.var["symbol_scgpt"] = symbol_scgpt.astype(str)
    return adata


def main() -> None:
    """Run the full Replogle20 processing pipeline."""
    np.random.seed(RANDOM_SEED)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print("Loading raw data...")
    adata = load_raw_adata()
    adata = add_conditions(adata)
    adata = remove_duplicate_genes(adata)
    adata.X = csr_matrix(adata.X)

    print("Filtering cells and genes...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Downsampling perturbations...")
    adata = downsample_perturbations(adata)
    print("Filtering perturbations with at least 4 cells...")
    adata = filter_min_cells_for_baselines(adata)

    print("Selecting HVGs + perturbation genes...")
    adata = select_hvg_and_perturbation_genes(adata)

    print("Assigning technical duplicate splits...")
    adata = assign_technical_duplicate_splits(adata)

    print("Adding synthetic controls...")
    adata = add_synthetic_controls(adata, n_controls=N_SYNTHETIC_CONTROLS)

    print("Creating 2-fold combo splits...")
    adata = create_combo_splits(adata)

    print("Calculating DEGs...")
    adata = calculate_degs(adata)

    print("Adding baselines...")
    adata = add_universal_baselines(adata)
    adata = add_baselines(adata, split_col="split_fold_0")
    adata = add_baselines(adata, split_col="split_fold_1")

    print("Computing PCA...")
    sc.pp.pca(adata)

    print("Adding gene metadata...")
    adata = add_gene_metadata(adata)

    output_path = DATA_CACHE_DIR / f"{DATASET_NAME}_processed_complete.h5ad"
    print(f"Saving processed data to {output_path}")
    adata.write_h5ad(str(output_path))
    print("Processing complete.")


if __name__ == "__main__":
    main()
