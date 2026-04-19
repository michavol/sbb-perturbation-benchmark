# Perturbation Discrimination: Comprehensive Guide

This guide documents the perturbation discrimination analyses from the **Signal pillar** of the SBB framework:

> **Signal, Bounds, and Baselines: Principles for Rigorous Evaluation of High-Dimensional Biological Perturbation Prediction**
> Vollenweider & Bühlmann, 2026.

These analyses verify that evaluation metrics are sensitive to biological signal (BDS, PDS meta-metrics), quantify signal dilution in high-dimensional spaces (effective genes curves, bag-size curves), and compare metric sensitivity across datasets.

**Mapping to paper figures:**
- BDS and PDS heatmaps (Fig. 4a, 4b): `scripts/plot_technical_baseline_diagnostics.py` (from benchmarking results, not from this directory)
- BDS and PDS permutation tests (Supplementary): `run_pds_test.py`, `run_bds_test.py`
- Cross-dataset metric comparison (Fig. 4): `run_metric_comparison.py`
- Effective genes / DEG filtering curve (Fig. 5a, 5b): `run_signal_dilution_curves.py --mode effective_genes_curve`
- Bag-size curve (Fig. 5c): `run_signal_dilution_curves.py --mode bag_size_curve`
- Per-gene filtering comparison: `compute_full_pds.py` + `discrimination_comparison.py`

End-to-end workflow:

1. prepare data/config and compute DEG weights
2. compute full PDS
3. run dilution/curve analyses
4. run PDS/BDS permutation tests
5. run cross-dataset metric comparisons
6. generate visualization summaries

All commands assume the repository root as the working directory.

---

## 1) Data and configuration

Each dataset requires:

- `cellsimbench/configs/dataset/<dataset>.yaml`
- a valid processed `.h5ad` referenced in that config

If needed, build data via:

- `data/<dataset>/get_data.py`

Before running analyses, verify:

- dataset config exists
- `.h5ad` loads correctly
- perturbation labels and control label are correct

### DEG weight prerequisite

Many analyses require pre-computed DEG weights. Run the standalone script before proceeding:

```bash
uv run python scripts/compute_deg_weights.py \
  --datasets adamson16 frangieh21 nadig25hepg2 norman19 replogle20 replogle22k562 wessels23 \
  --mode both
```

This populates `analyses/perturbation_discrimination/results/<dataset>/deg_scanpy/` with the required weight CSVs. Alternatively, weights are computed on-the-fly by `launch_pds_bds_full.sh` (step 1).

---

## 2) Compute full PDS

Script:

- `analyses/perturbation_discrimination/compute_full_pds.py`

Main output:

- `analyses/perturbation_discrimination/results/<dataset>/pds_full.h5ad`

Example:

```bash
uv run python analyses/perturbation_discrimination/compute_full_pds.py \
  --datasets wessels23 norman19 frangieh21 adamson16 replogle22k562 \
  --metrics Energy_Distance Wasserstein_1D \
  --n-trials 5 \
  --seed 42 \
  --n-jobs 16
```

Notes:

- Run once per dataset/metric setup; downstream analyses reuse `pds_full.h5ad`.
- Tune `--n-jobs` based on available RAM/CPU.

---

## 3) Signal dilution analyses

Main script:

- `analyses/perturbation_discrimination/run_signal_dilution_curves.py`

Modes:

- `single_cell_pds_rank_curve`
- `bag_size_curve`
- `effective_genes_curve`
- `bag_rank_curve`

Common options:

- `--datasets`
- `--metrics`
- `--n-trials`, `--seed`
- `--trial-cell-budget`
- `--split-mode budget|all_cells` — cell splitting strategy (see below)
- `--perturbation-filter all|single|double` — filter perturbations by type
- `--max-perturbations` — limit number of perturbations (default: None, use all). When set, keeps perturbations with most cells. Note: 50 perturbations is a common choice for initial analyses; for double combo filtering, use all perturbations to avoid filtering out combos inadvertently.
- `--weight-source deg|pds`
- `--weight-scheme normal|rank`
- `--target-effective-genes`, `--target-effective-genes-mode`, `--pds-target-effective-genes-mode`
- `--no-log-x` — use a **linear** x-axis instead of logarithmic (applies to `effective_genes_curve`); also switches default x-value sampling to `auto_linear` unless `--effective-genes-values` is set explicitly
- `--output-dir` (optional override)

**Cell splitting modes (`--split-mode`):**

- `budget` (default): Use up to `--trial-cell-budget` cells per perturbation. Creates multiple single-cell pairs per trial by randomly sampling cells.
- `all_cells`: Use ALL available cells per perturbation, split randomly into two halves. Imbalance is allowed if a perturbation has an odd cell count. Returns exactly one comparison pair per trial with maximum statistical power.

**Perturbation filters (`--perturbation-filter`):**

- `all` (default): Use all perturbations
- `single`: Only single-gene perturbations (no `__`, `|`, `;`, `+`, or `_` separator)
- `double`: Only double combo perturbations (contains separator like `_`, `__`, `|`, `;`, `+`)

---

## 4) Analyses by mode

### 4.1 Single-cell PDS rank curve

Purpose:

- x-axis: top-k genes
- one-cell-per-side discrimination scoring

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode single_cell_pds_rank_curve \
  --datasets wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta,pearson_delta,w_pearson_delta \
  --topk-values 10,20,50,100,200,400,800,1600,3200,6400 \
  --n-trials 3 \
  --trial-cell-budget 100 \
  --seed 42 \
  --weight-source deg \
  --weight-scheme normal \
  --n-jobs 8
```

Default output:

- `.../results/<dataset>/signal_dilution_curves/single_cell_pds_rank_curve/<dataset>/single_cell_pds_rank_curve/`

---

### 4.2 Bag-size curve (bag analysis)

Purpose:

- x-axis: bag size
- bag-vs-bag discrimination

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode bag_size_curve \
  --datasets wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta,pearson_delta,w_pearson_delta \
  --bag-sizes 2,5,10,15,20,30,40,50,70,90 \
  --n-trials 3 \
  --trial-cell-budget 100 \
  --seed 42 \
  --weight-source deg \
  --weight-scheme normal \
  --n-jobs 8
```

Default output:

- `.../results/<dataset>/signal_dilution_curves/bag_size_curve/<dataset>/bag_size_curve/`

---

### 4.3 Bag-rank curve (ranking with bag numbers)

Purpose:

- x-axis: top-k genes (as in single-cell ranking)
- scoring uses bag means
- one set of curves per bag size

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode bag_rank_curve \
  --datasets wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta,pearson_delta,w_pearson_delta \
  --bag-sizes 1,5,10,20,50 \
  --n-trials 3 \
  --trial-cell-budget 100 \
  --seed 42 \
  --weight-source deg \
  --weight-scheme normal \
  --n-jobs 8
```

Default output:

- `.../results/<dataset>/signal_dilution_curves/pds_rank_curve_different_bag_sizes/<dataset>/bag_rank_curve/`

Per bag size:

- `bag_rank_curve_bagsize<k>.csv`
- `bag_rank_curve_bagsize<k>.png/.pdf` (default y-axis)
- `bag_rank_curve_bagsize<k>_ylim01.png/.pdf` (fixed y-axis `[0,1]`)

---

### 4.4 Effective-genes curve

Purpose:

- x-axis: target effective gene number
- weighted metrics under calibrated weight sharpness

The `--effective-genes-values` argument controls which x-axis values are swept:

- `auto_log10` (default) — log-spaced values (10, 15, 20, 30, 50, 70, 100, …, n_genes)
- `auto_linear` — linearly-spaced values (~20 steps from 1 to n_genes)
- comma-separated list — explicit values, e.g. `10,50,100,500`

Pass `--no-log-x` to draw the x-axis on a linear scale and automatically use `auto_linear` sampling (unless you provide explicit values).

Standard example:

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode effective_genes_curve \
  --datasets wessels23 \
  --metrics wmse,w_r2_delta,w_pearson_delta,weighted_energy_distance \
  --effective-genes-values auto_log10 \
  --target-effective-genes 20 \
  --target-effective-genes-mode mean \
  --weight-source deg \
  --weight-scheme normal \
  --n-trials 3 \
  --trial-cell-budget 100 \
  --seed 42 \
  --n-jobs 8
```

Linear-axis variant (same run, linear x-axis and sampling):

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode effective_genes_curve \
  --datasets wessels23 \
  --metrics wmse,w_r2_delta,w_pearson_delta,weighted_energy_distance \
  --no-log-x \
  --target-effective-genes 20 \
  --target-effective-genes-mode mean \
  --weight-source deg \
  --weight-scheme normal \
  --n-trials 3 \
  --trial-cell-budget 100 \
  --seed 42 \
  --n-jobs 8
```

Example with **all cells** and **double combo perturbations only**:

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode effective_genes_curve \
  --dataset wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta \
  --split-mode all_cells \
  --perturbation-filter double \
  --effective-genes-values auto_log10 \
  --target-effective-genes 20 \
  --target-effective-genes-mode mean \
  --weight-source deg \
  --weight-scheme normal \
  --n-trials 5 \
  --seed 42 \
  --n-jobs 8 \
  --output-dir analyses/perturbation_discrimination/results/wessels23/signal_dilution_curves/effective_genes_curve_double_allcells
```

Pre-configured **frangieh21** runs (see `scripts/train_all_models.sh` lines 47–72):

- **Weighted effective-genes curve (all cells):**

  ```bash
  uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
    --mode effective_genes_curve \
    --dataset frangieh21 \
    --split-mode all_cells \
    --effective-genes-values 10,15,20,30,50,70,100,150,200,300,500,700,1000,1500,2000 \
    --metrics wmse,w_r2_delta,w_pearson_delta \
    --target-effective-genes-mode mean \
    --weight-source deg \
    --weight-scheme normal \
    --n-trials 3 \
    --seed 42 \
    --n-jobs 16 \
    --output-dir analyses/perturbation_discrimination/results/frangieh21/signal_dilution_curves/effective_genes_curve_double_allcells_weighted
  ```

- **TopN DEG-only effective-genes curve (all cells):**

  ```bash
  uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
    --mode effective_genes_curve \
    --dataset frangieh21 \
    --split-mode all_cells \
    --metrics top_n_deg_mse,top_n_deg_r2,top_n_deg_pearson \
    --target-effective-genes-mode mean \
    --weight-source deg \
    --weight-scheme normal \
    --n-trials 3 \
    --seed 42 \
    --n-jobs 16 \
    --output-dir analyses/perturbation_discrimination/results/frangieh21/signal_dilution_curves/effective_genes_curve_double_allcells_topn
  ```

Default output:

- `.../results/<dataset>/signal_dilution_curves/effective_genes_curve/<dataset>/effective_genes_curve/`

---

## 5) Top20 vs rest comparisons

Use Top-N DEG metrics against weighted full-gene metrics:

- `top_n_deg_mse` vs `wmse`
- `top_n_deg_r2` vs `w_r2_delta`
- `top_n_deg_pearson` vs `w_pearson_delta`

Recommended metric set:

- `top_n_deg_mse,wmse,top_n_deg_r2,w_r2_delta,top_n_deg_pearson,w_pearson_delta`

Example:

```bash
METRICS_TOP20="top_n_deg_mse,wmse,top_n_deg_r2,w_r2_delta,top_n_deg_pearson,w_pearson_delta"

uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode single_cell_pds_rank_curve \
  --datasets frangieh21 norman19 adamson16 \
  --metrics $METRICS_TOP20 \
  --target-effective-genes 20 \
  --target-effective-genes-mode mean \
  --weight-source deg \
  --weight-scheme normal \
  --n-trials 3 \
  --trial-cell-budget 100 \
  --seed 42 \
  --n-jobs 8
```

The same metric set can be used with:

- `bag_size_curve`
- `effective_genes_curve`
- `bag_rank_curve`

---

## 6) Cross-dataset metric comparison

Script: `analyses/perturbation_discrimination/run_metric_comparison.py`

This script performs a cross-dataset comparison of distance metrics using Cohen's d' and AUC as standardized effect sizes. It evaluates how well each metric discriminates between different perturbations across multiple datasets.

### 6.1 Standard mode (d', AUC, PDS)

In standard mode, the script computes for each dataset and metric:
- **Cohen's d'** — Cross-metric comparable effect size
- **AUC** — Probability that between-perturbation distance exceeds within-perturbation distance
- **PDS** — Perturbation Discrimination Score (retrieval accuracy)

The analysis uses top-50 perturbations (by cell count) and all available cells per perturbation, split into two halves (technical duplicates) to compute self-distances.

```bash
uv run python analyses/perturbation_discrimination/run_metric_comparison.py \
  --mode standard \
  --datasets adamson16 frangieh21 norman19 replogle20 wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta,pearson_delta,w_pearson_delta,energy_distance,weighted_energy_distance \
  --n-trials 5 \
  --n-resamples 20 \
  --max-perturbations 50 \
  --seed 42 \
  --n-jobs 8
```

**Outputs:**

- `metric_comparison.csv` — Aggregated results per (dataset, metric, trial)
- `metric_comparison_per_pert.csv` — Per-perturbation values (one row per perturbation × trial)
- `bar_dprime.{png,pdf}` — Grouped bar plot of Cohen's d' by metric
- `bar_auc.{png,pdf}` — Grouped bar plot of AUC by metric  
- `bar_pds.{png,pdf}` — Grouped bar plot of PDS by metric
- `heatmap_dprime.{png,pdf}` — Heatmap of datasets × metrics for d'
- `heatmap_auc.{png,pdf}` — Heatmap of datasets × metrics for AUC
- `heatmap_pds.{png,pdf}` — Heatmap of datasets × metrics for PDS
- `dist_dprime.{png,pdf}` — Distribution plot (violin + strip) of per-perturbation d'
- `dist_auc.{png,pdf}` — Distribution plot of per-perturbation AUC
- `dist_pds.{png,pdf}` — Distribution plot of per-perturbation PDS

### 6.2 Self-control mode

Self-control mode evaluates whether metrics correctly rank technical duplicates (same perturbation, different cell halves) as more similar than control samples. This reveals calibration failures where a metric cannot distinguish perturbed cells from control.

For each perturbation:
- **Self-distance**: distance to the other half of the same perturbation
- **Control-distance**: distance to the control mean
- **Accuracy**: ratio of perturbations where self-distance < control-distance

```bash
uv run python analyses/perturbation_discrimination/run_metric_comparison.py \
  --mode self_control \
  --datasets adamson16 frangieh21 norman19 replogle20 wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta,pearson_delta,w_pearson_delta,energy_distance,weighted_energy_distance \
  --n-trials 5 \
  --n-resamples 20 \
  --max-perturbations 50 \
  --seed 42 \
  --n-jobs 8
```

**Outputs:**

- `self_control_results.csv` — Per-trial accuracy results
- `self_control_heatmap.{png,pdf}` — Heatmap of discrimination accuracy (datasets × metrics)

**Interpretation:**

- Green cells (high accuracy): Metric correctly identifies that perturbations are more similar to themselves than to control
- Red cells (low accuracy): Metric fails to separate perturbed cells from control
- In the example output, MSE shows ~0% accuracy on Frangieh '21 (technical duplicates are further apart than control), while wMSE achieves ~92% accuracy

### 6.3 Common options

| Option | Description |
|--------|-------------|
| `--mode {standard,self_control}` | Analysis mode: standard for d'/AUC/PDS, self_control for self-vs-control accuracy |
| `--datasets` | List of datasets to evaluate |
| `--metrics` | Comma-separated list of distance metrics |
| `--n-trials` | Number of independent random trials (for error bars) |
| `--n-resamples` | Number of half-cell resamples per trial (for variance estimation) |
| `--max-perturbations` | Keep top-N perturbations by cell count (default: 50) |
| `--n-jobs` | Parallel processes for dataset-level parallelism |
| `--output-dir` | Output directory for results |
| `--formats` | Output image formats: png, pdf, or both |

---

## 7) Visualize and compare results

Primary scripts:

- `analyses/perturbation_discrimination/visualize_results.py`
- `analyses/perturbation_discrimination/discrimination_comparison.py`
- `analyses/perturbation_discrimination/plot_umap_filters.py`
- `analyses/perturbation_discrimination/visualize_distributions.py`

Example:

```bash
uv run python analyses/perturbation_discrimination/visualize_results.py \
  --results-root analyses/perturbation_discrimination/results \
  --datasets wessels23 norman19 frangieh21 adamson16 \
  --metrics Energy_Distance \
  --style nature
```

---

## 8) Minimal end-to-end checklist

For each dataset:

1. Validate dataset YAML + `.h5ad`.
2. Run `compute_full_pds.py`.
3. Run:
   - `single_cell_pds_rank_curve`
   - `bag_size_curve`
   - `bag_rank_curve`
   - `effective_genes_curve`
4. Run top20-vs-rest variants (optional, recommended).
5. Run cross-dataset metric comparison (Section 6) to compare metrics across datasets.
6. Generate summary/comparison plots.

