# Signal, Bounds, and Baselines (SBB)

Code for reproducing the analyses in:

> **Signal, Bounds, and Baselines: Principles for Rigorous Evaluation of High-Dimensional Biological Perturbation Prediction**
> Michael Vollenweider and Peter Bühlmann
> *bioRxiv*, April 2026

This repository implements the SBB evaluation framework for perturbation prediction, including DEG-weighted metrics, perturbation-wise metric calibration, a hierarchy of linear baselines, and diagnostic meta-metrics (BDS, PDS). It benchmarks several deep learning models across seven Perturb-seq datasets and provides tools for perturbation discrimination analysis.

**Codebase origin.** This repository is adapted from [CellSimBench](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines) by Miller et al. (2025), which provides the Docker-based model training/inference pipeline and dataset preprocessing. We extended it with new baselines, metrics, and the perturbation discrimination analyses described in the paper.

For questions, contact: [michael.vollenweider@stat.math.ethz.ch](mailto:michael.vollenweider@stat.math.ethz.ch)

---

## Table of Contents

1. [Setup](#1-setup)
2. [Pull Datasets and Containers](#2-pull-datasets-and-containers)
3. [Compute DEG Weights](#3-compute-deg-weights)
4. [Train and Benchmark Models](#4-train-and-benchmark-models)
5. [Visualize Results](#5-visualize-results)
6. [Perturbation Discrimination Analyses](#6-perturbation-discrimination-analyses)
7. [Available Models, Datasets, Metrics, and Baselines](#7-available-models-datasets-metrics-and-baselines)
8. [Building from Scratch](#8-building-from-scratch)
9. [Docker with Podman (cluster nodes)](#9-docker-with-podman-cluster-nodes)
10. [Citation](#10-citation)

---

## 1. Setup

**Prerequisites:**

- Python >= 3.12
- Docker installed and running
- AWS CLI installed (for downloading datasets)
- OpenAI API key (required only for scLambda)

**Recommended hardware (full reproduction):**

- 5 GPUs with at least 24 GB VRAM each
- 384 GB CPU RAM, 64 CPU cores
- 2 TB storage

```bash
# Clone and install
git clone <repository-url>
cd <repository-name>
uv sync          # or: pip install -e .
source .venv/bin/activate

# (optional) Jupyter notebooks in VS Code
uv add --dev ipykernel

# Create .env file (required only for scLambda)
echo "OPENAI_API_KEY=<your_api_key>" > .env
```

---

## 2. Pull Datasets and Containers

```bash
# Download pre-processed datasets from S3
bash scripts/pull_all_datasets.sh

# Pull pre-built Docker images for deep learning models
bash scripts/pull_all_models.sh

# Build PRESAGE manually (not redistributed due to license restrictions)
bash docker/presage/build.sh

# Build PDAE container
bash docker/pdae/build.sh
```

By default, `pull_all_datasets.sh` downloads a subset of datasets. Edit the `DATASETS` array in the script to include all datasets needed for your analyses. The seven datasets used in the paper are: `adamson16`, `frangieh21`, `nadig25hepg2`, `norman19`, `replogle20`, `replogle22k562`, `wessels23`.

### Prepare `replogle20` from raw files

`replogle20` is not included in the S3 pull. It has not been used in any combinatorial extrapolation benchmarking frameworks. To build it locally, place the following files in `data/replogle20/` (download and extract from [GEO GSM4367984](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4367984)):

- `GSM4367984_exp6.matrix.mtx`
- `GSM4367984_exp6.features.tsv`
- `GSM4367984_exp6.barcodes.tsv`
- `GSM4367984_exp6.cell_identities.csv`

Then run:

```bash
uv run python data/replogle20/get_data.py
```

---

## 3. Compute DEG Weights

DEG-based weights are required for weighted metrics (wMSE, wPearson, wPDS) during benchmarking when using the `vscontrol` or `synthetic` weight source. Run the standalone DEG weight computation before benchmarking:

```bash
# Compute DEG weights for all paper datasets (both control and synthetic modes)
uv run python scripts/compute_deg_weights.py \
  --datasets adamson16 frangieh21 nadig25hepg2 norman19 replogle20 replogle22k562 wessels23 \
  --mode both

# Or for specific datasets/modes
uv run python scripts/compute_deg_weights.py --datasets wessels23 --mode control
```

Weights are cached under `analyses/perturbation_discrimination/results/<dataset>/deg_scanpy/` and reused by both the benchmarking pipeline and the discrimination analyses.

The weights originally included in the cellsimbench repository (`vsrest`) require no additional setup, as they are computed during dataset preprocessing and stored in the preocessed dataset.

---

## 4. Train and Benchmark Models

All training and benchmarking is driven through the `cellsimbench` CLI with Hydra configuration.

```bash
# Train a single model on a dataset
uv run cellsimbench train model=pdae dataset=norman19

# Benchmark (predict + evaluate)
uv run cellsimbench benchmark model=pdae dataset=norman19
```

**Useful options:**

```bash
# Store stdout and stderr in a log file (replace the command as needed)
uv run cellsimbench train model=pdae dataset=norman19 > train.log 2>&1
# or:  your_command > logfilename.log 2>&1

# Sequential fold processing (default is parallel across GPUs)
uv run cellsimbench benchmark model=pdae dataset=norman19 execution.parallel_folds=false
# or:  uv run cellsimbench <train|benchmark> ... execution.parallel_folds=false

# Restrict to specific GPUs
CUDA_VISIBLE_DEVICES=0,1 uv run cellsimbench train model=gears dataset=wessels23

# Change DEG weight source for all weighted metrics
uv run cellsimbench benchmark model=pdae dataset=norman19 metrics.deg_weight_source=vscontrol
```

**Checking memory use:** `du -h --max-depth=2`

**Train and benchmark all models on all paper datasets:**

```bash
bash scripts/train_all_models.sh
```

This trains all deep learning models and runs the modelgroup benchmarks for both combinatorial and single perturbation datasets.

**Builtin baselines** (epistasis, linear regression, target scaling) are trained and predicted on-the-fly during benchmarking and do not require a separate `train` step.

---

## 5. Visualize Results

After benchmarking, each run produces a `detailed_metrics.csv` in its output directory under `outputs/`. Generate summary figures from these:

```bash
# Single benchmark run
python scripts/plot_multimodel_summary.py outputs/<benchmark_dir>/<timestamp>/detailed_metrics.csv
```

For multi-model comparisons, use the `modelgroup` configuration:

```bash
# Benchmark multiple models together
uv run cellsimbench benchmark modelgroup=models_combo dataset=wessels23
uv run cellsimbench benchmark modelgroup=models_single dataset=frangieh21

# Generate comparison figures
python scripts/plot_multimodel_summary.py outputs/<benchmark_dir>/<timestamp>/detailed_metrics.csv
```

Figures are saved under `additional_results/` within the benchmark output directory.

---

## 6. Perturbation Discrimination Analyses

The perturbation discrimination analyses (Signal pillar of the SBB framework) are documented in detail in [analyses/perturbation_discrimination/discrimination_analysis.md](analyses/perturbation_discrimination/discrimination_analysis.md).

Below is a summary of the key analyses and how they map to the paper.

### 6.1 BDS and PDS Heatmaps (Fig. 4a, 4b)

The BDS/PDS heatmaps are derived from benchmarking results (technical-duplicate baseline scores in `detailed_metrics.csv`, and metric calibration logs). After running benchmarks (§4), generate the heatmaps with:

```bash
python scripts/plot_technical_baseline_diagnostics.py
```

### 6.2 BDS and PDS Permutation Tests (Supplementary)

Permutation tests for whether metrics can discriminate perturbations from controls (BDS) and from each other (PDS), used in the supplementary metric comparison:

```bash
# Run both tests for all paper datasets (from root directory). 
# Make sure step 3 for computing deg weights has been performed.  
bash scripts/launch_pds_bds_full.sh
```

Or run individually:

```bash
# PDS test
uv run python analyses/perturbation_discrimination/run_pds_test.py \
  --datasets adamson16 frangieh21 norman19 replogle20 replogle22k562 wessels23 nadig25hepg2 \
  --n-permutations 999 --n-trials 1 --max-perturbations 500 --workers 64

# BDS test
uv run python analyses/perturbation_discrimination/run_bds_test.py \
  --datasets adamson16 frangieh21 norman19 replogle20 replogle22k562 wessels23 nadig25hepg2 \
  --n-permutations 999 --n-trials 1 --max-perturbations 500 --workers 64
```

### 6.3 Effective Genes Curve (Fig. 5a, 5b)

Sweep effective gene count to show how DEG weighting recovers signal:

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode effective_genes_curve \
  --datasets wessels23 \
  --metrics wmse,w_r2_delta,w_pearson_delta,weighted_energy_distance \
  --effective-genes-values auto_log10 \
  --target-effective-genes-mode mean \
  --weight-source deg --weight-scheme normal \
  --n-trials 3 --trial-cell-budget 100 --seed 42 --n-jobs 8
```

### 6.4 Bag-Size Curve (Fig. 5c)

Quantify how increasing bag size (cells per pseudo-replicate) improves discrimination:

```bash
uv run python analyses/perturbation_discrimination/run_signal_dilution_curves.py \
  --mode bag_size_curve \
  --datasets wessels23 \
  --metrics mse,wmse,r2_delta,w_r2_delta,pearson_delta,w_pearson_delta \
  --bag-sizes 2,5,10,15,20,30,40,50,70,90 \
  --n-trials 3 --trial-cell-budget 100 --seed 42 \
  --weight-source deg --weight-scheme normal --n-jobs 8
```

### 6.5 Per-Gene Analysis with Metric and Filtering Comparison

Compare filtering strategies (HVG, DEG, PDS-ranked genes) for perturbation discrimination:

```bash
# Compute full PDS tensors
uv run python analyses/perturbation_discrimination/compute_full_pds.py \
  --datasets norman19 \
  --n-trials 5 --metrics MAE_Mean Energy_Distance Wasserstein_1D

# Compare filtering methods
uv run python analyses/perturbation_discrimination/discrimination_comparison.py \
  --datasets norman19 wessels23 --n-genes 1000

# Visualize
uv run python analyses/perturbation_discrimination/visualize_results.py \
  --datasets norman19 wessels23
```

See `[analyses/perturbation_discrimination/discrimination_analysis.md](analyses/perturbation_discrimination/discrimination_analysis.md)` for all modes, options, and interpretation guidance.

---

## 7. Available Models, Datasets, Metrics, and Baselines

### Deep Learning Models


| Config name       | Method                       | Reference                |
| ----------------- | ---------------------------- | ------------------------ |
| `gears`           | GEARS                        | Roohani et al. 2024      |
| `presage`         | PRESAGE                      | Littman et al. 2025      |
| `scgpt`           | scGPT                        | Cui et al. 2024          |
| `sclambda`        | scLAMBDA                     | Wang et al. 2025         |
| `pdae`            | PDAE                         | von Kügelgen et al. 2025 |
| `fmlp_esm2`       | fMLP (ESM2 embeddings)       | Miller et al. 2025       |
| `fmlp_geneformer` | fMLP (Geneformer embeddings) | Miller et al. 2025       |
| `fmlp_scgpt`      | fMLP (scGPT embeddings)      | Miller et al. 2025       |
| `fmlp_genept`     | fMLP (GenePT embeddings)     | Miller et al. 2025       |


### Baselines

**Unlearned baselines** (no training, computed from data):

- Control mean, dataset mean, additive, technical duplicate, interpolated duplicate

**Learned baselines** (fit on training data):


| Config name         | Paper name        | Type                                                                                    |
| ------------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `linear_regression` | Linear Regression | One-hot encoded linear model (combo)                                                    |
| `epistasis_shared`  | Global Epistasis  | Shared scaling of additive effect + gene-wise offset (combo)                            |
| `target_scaling`    | Target Scaling    | Learned penetrance fraction for target gene and dataset mean for rest of genes (single) |


**Oracles** (fit on test data):


| Config name             | Paper name            | Type                                                                                |
| ----------------------- | --------------------- | ----------------------------------------------------------------------------------- |
| `epistasis_specific`    | Specific Epistasis    | Pair-specific scaling coefficients (combo)                                          |
| `target_scaling_oracle` | Target Scaling Oracle | Ground-truth expression for target gene and dataset mean for rest of genes (single) |


### Datasets

Seven datasets are used in the paper:


| Config name      | Perturbation type | Reference            |
| ---------------- | ----------------- | -------------------- |
| `wessels23`      | Double (combo)    | Wessels et al. 2023  |
| `norman19`       | Double (combo)    | Norman et al. 2019   |
| `replogle20`     | Double (combo)    | Replogle et al. 2020 |
| `frangieh21`     | Single            | Frangieh et al. 2021 |
| `adamson16`      | Single            | Adamson et al. 2016  |
| `replogle22k562` | Single            | Replogle et al. 2022 |
| `nadig25hepg2`   | Single            | Nadig et al. 2025    |


Additional datasets available for extended evaluation: `kaden25fibroblast`, `kaden25rpe1`, `nadig25jurkat`, `replogle22k562gwps`, `replogle22rpe1`, `sunshine23`, `tian21crispra`, `tian21crispri`.

### Key Metrics


| Metric                                 | Variant                   | Description                                                               |
| -------------------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| MSE / wMSE                             | Unweighted / DEG-weighted | Mean squared error on perturbation means                                  |
| Pearson_DeltaPert / wPearson_DeltaPert | Unweighted / DEG-weighted | Pearson correlation on expression deltas (with dataset mean as reference) |
| R2_DeltaPert / wR2_DeltaPert           | Unweighted / DEG-weighted | R-squared on expression deltas                                            |
| PDS (MSE) / wPDS (MSE)                 | Unweighted / DEG-weighted | Perturbation Discrimination Score                                         |


For each metric family in the table, a perturbation-wise calibrated variant is available in the codebase.

**List of available metrics:**

`mse`, `wmse`, `pearson_deltactrl`, `pearson_deltactrl_degs`, `r2_deltactrl`, `r2_deltactrl_degs`, `weighted_pearson_deltactrl`, `weighted_r2_deltactrl`, `pearson_deltapert`, `pearson_deltapert_degs`, `weighted_pearson_deltapert`, `r2_deltapert`, `r2_deltapert_degs`, `weighted_r2_deltapert`, `pds`, `pds_wmse`, `pds_pearson_deltapert`, `pds_weighted_pearson_deltapert`, `pds_r2_deltapert`, `pds_weighted_r2_deltapert`.

Metrics are configured in `cellsimbench/configs/config.yaml` under `metrics.enabled`. The full set of names (including calibrated and gene-filtered variants) is enumerated in `cellsimbench/core/metrics_engine.py` (`metric_order`).

---

## 8. Building from Scratch

For users who want to rebuild datasets and Docker containers from source rather than using pre-built versions.

### Data Preparation

```bash
# Download and preprocess all datasets
python data/run_all_get_data.py --workers 4

# Calculate ground-truth DEGs (first half of technical duplicates)
python data/add_ground_truth_degs.py --all --workers 4

# Add interpolated duplicate baseline
python data/add_interpolated_baseline.py --all --workers 4
```

To build `replogle20` from GEO raw files instead of S3, see [Prepare `replogle20` from raw files](#prepare-replogle20-from-raw-files) in §2.

---

## Licensing

The core framework is released under the MIT License. Certain benchmark components use third-party models under their original terms:

- `docker/presage/` -- Genentech Non-Commercial Software License v1.0
- `docker/sclambda/` -- GNU General Public License v3.0 (derivative work)

---

## 9. Docker with Podman (cluster nodes)

On some clusters (for example environments where `docker` is a Podman-compatible CLI), the Python code is unchanged, but the Docker client must see a running container socket and model images must exist under the expected local names (for example `cellsimbench/gears:latest`).

1. **Docker SDK and socket.** Point the client at the user-level Podman socket, for example:
  - Set `DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock`
  - Ensure the directory exists: `mkdir -p /run/user/$(id -u)/podman`
  - If needed, start the service: `podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock`  
   You can add these to `~/.bashrc` on the node so new shells pick them up automatically.
2. **One-time check** in a fresh shell:

```bash
echo "$DOCKER_HOST"
docker version
uv run python - <<'PY'
import docker
c = docker.from_env()
print("ping:", c.ping())
PY
```

You should see `ping: True`.

---

## 10. Citation

If you use this code, please cite:

```bibtex
@article{vollenweider2026sbb,
  title={Signal, Bounds, and Baselines: Principles for Rigorous Evaluation of High-Dimensional Biological Perturbation Prediction},
  author={Vollenweider, Michael and B{\"u}hlmann, Peter},
  journal={bioRxiv},
  year={2026}
}
```

This codebase builds upon CellSimBench; please also cite:

```bibtex
@article{miller2025perturbation,
  title={Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics},
  author={Miller, Henry E. and Mejia, Gabriel M. and Leblanc, Francis J. A. and Swain, Brendan and Wang, Bo and Camillo, Lucas Paulo de Lima},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.10.20.683304}
}
```

## Contact

For questions, contact: [michael.vollenweider@stat.math.ethz.ch](mailto:michael.vollenweider@stat.math.ethz.ch)