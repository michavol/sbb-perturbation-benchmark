#!/bin/bash
# Train all models on all datasets
# Usage: ./scripts/train_all_models.sh

# Train and benchmark all models on all datasets
for dataset in frangieh21 norman19 wessels23 adamson16 replogle20 adamson16 nadig25hepg2 replogle22k562; do
  for model in presage scgpt gears sclambda fmlp_esm2 fmlp_geneformer fmlp_scgpt fmlp_genept; do
    uv run cellsimbench train model=$model dataset=$dataset
    uv run cellsimbench benchmark model=$model dataset=$dataset
  done
done

# Benchmark all models on all datasets in model groups
for dataset in norman19 wessels23 replogle20; do
  uv run cellsimbench benchmark modelgroup=models_combo dataset=$dataset
done

for dataset in adamson16 frangieh21 nadig25hepg2 replogle22k562; do
  uv run cellsimbench benchmark modelgroup=models_single dataset=$dataset
done