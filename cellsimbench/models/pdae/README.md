# Perturbation Distribution Autoencoder (PDAE)

This repository contains the implementation of the [Perturbation Distribution Autoencoder (PDAE)](https://arxiv.org/abs/2504.18522) estimation method that can be used to predict the effect of unseen gene or drug perturbations on the transcriptome. 

## Overview
![pdae_full_v2.png](pdae_full_v2.png)

## Installation
- clone the `main` branch of the repository
- enter the repository directory, e.g. `cd PerturbationExtrapolation/`
- make sure to have `miniconda` installed, see https://www.anaconda.com/docs/getting-started/miniconda/install
- create conda environment `pdae` from requirements file: `conda env create --file environment.yml`

## Getting Started
- the notebook ```examples/simulation.ipynb``` shows how to prepare simulated data, train a PDAE model and evaluate it
- the notebook ```examples/norman.ipynb``` shows how to prepare the Norman data, train a PDAE model and evaluate it
- the notebook ```experiments/simulation/simulation_arxiv_paper.ipynb``` contains simulation results as published in the Arxiv [preprint](https://arxiv.org/abs/2504.18522).

## Key Features
PDAE provides functionality to 
- generate simulated data
- load real-world sequencing data from AnnData
- train PDAE models
- evaluate the performance of the trained model
- check whether extrapolation guarantees hold for your data
- check whether pdae model is trained to a point where an MMD test between true and predicted train distributions can't reject
- plot true vs. predicted observational and latent distributions

## Collaboration guideline
- please file issues if you encounter errors or unexpected behaviours
- please file pull requests if you want to add code to main for everyone to use

## Future releases
- [x] example notebook with PDAE training on Norman 2019 data
- [x] download of Norman data
- [x] license
- [ ] extending `PDAEData.from_adata` to accept and encode arbitrary covariate information

## Reference
If you use this code in your research, please cite the following [preprint](https://arxiv.org/abs/2504.18522):
```
@article{vonkügelgen2025representationlearningdistributionalperturbation,
      title={Representation Learning for Distributional Perturbation Extrapolation}, 
      author={Julius von Kügelgen and Jakob Ketterer and Xinwei Shen and Nicolai Meinshausen and Jonas Peters},
      year={2025},
      eprint={2504.18522},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2504.18522}, 
}
```

## Licence
[MIT License](https://github.com/Juliusvk/PerturbationExtrapolation?tab=MIT-1-ov-file#readme)
