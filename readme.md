# Neural Networks vs. Linear Factor Models in Daily Stock Return Prediction

Replication code for the NTNU Master's Thesis (TIØ4900, 2026)  
Author: Nikolaus Weidemann Wieland  

## Overview
This repository contains all code used to produce the results in the 
thesis. The analysis evaluates whether nonlinear neural network models 
improve one-day-ahead out-of-sample excess return prediction relative 
to the linear Fama-French three-factor benchmark, using a fixed 
information set and strict out-of-sample framework.

## Data
Data is not included in this repository due to licensing restrictions.
- Stock return data: CRSP via WRDS (https://wrds-www.wharton.upenn.edu)
- Factor data: Kenneth R. French Data Library 
  (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## Notebooks
| Notebook | Description |
|----------|-------------|
| `01_WRDS_pull.ipynb` | Data retrieval and construction from WRDS |
| `01.1_graphs_and_figures_dataset.ipynb` | Descriptive figures and dataset visualization |
| `02_ff3_vs_nn_debugging.ipynb` | Model development and debugging |
| `03_hyperparam_evaluation.ipynb` | Hyperparameter sweep and selection |
| `04_full_ff3_vs_nn.ipynb` | Main estimation (strict training sample) |
| `05_full_ff3_vs_nn_incl_val.ipynb` | Robustness: non-strict training sample |
| `06_forecast_tests_dm_cw.ipynb` | Diebold-Mariano forecast accuracy tests |
| `07_mean_vs_ff3.ipynb` | Historical mean benchmark comparison |
| `08_cross_sectional_analysis.ipynb` | Cross-sectional regression analysis |
| `09_XAI_analysis.ipynb` | Integrated Gradients attribution analysis |
| `10_full_ff3_vs_nn_unscaled_target.ipynb` | Robustness: no target scaling |

## Requirements
Python 3.x. Key dependencies: `torch`, `numpy`, `pandas`, 
`statsmodels`, `captum`, `wrds`.