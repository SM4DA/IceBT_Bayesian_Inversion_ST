# Bayesian Inference: Kernel-Based Model for Surface Temperature Reconstruction in Ice Borehole Thermometry

This repository provides the data and code supporting our work titled **"Bayesian Inference: Kernel-Based Model for Surface Temperature Reconstruction in Ice Borehole Thermometry"** by K. Shaju, T. Laepple, and P. Zaspel.

## Description

This repository contains the following directories and files:

- `Forward_Model/` – Heat transfer model.

- `Data/` – Data generation.

- `Inverse_Model/` – Jobs_sbatch_codes, emcee_inversion_codes, and RJMCMC_inversion_codes.

- `Results/` – Final results 

- `Plots_case_study.ipynb`  – Case study section plots.

- `Plots_results.ipynb`  – Result section plots.

- `README.md` – Project overview and instructions (that's me!).


### Data generation

The data consist of borehole temperature–depth profiles simulated using a heat transfer model based on various surface temperature time series. The data (borehole measurements), code for generating data, along with the surface temperature time series data, is organized in the `Data/` directory.

* `Data/` contains considered timeseries, simulated measurements for both results and casestudy section.

* `Synthetic signals.ipynb` demonstrates the borehole temperature profile simulation for results section.

* `Realistic signals.ipynb` demonstrates the borehole temperature profile simulation for case study.

The simulated measurements for Synthetic signals in Results section are stored under `Data/Gaussian_pulse_signals/Artificial_measurements/` and realistic measurements for case study are stored under `Data/case_study/Realistic_artificial_measurements/`.

### Inverse model

The code used to produce the results presented in the paper are organized in the `Inverse model/` directory. This directory includes the following subdirectories and files:

* `Jobs_sbatch_codes/` contains sbatch job scripts for running emcee inverse codes for the experiments performed in the paper.

* `emcee_inversion_codes/` contains emcee inversion codes and sampler result generation codes.

* `RJMCMC_inversion_codes/` contains RJMCMC inversion codes and sampler result generation codes.

### Results

* `Results/` contains final results which are used to create the figures in the paper.

### Plots

Locating the corresponding results from `Results` to produce plots presented in our work are detailed in `Plots_results.ipynb` and `Plots_case_study.ipynb`. 

### Instructions

`README.md` - That’s me! 🙂 Your guide to the repository, with an overview and usage instructions.
