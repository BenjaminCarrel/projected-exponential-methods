# Projected exponential methods for stiff dynamical low-rank approximation problems

## Abstract

(To be updated with the published abstract)

## Authors

- Benjamin Carrel (University of Geneva)
- Bart Vandereycken (University of Geneva)

## Reference

(To be updated when published)

## Experiments

The experiments are contained in the folder `experiments`.
Those experiments are written in Python and each file corresponds to a figure in the paper and can be run independently.

All experiments can be run on a laptop, but experiments with small step sizes might take a long time (hours) to run. You can reduce the number of time steps to reduce the computation time.

If you have any question or issue while running the experiments, please contact me at
benjamin.carrel@unige.ch.

## Installation

### Conda

If you have conda installed, you can easily create a conda environment with

`conda env create --file environment.yml`

Then, activate the environment with

`conda activate projected-exponential-methods`

Then, install the package with

`pip install --compile .`

You can update your environment with

`conda env update --file environment.yml --prune`

### Apple Silicon

If you have an Apple Silicon processor, you might want to speed up the computations by using Apple's Accelerate BLAS library.

To do so, after the conda installation, you have to run the following command:
```
conda install numpy scipy "libblas=*=*accelerate"
```

Note: I've noticed that the code runs faster when conda is installed with miniforge instead of anaconda.

### General settings

If you have other Python distribution, you can install the package by following these steps:

- Clone this repo on your computer
- Install Python (>=3.10) with the following packages:
  - numpy (>=1.22)
  - scipy (>=1.8)
  - matplotlib (>=3.5)
  - jupyter (>=1.0)
  - tqdm (>=4.63)
- Install the package with `pip install .` or `pip install --compile .` (for faster execution).
