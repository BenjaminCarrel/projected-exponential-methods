# Projected exponential methods for stiff dynamical low-rank approximation problems

## Abstract

The numerical integration of stiff equations is a challenging problem that needs to be approached by specialized numerical methods. Exponential integrators form a popular class of such methods since they are provably robust to stiffness and have been successfully applied to a variety of problems. The dynamical low- rank approximation is a recent technique for solving high-dimensional differential equations by means of low-rank approximations. However, the domain is lacking numerical methods for stiff equations since existing methods are either not robust- to-stiffness or have unreasonably large hidden constants.
In this paper, we focus on solving large-scale stiff matrix differential equations with a Sylvester-like structure, that admit good low-rank approximations. We propose two new methods that have good convergence properties, small memory footprint and that are fast to compute. The theoretical analysis shows that the new methods have order one and two, respectively. We also propose a practical implementation based on Krylov techniques. The approximation error is analyzed, leading to a priori error bounds and, therefore, a mean for choosing the size of the Krylov space. Numerical experiments are performed on several examples, confirming the theory and showing good speedup in comparison to existing techniques.

## Authors

- [Benjamin Carrel](benjamin.carrel@unige.ch) (University of Geneva)
- Bart Vandereycken (University of Geneva)

## Reference

The paper has been submitted to a journal and is currently under review.
It is available on [arXiv](https://arxiv.org/abs/2312.00172) with DOI number arXiv:2312.00172.

To cite the preprint, you can use the following BibTeX entry:

```
@misc{carrel2023projected,
      title={Projected exponential methods for stiff dynamical low-rank approximation problems}, 
      author={Benjamin Carrel and Bart Vandereycken},
      year={2023},
      eprint={2312.00172},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

## Experiments

The experiments are contained in the folder `experiments`.
Those experiments are written in Python and each file corresponds to a figure in the paper and can be run independently.

All experiments can be run on a laptop, but experiments with small step sizes might take a long time (hours) to run. You can reduce the number of time steps to reduce the computation time.

The experiments in the paper were done with [Release 1.0](https://github.com/BenjaminCarrel/projected-exponential-methods/releases/tag/arXiv). 

If you have any question or issue while running the experiments, please contact me at
[benjamin.carrel@unige.ch](benjamin.carrel@unige.ch).

## Installation

Download the current version of the code from GitHub or download the [Release 1.0](https://github.com/BenjaminCarrel/projected-exponential-methods/releases/tag/arXiv).

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
