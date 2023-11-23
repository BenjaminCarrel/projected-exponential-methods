## Riccati experiments

This folder contains the code for reproducing the experiments related to the Riccati equation.

### Instructions

You can reproduce the experiments in the paper by simply running the corresponding file (see the list below).
The figures are saved in the `figures` folder (which is created if it does not exist).
Some of the experiments are computationally expensive and may take a long time to run (several hours).
The parameters for each problem are set at the beginning of each file.

### Parameters

- The file `graphics_parameters.py` contains the parameters for the graphics.
- The file `problems.py` contains functions for generating the problems used in the experiments.

### Experiments

- The file `global_error_and_performance.py` contains the code for computing the global error convergence and performance (Figure 5) in the paper.
- The files `krylov_approx_first_order.py` and `krylov_approx_second_order.py` contain the code for the Krylov approximation experiments (Figure 6) in the paper.