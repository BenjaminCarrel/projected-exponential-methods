## Lyapunov experiments

This folder contains the code for reproducing the experiments related to the Lyapunov equation.

### Instructions

You can reproduce the experiments in the paper by simply running the corresponding file (see the list below).
The figures are saved in the `figures` folder (which is created if it does not exist).
Some of the experiments are computationally expensive and may take a long time to run (several hours).
The parameters for each problem are set at the beginning of each file.

### Parameters

- The file `graphics_parameters.py` contains the parameters for the graphics.
- The file `problems.py` contains functions for generating the problems used in the experiments.
  
### Experiments

- The file `motivating_example.py` contains the code for the motivating example (Figure 1) in the paper.
- The file `robust_to_stiffness.py` contains the code for showing the robustness to stiffness property (Figure 2) in the paper.
- The file `robust_to_small_singular_values.py` contains the code for showing the robustness to small singular values property (Figure 3) in the paper.
- The file `global_error_and_performance.py` contains the code for computing the global error convergence and performance (Figure 4) in the paper.
- The file `rank_adaptive_experiment.py` contains the code for the rank adaptive experiment (Figure 8 and 9) in the paper.