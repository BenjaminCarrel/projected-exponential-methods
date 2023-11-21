"""
Global error and performance of DLRA methods for the Riccati ODE.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Importations
from graphics_parameters import *
import numpy as np
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from problems import make_riccati_ostermann
import scipy.sparse.linalg as spala
import time

#%% Setup the ODE
size = 200
q = 9
t_span = (0, 0.1)
make_ode = lambda size: make_riccati_ostermann(size, q)
ode, X0 = make_ode(size)

# Print the ode defined in setup_problem.py
print(ode)
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)

# Preprocess the problem
invA = spala.splu(ode.A).solve
invB = spala.splu(ode.B).solve

#%% DLRA PARAMETERS

# DLRA Initial value
rank = 20
Y0 = X0.truncate(rank)

# Methods parameters
dlra_solvers = []
methods_kwargs = []
methods_labels = []
methods_styles = []

# Methods parameters - PERK
## Projected exponential Euler
dlra_solvers += ['PERK']
methods_kwargs += [{'order': 1, 'krylov_kwargs': {'size': 1, 'kind': 'extended', 'is_symmetric': True, 'invA': invA, 'invB': invB}}]
methods_labels += ['Proj. exponential Euler']
methods_styles += ['-o']
## Projected exponential Runge (strict)
dlra_solvers += ['PERK']
methods_kwargs += [{'order': 2, 'krylov_kwargs': {'size': 1, 'kind': 'extended', 'is_symmetric': True, 'invA': invA, 'invB': invB}, 'strict_order_conditions': True}]
methods_labels += ['Proj. exponential Runge (strict)']
methods_styles += ['-x']

# Methods parameters - LOW-RANK SPLITTING
## Low-rank splitting (Lie-Trotter)
dlra_solvers += ['low_rank_splitting']
methods_kwargs += [{'order': 1}]
methods_labels += ['Low-rank splitting (Lie-Trotter)']
methods_styles += ['-s']
## Low-rank splitting (Strang)
dlra_solvers += ['low_rank_splitting']
methods_kwargs += [{'order': 2}]
methods_labels += ['Low-rank splitting (Strang)']
methods_styles += ['-+']


#%% GLOBAL ERROR AND PERFORMANCE - COMPUTATIONS

# Number of steps
nb_steps = np.logspace(1, 3, 4, dtype=int)
stepsizes = t_span[1] / nb_steps

# Preallocate
global_errors = np.zeros((len(nb_steps), len(dlra_solvers)))
times = np.zeros((len(nb_steps), len(dlra_solvers)))

# Loop over the number of steps
for i, nb in enumerate(nb_steps):
    print('*************************************************************************')
    print(f'Solving with {nb} steps. ({i+1}/{len(nb_steps)})')
    t_eval = np.linspace(*t_span, nb+1)

    # Compute the reference solution
    X1 = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=t_eval, dense_output=True, monitor=True).X1
    
    # Loop over the methods
    for j, method in enumerate(dlra_solvers):
        # Compute the solution with the current method
        t0 = time.time()
        Y1 = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=t_eval, monitor=True, solver_kwargs=methods_kwargs[j]).X1
        times[i, j] = time.time() - t0

        # Compute the relative error
        global_errors[i, j] = np.linalg.norm(Y1.todense() - X1, 'fro') / np.linalg.norm(X1, 'fro')

# Approximate error
best_approx_error = np.linalg.norm(X1 - SVD.truncated_svd(X1, rank).todense(), 'fro') / np.linalg.norm(X1, 'fro')

#%% GLOBAL ERROR - PLOT
fig = plt.figure()
for j, method in enumerate(dlra_solvers):
    plt.loglog(stepsizes, global_errors[:, j], methods_styles[j], label=methods_labels[j])
plt.loglog(stepsizes, 4*stepsizes, 'k') # , label=r'$O(h)$'
plt.loglog(stepsizes, 4*stepsizes**2, 'k') # , label=r'$O(h^2)$'
plt.axhline(best_approx_error, linestyle='--', color='gray', label=f'Best approx. error ($r={rank}$)')
plt.legend(loc='upper left')
plt.xlabel("Step size")
plt.ylabel("Relative error in Frobenius norm")
plt.ylim([1e-9, 1e3])
plt.tight_layout()
plt.show()

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
fig.savefig(f'figures/{X0.shape}_global_error_T_{t_span[1]}_rank_{rank}_nb_steps_{nb_steps}_{timestamp}.pdf', bbox_inches='tight')

#%% PERFORMANCE - PLOT
fig = plt.figure()
for j, method in enumerate(dlra_solvers):
    plt.loglog(global_errors[:, j], times[:, j], methods_styles[j], label=methods_labels[j])
plt.axvline(best_approx_error, linestyle='--', color='gray', label=f'Best approx.error ($r={rank}$)')
plt.legend(loc='upper left')
plt.gca().invert_xaxis()
plt.xlabel("Relative error in Frobenius norm")
plt.ylabel("Time of computation (s)")
plt.tight_layout()
plt.show()

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
fig.savefig(f'figures/{X0.shape}_perf_T_{t_span[1]}_rank_{rank}_nb_steps_{nb_steps}_{timestamp}.pdf', bbox_inches='tight')

# %%
