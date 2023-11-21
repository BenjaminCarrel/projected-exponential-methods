"""
File for the motivation figure - comparison between projected methods and existing methods.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% IMPORTATIONS
from graphics_parameters import *
from problems import make_lyapunov_heat_square_dirichlet
import numpy as np
from low_rank_toolbox import SVD
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
import time

#%% SETUP THE ODE
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_dirichlet


#%% METHODS PARAMETERS

# Methods parameters
dlra_solvers = []
methods_kwargs = []
methods_labels = []
methods_styles = []

# Methods parameters - Projector splitting (Lie-Trotter)
dlra_solvers += ['KSL']
methods_kwargs += [{'order': 1, 'substep_kwargs': {'solver': 'explicit_runge_kutta', 'order': 4, 'nb_substeps': 1000}}]
methods_labels += ['Proj. splitting (Lie-Trotter)']
methods_styles += ['-v']

# Methods parameters - Unconventional
dlra_solvers += ['unconventional']
methods_kwargs += [{'substep_kwargs': {'solver': 'explicit_runge_kutta', 'order': 4, 'nb_substeps': 1000}}]
methods_labels += ['Unconventional']
methods_styles += ['-^']

# Methods parameters - Low-rank splitting (Lie-Trotter)
dlra_solvers += ['low_rank_splitting']
methods_kwargs += [{'order': 1}]
methods_labels += ['Low-rank splitting (Lie-Trotter)']
methods_styles += ['-+']

# Methods parameters - New method (projected exponential Euler)
dlra_solvers += ['PERK']
methods_kwargs += [{'order': 1, 'krylov_kwargs': {'size': 1, 'kind': 'extended', 'is_symmetric': True}}]
methods_labels += ['Proj. exponential Euler']
methods_styles += ['-o']

#%% COMPUTE THE SOLUTIONS
# Parameters
nb_steps = 100
ts = np.linspace(t_span[0], t_span[1], nb_steps+1)
sizes = [32, 64, 96, 128, 192, 256, 384, 512]
rank = 20

# Preallocate the errors
errors = np.zeros((len(sizes), len(dlra_solvers)))
approx_errors = np.zeros(len(sizes))
# cond_numbers = np.zeros(len(sizes))

## LOOP OVER THE SIZES
for i, n in enumerate(sizes):
    ## MAKE THE ODE
    print('*********************************************************************************')
    print("Solving for n = {}".format(n))
    ode, X0 = make_ode(n)
    X0 = SVD.reduced_svd(X0)
    Y0 = SVD.truncated_svd(X0, rank)
    # cond_numbers[i] = la.cond(ode.A.todense())

    ## COMPUTE REFERENCE SOLUTION
    X1 = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=ts, monitor=True, dense_output=True).X1
    approx_errors[i] = np.linalg.norm(X1 - SVD.truncated_svd(X1, rank).todense(), 'fro') / np.linalg.norm(X1, 'fro')

    ## LOOP OVER THE METHODS
    for j, method in enumerate(dlra_solvers):
        # COMPUTE THE SOLUTION WITH THE CURRENT METHOD
        try:
            Y1 = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=ts, monitor=True, solver_kwargs=methods_kwargs[j]).X1

            # COMPUTE THE RELATIVE ERROR
            errors[i, j] = np.linalg.norm(Y1.todense() - X1, 'fro') / np.linalg.norm(X1, 'fro')
        except:
            # Some methods may not work for stiff problems
            errors[i, j] = 1e10

# %% ROBUST TO STIFFNESS - PLOT
fig = plt.figure()
for j, method in enumerate(dlra_solvers):
    plt.semilogy(sizes, errors[:, j], methods_styles[j], label=methods_labels[j])
plt.semilogy(sizes, approx_errors, '--', label=f'Best approx. error ($r={rank}$)')
# plt.tight_layout()
# x-axis in log scale power of 2
plt.xticks(sizes, sizes)
plt.legend()
plt.ylim([1e-18, 1e0])
plt.xlabel("Size (mesh refinement)")
plt.ylabel("Relative error in Frobenius norm")
plt.show()

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
fig.savefig(f'figures/{X0.shape}_sizes_{sizes}_errors_methods_{methods_labels}_without_cond_numbers_{timestamp}.pdf', bbox_inches='tight')

# %%
