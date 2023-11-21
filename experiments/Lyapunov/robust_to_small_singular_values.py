"""
File for plotting the global error of one method for several ranks.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Importations
from graphics_parameters import *
from problems import make_lyapunov_heat_square_with_time_dependent_source
import numpy as np
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra
from scipy.sparse import linalg as spala
import time

#%% Setup the ODE
size = 128
q = 5
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_with_time_dependent_source
ode, X0 = make_ode(size, q)

# Print the ode defined in setup_problem.py
print(ode)
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)

# Preprocess the problem
invA = spala.splu(ode.A).solve
invB = spala.splu(ode.B).solve

#%% GLOBAL ERROR - PARAMETERS

# Ranks to test
ranks = [4, 7, 11, 16]

# Number of steps
list_nb_steps = np.logspace(1, 3, 4, dtype=int)

# Method parameters - PERK
method = 'PERK'
invA = spala.inv(ode.A).dot
invB = spala.inv(ode.B).dot
krylov_kwargs = {'size': 1, 
                 'kind': 'extended', 
                 'is_symmetric': True, 
                 'invA': invA, 
                 'invB': invB}
method_kwargs = {'order': 2, 
                 'krylov_kwargs': krylov_kwargs, 
                 'strict_order_conditions': True}

# Preallocation
global_errors = np.zeros((len(list_nb_steps), len(ranks)))
approx_error = np.zeros((len(list_nb_steps), len(ranks)))
times = np.zeros((len(list_nb_steps), len(ranks)))

#%% GLOBAL ERROR - COMPUTATIONS
# Loop over the number of steps
for i, nb in enumerate(list_nb_steps):
    print('*************************************************************************')
    print(f'Solving with {nb} steps. ({i+1}/{len(list_nb_steps)})')
    t_eval = np.linspace(*t_span, nb+1)

    # Compute the reference solution
    X1 = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=t_eval, dense_output=True, monitor=True).X1

    for rank in ranks:
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(f'Rank {rank}')
        t_eval = np.linspace(*t_span, nb+1)

        # Compute the best rank approximation
        approx_error[i, ranks.index(rank)] = np.linalg.norm(X1 - SVD.truncated_svd(X1, rank).todense(), 'fro') / np.linalg.norm(X1, 'fro')

        # Compute the solution with the DLRA method
        Y0 = SVD.truncated_svd(X0, rank)
        t0 = time.time()
        Y1 = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=t_eval, monitor=True, solver_kwargs=method_kwargs).X1
        times[i, ranks.index(rank)] = time.time() - t0

        # Compute the relative error
        global_errors[i, ranks.index(rank)] = np.linalg.norm(Y1.todense() - X1, 'fro') / np.linalg.norm(X1, 'fro')

#%% GLOBAL ERROR - PLOT
fig = plt.figure()
stepsizes = t_span[1] / list_nb_steps
# One color per rank
names = [f'Rank {r}' for r in ranks]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown'] # , 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
styles = ['-o', '-s', '-p', '-D']
for j, rank in enumerate(ranks):
    plt.loglog(stepsizes, global_errors[:, j], styles[j], label=names[j], color=colors[j])
    plt.loglog(stepsizes, approx_error[:, j], '--', color=colors[j])
plt.loglog(stepsizes, stepsizes**2, color='k')
plt.legend()
plt.xlabel("Step size")
plt.ylabel("Relative error in Frobenius norm")
plt.ylim(1e-9, 1e-1)
plt.tight_layout()
plt.show()


fig.savefig(f'figures/{X0.shape}_global_error_T_{t_span[1]}_ranks_{ranks}_nb_steps_{list_nb_steps}.pdf', bbox_inches='tight')


# %% PERFORMANCE - PLOT
fig = plt.figure()
for j, rank in enumerate(ranks):
    plt.loglog(global_errors[:, j], times[:, j], styles[j], label=names[j], color=colors[j])
    plt.axvline(approx_error[0, j], linestyle='--', color=colors[j])
plt.legend()
plt.xlabel("Relative error in Frobenius norm")
plt.ylabel("Time of computation (s)")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

fig.savefig(f'figures/{X0.shape}_performance_T_{t_span[1]}_ranks_{ranks}_nb_steps_{list_nb_steps}.pdf', bbox_inches='tight')

# %%
