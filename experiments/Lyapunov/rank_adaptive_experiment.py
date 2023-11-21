"""
File for testing the rank-adaptive projected exponential integrator

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Importations
from graphics_parameters import *
from problems import make_lyapunov_heat_square_with_time_dependent_adaptive
import numpy as np
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_adaptive_dlra
import time
nb_steps = 100

#%% Setup the ODE
size = 128
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_with_time_dependent_adaptive
ode, X0 = make_ode(size)
# Print the ode defined in setup_problem.py
print(ode)
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)
print('Dimensions: ', X0.shape)

#%% Reference solution
ts = np.linspace(*t_span, nb_steps+1)
ref_sol = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=ts, monitor=True)
Xs_ref = ref_sol.todense()

#%% Adaptive PERK - Parameters

# Define the methods
from scipy.sparse import linalg as spala
invA = spala.inv(ode.A).dot
dlra_solver = 'adaptive_PERK'
krylov_kwargs = {'size': 1, 'kind': 'extended', 'invA': invA, 'invB': invA}
method_kwargs = {'order': 2, 'krylov_kwargs': krylov_kwargs, 'strict_order_conditions': True}
tolerances = [1e-5, 1e-8, 1e-11]

# Initialize the error over time matrix
nb_t_steps = len(ref_sol.Xs)
dlra_solutions = np.zeros(len(tolerances), dtype=object)

#%% Adaptive PERK - Computations
# Loop over the tolerances
for j, tol in enumerate(tolerances):
    # Compute the solution with the current method
    Y0 = X0.truncate(rtol = tol)
    dlra_solutions[j] = solve_adaptive_dlra(ode, t_span, Y0, adaptive_dlra_solver=dlra_solver, solver_kwargs=method_kwargs, monitor=True, t_eval=ts, rtol=tol, atol=None)


#%% Adaptive PERK - Compute and plot the error over time (in 2-norm)
# Compute the error over time
errors = np.zeros((nb_t_steps, len(tolerances)))
for i in range(nb_t_steps):
    for j, tol in enumerate(tolerances):
        errors[i, j] = np.linalg.norm(Xs_ref[i] - dlra_solutions[j].Xs[i].todense(), 'fro') / np.linalg.norm(Xs_ref[i], 'fro')

# Plot the error over time
fig = plt.figure()
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, tol in enumerate(tolerances):
    plt.semilogy(ts, errors[:, i], label=f'Tolerance {tol}', color=colors[i])
    plt.axhline(tol, linestyle='--', color=colors[i])
plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Relative error in Frobenius norm")
plt.ylim(1e-12, 1e-1)
plt.show()

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
fig.savefig(f'figures/{X0.shape}_adaptive_PERK_error_with_tolerances_{tolerances}_and_{nb_steps}_steps_{timestamp}.pdf', bbox_inches='tight')

#%% Adaptive PERK - Extract and plot the rank over time
ranks = np.zeros((nb_t_steps, len(tolerances)))
for i in np.arange(0, nb_t_steps):
    for j, tol in enumerate(tolerances):
        ranks[i, j] = dlra_solutions[j].Xs[i].rank

fig = plt.figure()
for i, tol in enumerate(tolerances):
    plt.plot(ts, ranks[:, i], label=f'Tolerance {tol}', color=colors[i])
plt.legend(loc = 'upper left')
plt.ylim(0, 50)
plt.xlabel("Time")
plt.ylabel("Rank")
plt.show()

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
fig.savefig(f'figures/{X0.shape}_adaptive_PERK_rank_with_tolerances_{tolerances}_and_{nb_steps}_steps_{timestamp}.pdf', bbox_inches='tight')



# %%
