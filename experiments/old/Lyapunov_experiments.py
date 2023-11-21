# %% [markdown]
# # Lyapunov experiments
# 
# Notebook for Lyapunov experiments.
# 
# **Author:** [Benjamin Carrel](benjamin.carrel@unige.ch)

# %% [markdown]
# ## Robust to stiffness - 1st example

# %% [markdown]
# ### Setup the problem

# %%
# Imports
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spala
import time
import matplotlib.pyplot as plt
from low_rank_toolbox import LowRankMatrix, SVD
from problems import make_lyapunov_heat_square_dirichlet
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra

# Problem parameters
size = 128
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_dirichlet
ode, X0 = make_ode(size)

# Print the ode
print(ode)

# %% [markdown]
# ### Setup the solvers

# %%
# The sizes
sizes = [32, 64, 96, 128, 192, 256, 384, 512]

# Other parameters
nb_steps = 100
ts = np.linspace(t_span[0], t_span[1], nb_steps+1)
rank = 20

# Define the methods (projector splitting comparison)
dlra_solvers = ['KSL', 'unconventional', 'low_rank_splitting', 'PERK']
krylov_kwargs = {'size': 1, 'kind': 'extended'} # this is overwritten later
substep_kwargs = {'solver': 'explicit_runge_kutta', 'order': 4, 'nb_substeps': 1000}
methods_kwargs = [{'order': 1, 'substep_kwargs': substep_kwargs},
                {'substep_kwargs': substep_kwargs},
                {'order': 1},
                {'order': 1, 'krylov_kwargs': krylov_kwargs}]

## PREALLOCATE THE ERRORS
approx_errors = np.zeros(len(sizes))
cond_numbers = np.zeros(len(sizes))
errors = np.zeros((len(sizes), len(dlra_solvers)))
times = np.zeros((len(sizes), len(dlra_solvers)))

# %% [markdown]
# ### Run the solvers

# %%
## LOOP OVER THE SIZES
for i, n in enumerate(sizes):
    ## MAKE THE ODE
    print('*********************************************************************************')
    print("Solving for n = {}".format(n))
    ode, X0 = make_ode(n)
    X0 = SVD.reduced_svd(X0)
    Y0 = SVD.truncated_svd(X0, rank)
    cond_numbers[i] = la.cond(ode.A.todense())

    ## COMPUTE REFERENCE SOLUTION
    X1 = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=ts, monitor=True, dense_output=True).X1
    approx_errors[i] = np.linalg.norm(X1 - SVD.truncated_svd(X1, rank).todense(), 'fro') / np.linalg.norm(X1, 'fro')

    ## LOOP OVER THE METHODS
    for j, method in enumerate(dlra_solvers):
        # COMPUTE THE SOLUTION WITH THE CURRENT METHOD
        try:
            t0 = time.time()
            Y1 = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=ts, monitor=True, solver_kwargs=methods_kwargs[j]).X1
            times[i, j] = time.time() - t0

            # COMPUTE THE RELATIVE ERROR
            errors[i, j] = np.linalg.norm(Y1.todense() - X1, 'fro') / np.linalg.norm(X1, 'fro')
        except:
            # Some methods may not work for stiff problems
            errors[i, j] = 1e10

# %% [markdown]
# ### Plot the results

# %%
# Plot the errors
fig1 = plt.figure()
method_names = ['Projector-splitting (Lie-Trotter)', 'Unconventional', 'Low-rank splitting (Lie-Trotter)', 'New method (order 1)']
styles = ['-v', '-^', '-o', '-+']
for j, method in enumerate(dlra_solvers):
    plt.semilogy(sizes, errors[:, j], styles[j], label=method_names[j])
plt.semilogy(sizes, approx_errors, '--', label=f'Approx. error ($r={rank}$)')
# plt.semilogy(sizes, 1e-10*cond_numbers, '-k', label=r'Cond. number of A ($\times 10^{-10}$)')
plt.tight_layout()
# x-axis in log scale power of 2
plt.xticks(sizes, sizes)
plt.legend()
plt.grid()
plt.ylim([1e-18, 1e0])
plt.xlabel("Size (mesh refinement)")
plt.ylabel("Relative error in Frobenius norm")
plt.show()

# # Plot the performance
# fig2 = plt.figure()
# for j, method in enumerate(dlra_solvers):
#     plt.semilogy(sizes, times[:, j], styles[j], label=method_names[j])
# plt.tight_layout()
# plt.legend()
# plt.xlabel("Size (mesh refinement)")
# plt.xticks(sizes, sizes)
# plt.ylabel("Time of computation (s)")
# plt.grid()
# plt.show()

# %% [markdown]
# ## Robust to stiffness - 2nd example

# %% [markdown]
# ### Setup the problem

# %%
# Imports
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spala
import time
import matplotlib.pyplot as plt
from low_rank_toolbox import LowRankMatrix, SVD
from problems import make_lyapunov_heat_square_with_time_dependent_source
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra

# Problem parameters
size = 128
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_with_time_dependent_source
ode, X0 = make_ode(size)

# Print the ode
print(ode)

# %% [markdown]
# ### Setup the solvers

# %%
# The sizes
sizes = [32, 64, 96, 128, 160, 192]

# Other parameters
nb_steps = 1000
ts = np.linspace(t_span[0], t_span[1], nb_steps+1)
rank = 8

# Define the methods (low-rank splitting comparison)
dlra_solvers = ['PERK', 'PERK', 'low_rank_splitting', 'low_rank_splitting']
krylov_kwargs = {'size': 1, 'kind': 'extended'}
methods_kwargs = [{'order': 1, 'krylov_kwargs': krylov_kwargs},
                  {'order': 2, 'krylov_kwargs': krylov_kwargs, 'strict_order_conditions': True},
                  {'order': 1}, 
                  {'order': 2}]

## PREALLOCATE THE ERRORS
approx_errors = np.zeros(len(sizes))
cond_numbers = np.zeros(len(sizes))
errors = np.zeros((len(sizes), len(dlra_solvers)))
times = np.zeros((len(sizes), len(dlra_solvers)))

# %% [markdown]
# ### Run the solvers

# %%
## LOOP OVER THE SIZES
for i, n in enumerate(sizes):
    ## MAKE THE ODE
    print('*********************************************************************************')
    print("Solving for n = {}".format(n))
    ode, X0 = make_ode(n)
    X0 = SVD.reduced_svd(X0)
    Y0 = SVD.truncated_svd(X0, rank)
    cond_numbers[i] = la.cond(ode.A.todense())

    ## COMPUTE REFERENCE SOLUTION
    X1 = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=ts, monitor=True, dense_output=True).X1
    approx_errors[i] = np.linalg.norm(X1 - SVD.truncated_svd(X1, rank).todense(), 'fro') / np.linalg.norm(X1, 'fro')

    ## LOOP OVER THE METHODS
    for j, method in enumerate(dlra_solvers):
        # COMPUTE THE SOLUTION WITH THE CURRENT METHOD
        try:
            t0 = time.time()
            Y1 = solve_dlra(ode, t_span, Y0, dlra_solver=method, t_eval=ts, monitor=True, solver_kwargs=methods_kwargs[j]).X1
            times[i, j] = time.time() - t0

            # COMPUTE THE RELATIVE ERROR
            errors[i, j] = np.linalg.norm(Y1.todense() - X1, 'fro') / np.linalg.norm(X1, 'fro')
        except:
            # Some methods may not work for stiff problems
            errors[i, j] = 1e10

# %% [markdown]
# ### Plot the results

# %%
# Plot the errors
fig1 = plt.figure()
method_names = ['Proj. exponential Euler', 'Proj. exponential Runge', 'Low-rank splitting (Lie-Trotter)', 'Low-rank splitting (Strang)']
styles = ['-o', '-s', '-+', '-x']
for j, method in enumerate(dlra_solvers):
    plt.semilogy(sizes, errors[:, j], styles[j], label=method_names[j])
plt.semilogy(sizes, approx_errors, '--', label=f'Approx. error ($r={rank}$)')
# plt.semilogy(sizes, 1e-10*cond_numbers, '-k', label=r'Cond. number of A ($\times 10^{-10}$)')
plt.tight_layout()
# x-axis in log scale power of 2
plt.xticks(sizes, sizes)
plt.legend()
plt.grid()
plt.ylim([1e-10, 1e0])
plt.xlabel("Size (mesh refinement)")
plt.ylabel("Relative error in Frobenius norm")
plt.show()

# Plot the performance
fig2 = plt.figure()
for j, method in enumerate(dlra_solvers):
    plt.semilogy(sizes, times[:, j], styles[j], label=method_names[j])
plt.tight_layout()
plt.legend()
plt.xlabel("Size (mesh refinement)")
plt.xticks(sizes, sizes)
plt.ylabel("Time of computation (s)")
plt.grid()
plt.show()

# %% [markdown]
# ## Compare the methods - Global error and performance

# %% [markdown]
# ### Setup the problem

# %%
# Imports
import numpy as np
import scipy.sparse.linalg as spala
import time
import matplotlib.pyplot as plt
from low_rank_toolbox import LowRankMatrix, SVD
from problems import make_lyapunov_heat_square_with_time_dependent_source
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra

# Problem parameters
size = 128
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_with_time_dependent_source
ode, X0 = make_ode(size)

# Print the ode
print(ode)

# Preprocess the problem
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)
invA = spala.splu(ode.A).solve
invB = spala.splu(ode.B).solve

# %% [markdown]
# ### Setup the solvers

# %%
# DLRA parameters
rank = 8
Y0 = X0.truncate(rank)

# Methods parameters - ALL PERK
nb_steps = np.logspace(1, 4, 8, dtype=int)
stepsizes = t_span[1] / nb_steps
dlra_solvers = ['PERK', 'PERK', 'PERK']
krylov_kwargs = {'size': 1,
                 'kind': 'extended',
                 'is_symmetric': True,
                 'invA': invA,
                 'invB': invB}
methods_kwargs = [{'order': 1, 'krylov_kwargs': krylov_kwargs},
                  {'order': 2, 'krylov_kwargs': krylov_kwargs, 'strict_order_conditions': True},
                  {'order': 2, 'krylov_kwargs': krylov_kwargs, 'strict_order_conditions': False}]

## Pre-allocate some variables
global_errors = np.zeros((len(nb_steps), len(dlra_solvers)))
times = np.zeros((len(nb_steps), len(dlra_solvers)))

# %% [markdown]
# ### Run the solvers

# %%
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
approx_error = np.linalg.norm(X1 - SVD.truncated_svd(X1, rank).todense(), 'fro') / np.linalg.norm(X1, 'fro')

# %% [markdown]
# ### Plot the comparison between all three methods

# %%
# Plot the errors
fig1 = plt.figure()
method_names = ['Proj. exponential Euler',  'Proj. exponential Runge (strict)', 'Proj. exponential Runge (non strict)']
styles = ['-o', '-x', '-p']
plt.loglog(stepsizes, global_errors[:, 0], styles[0], label=method_names[0])
plt.loglog(stepsizes, global_errors[:, 1], styles[1], label=method_names[1])
plt.loglog(stepsizes, global_errors[:, 2], styles[2], label=method_names[2])
plt.loglog(stepsizes, 4*stepsizes, 'k') # , label=r'$O(h)$'
plt.loglog(stepsizes, 4*stepsizes**2, 'k') # , label=r'$O(h^2)$'
plt.axhline(approx_error, linestyle='--', color='gray', label=f'Approx. error ($r={rank}$)')
plt.legend(loc='upper left')
plt.grid()
plt.xlabel("Step size")
plt.ylabel("Relative error in Frobenius norm")
plt.ylim([1e-7, 1e3])
plt.tight_layout()
plt.show()

# Plot the performance
fig2 = plt.figure()
plt.loglog(global_errors[:, 0], times[:, 0], styles[0], label=method_names[0])
plt.loglog(global_errors[:, 1], times[:, 1], styles[1], label=method_names[1])
plt.loglog(global_errors[:, 2], times[:, 2], styles[2], label=method_names[2])
plt.axvline(approx_error, linestyle='--', color='gray', label=f'Approx.error ($r={rank}$)')
plt.legend(loc='upper left')
plt.gca().invert_xaxis()
plt.grid()
plt.xlabel("Relative error in Frobenius norm")
plt.ylabel("Time of computation (s)")
plt.tight_layout()
plt.show()

# # Print the slopes of the errors
# for j, method in enumerate(dlra_solvers):
#     print(f'{method_names[j]}: {np.polyfit(np.log(stepsizes), np.log(global_errors[:, j]), 1)[0]}')


# %% [markdown]
# ## Compare the ranks - Global error and performance

# %% [markdown]
# ### Setup the problem

# %%
# Imports
import numpy as np
import scipy.sparse.linalg as spala
import time
import matplotlib.pyplot as plt
from low_rank_toolbox import LowRankMatrix, SVD
from problems import make_lyapunov_heat_square_with_time_dependent_source
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_dlra

# Problem parameters
size = 128
t_span = (0, 1)
make_ode = make_lyapunov_heat_square_with_time_dependent_source
ode, X0 = make_ode(size)

# Print the ode
print(ode)

# Preprocess the problem
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)
invA = spala.splu(ode.A).solve
invB = spala.splu(ode.B).solve

# %% [markdown]
# ### Setup the solver and other parameters

# %%
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

# Ranks to test
ranks = [4, 7, 11, 16]

# Number of steps
list_nb_steps = np.logspace(1, 5, 10, dtype=int)

# Preallocation
global_errors = np.zeros((len(list_nb_steps), len(ranks)))
approx_error = np.zeros((len(list_nb_steps), len(ranks)))
times = np.zeros((len(list_nb_steps), len(ranks)))

# %% [markdown]
# ### Run the solver for different ranks

# %%
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


# %% [markdown]
# ### Plot the results

# %%
# Plot the errors
fig1 = plt.figure()
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
plt.grid()
plt.tight_layout()
plt.show()


# Plot the performance
fig2 = plt.figure()
for j, rank in enumerate(ranks):
    plt.loglog(global_errors[:, j], times[:, j], styles[j], label=names[j], color=colors[j])
    plt.axvline(approx_error[0, j], linestyle='--', color=colors[j])
plt.legend()
plt.xlabel("Relative error in Frobenius norm")
plt.ylabel("Time of computation (s)")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.grid()
plt.show()

# %% [markdown]
# ## Rank-adaptive method

# %% [markdown]
# ### Setup the problem

# %%
# Imports
import numpy as np
import scipy.sparse.linalg as spala
import time
import matplotlib.pyplot as plt
from low_rank_toolbox import LowRankMatrix, SVD
from problems import make_lyapunov_heat_square_with_time_dependent_special
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.dlra import solve_adaptive_dlra

# Problem parameters
size = 128 # 128
t_span = (0, 1) # (0, 1)
make_ode = make_lyapunov_heat_square_with_time_dependent_special
ode, X0 = make_ode(size)

# Print the ode
print(ode)
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)
print('Dimensions: ', X0.shape)

# Precompute reference solution
nb_steps = 1000 # 1000
ts = np.linspace(*t_span, nb_steps+1)
ref_sol = solve_matrix_ivp(ode, t_span, X0, solver="automatic", t_eval=ts, monitor=True)
Xs_ref = ref_sol.todense()

# %% [markdown]
# ### Plot the singular values

# %%
# Selected time steps
time_steps = [0, 30, 50, 70, 100]
labels = [f'Singular values at t = {round(ts[index], 3)}' for index in time_steps]
sing_vals = np.zeros(len(time_steps), dtype=object)

# Compute the singular values
for i, index in enumerate(time_steps):
    if isinstance(Xs_ref[index], SVD):
        Xs_ref[index] = Xs_ref[index].todense()
    sing_vals[i] = np.linalg.svd(Xs_ref[index], compute_uv=False)

# Plot the singular values at each time step
for i, _ in enumerate(time_steps):
    new_fig = plt.figure()
    indexes = np.arange(1, len(sing_vals[0])+1)
    plt.semilogy(indexes, sing_vals[i], 'o', label=labels[i])
    # Machine precision
    plt.axhline(np.finfo(float).eps, linestyle='--', color='gray', label='Machine precision')
    plt.legend()
    plt.grid()
    plt.xlabel("Index")
    plt.xticks(np.linspace(1, len(sing_vals[0]), 9, dtype=int))
    # plt.ylabel("Singular value")
    plt.tight_layout()
    plt.ylim([1e-20, 1e1])
    plt.show()


# %% [markdown]
# ### Setup the solver and other parameters

# %%
# Define the methods
from scipy.sparse import linalg as spala
invA = spala.inv(ode.A).dot
dlra_solver = 'adaptive_PERK'
krylov_kwargs = {'size': 1, 'kind': 'extended', 'invA': invA, 'invB': invA}
method_kwargs = {'order': 2, 'krylov_kwargs': krylov_kwargs, 'strict_order_conditions': True}

# Define the tolerances
tolerances = [1e-5, 1e-8, 1e-11]

# Initialize the error over time matrix
nb_t_steps = len(ref_sol.Xs)
dlra_solutions = np.zeros(len(tolerances), dtype=object)

# %% [markdown]
# ### Run the solver for different tolerances

# %%
# Loop over the tolerances
for j, tol in enumerate(tolerances):
    # Compute the solution with the current method
    Y0 = X0.truncate(rtol = tol)
    dlra_solutions[j] = solve_adaptive_dlra(ode, t_span, Y0, adaptive_dlra_solver=dlra_solver, solver_kwargs=method_kwargs, monitor=True, t_eval=ts, rtol=tol, atol=None)

# Compute the error over time
errors = np.zeros((nb_t_steps, len(tolerances)))
for i in range(nb_t_steps):
    for j, tol in enumerate(tolerances):
        errors[i, j] = np.linalg.norm(Xs_ref[i] - dlra_solutions[j].Xs[i].todense(), 'fro') / np.linalg.norm(Xs_ref[i], 'fro')

# Extract the rank over time
ranks = np.zeros((nb_t_steps, len(tolerances)))
for i in np.arange(0, nb_t_steps):
    for j, tol in enumerate(tolerances):
        ranks[i, j] = dlra_solutions[j].Xs[i].rank

# %% [markdown]
# ### Plot the results

# %%
# Plot the error over time
fig1 = plt.figure()
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, tol in enumerate(tolerances):
    plt.semilogy(ts, errors[:, i], label=f'Tolerance {tol}', color=colors[i])
    plt.axhline(tol, linestyle='--', color=colors[i])
plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Relative error in Frobenius norm")
plt.ylim(1e-12, 1e-1)
plt.grid()
plt.tight_layout()
plt.show()

# Plot the rank over time
fig2 = plt.figure()
for i, tol in enumerate(tolerances):
    plt.plot(ts, ranks[:, i], label=f'Tolerance {tol}', color=colors[i])
plt.legend(loc = 'upper left')
plt.ylim(0, 50)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Rank")
plt.tight_layout()
plt.show()

# %% [markdown]
# 


