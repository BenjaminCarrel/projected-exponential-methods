"""
Second order Krylov approximation applied to the Riccati equation

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Importations
import numpy as np
from graphics_parameters import *
from problems import make_riccati_ostermann
from low_rank_toolbox import SVD, LowRankMatrix
import scipy.sparse.linalg as spala
from krylov_toolbox import KrylovSpace, ExtendedKrylovSpace, RationalKrylovSpace
from scipy import linalg as la
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox import SylvesterOde
import time

#%% SETUP THE ODE
print('Generating problem...')
size = 200
q = 9
t_span = (0, 0.1)
make_ode = lambda size: make_riccati_ostermann(size, q)
ode, X0 = make_ode(size)
print('Done!')

# Print the ode defined in setup_problem.py
print(ode)
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)

# Preprocess the problem
invA = spala.splu(ode.A).solve
invB = spala.splu(ode.B).solve


#%% Krylov parameters
t_span = (0, 0.01) # you can change final time here
h = t_span[1] - t_span[0]
rank = 1 # rank of initial value
nb_krylov_iter = 20 # number of Krylov iterations (extended Krylov space will have size 2*nb_krylov_iter)

#%% KRYLOV APPROXIMATION ERROR - FULL PROBLEM FOR REFERENCE
# Data for order 1
A, Ad = ode.A, ode.A.todense()
B, Bd = ode.B, ode.B.todense()
Y0 = SVD.truncated_svd(X0, rank)
PGY0 = Y0.project_onto_tangent_space(ode.non_linear_field(0, Y0))

# Solve the full problem (order 1)
full_ode = SylvesterOde(A, B, PGY0)
K1 = solve_matrix_ivp(full_ode, t_span, Y0, solver='closed_form', dense_output=True)

# Data for order 2
RK1 = SVD.truncated_svd(K1, rank)
GRK1 = ode.non_stiff_field(h, RK1)
PGRK1 = RK1.project_onto_tangent_space(GRK1)
PGRK1_minus_PGY0 = PGRK1 - PGY0

# Solve the full problem (order 2)
D = la.solve_sylvester(Ad, Bd, PGRK1_minus_PGY0.todense())
D_hat = la.solve_sylvester(Ad*h, Bd*h, D)
C = la.solve_sylvester(Ad, Bd, PGY0.todense())
Z = Y0 + D_hat + C
Z = spala.expm_multiply(A, Z, start=0, stop=h, num=2, endpoint=True)[-1]
Z = spala.expm_multiply(B.T.conj(), Z.T.conj(), start=0, stop=h, num=2, endpoint=True)[-1].T.conj()
Y1_full = Z - C - D_hat - D

#%% Define a solver for the reduced problem
def closed_form_solver(h, A, B, Y0, PGY0, PGRK1_minus_PGY0):
    D = la.solve_sylvester(A, B, PGRK1_minus_PGY0.todense())
    D_hat = la.solve_sylvester(A*h, B*h, D)
    C = la.solve_sylvester(A, B, PGY0.todense())
    Z = Y0 + D_hat + C
    Z = spala.expm_multiply(A, Z, start=0, stop=h, num=2, endpoint=True)[-1]
    Z = spala.expm_multiply(B.T.conj(), Z.T.conj(), start=0, stop=h, num=2, endpoint=True)[-1].T.conj()
    Y1_full = Z - C - D_hat - D
    return Y1_full

#%% KRYLOV APPROXIMATION ERROR - POLYNOMIAL KRYLOV SPACE
print('------------ POLYNOMIAL KRYLOV SPACE ------------')
# Pre-allocate some variables
krylov_error = np.zeros(nb_krylov_iter)
krylov_space_size = np.zeros(nb_krylov_iter, dtype=int)

# Initialization of the two spaces
U = la.orth(np.column_stack([Y0.U, PGY0.U, PGRK1_minus_PGY0.U]))
left_krylov_space = KrylovSpace(A, U)
V = la.orth(np.column_stack([Y0.V, PGY0.V, PGRK1_minus_PGY0.V]))
right_krylov_space = KrylovSpace(B, V)

# Extract the two basis
Vk = left_krylov_space.Q
Wk = right_krylov_space.Q

# Reduced data
A_reduced = Vk.T.dot(A.dot(Vk))
B_reduced = Wk.T.dot(B.dot(Wk))
Y0_reduced = Y0.dot(Wk).dot(Vk.T, side='left')
PGY0_reduced = PGY0.dot(Wk).dot(Vk.T, side='left')
PGRK1_minus_PGY0_reduced = PGRK1_minus_PGY0.dot(Wk).dot(Vk.T, side='left')

# Solve the reduced problem
Zk = closed_form_solver(h, A_reduced, B_reduced, Y0_reduced, PGY0_reduced, PGRK1_minus_PGY0_reduced)
Y1_reduced = Vk.dot(Zk.dot(Wk.T))

# Compute the error and store the current size
krylov_error[0] = la.norm(Y1_full - Y1_reduced) / la.norm(Y1_full)
krylov_space_size[0] = left_krylov_space.Q.shape[1]
print("Iteration: k=1, size of space: {}, current error: {}".format(krylov_space_size[0], krylov_error[0]))

# Loop over the iterations
for i in np.arange(1, nb_krylov_iter):
    # Augment the two basis
    left_krylov_space.augment_basis()
    right_krylov_space.augment_basis()

    # Define the reduced problem ode
    Vk = left_krylov_space.Q
    Wk = right_krylov_space.Q
    
    # Reduced data
    A_reduced = Vk.T.dot(A.dot(Vk))
    B_reduced = Wk.T.dot(B.dot(Wk))
    Y0_reduced = Y0.dot(Wk).dot(Vk.T, side='left')
    PGY0_reduced = PGY0.dot(Wk).dot(Vk.T, side='left')
    PGRK1_minus_PGY0_reduced = PGRK1_minus_PGY0.dot(Wk).dot(Vk.T, side='left')

    # Solve the reduced problem
    Zk = closed_form_solver(h, A_reduced, B_reduced, Y0_reduced, PGY0_reduced, PGRK1_minus_PGY0_reduced)
    Y1_reduced = Vk.dot(Zk.dot(Wk.T))


    # Compute the error and store the current size
    krylov_error[i] = la.norm(Y1_full - Y1_reduced)
    krylov_space_size[i] = left_krylov_space.Q.shape[1]
    print("Iteration: k={}, size of space: {}, current error: {}".format(i+1, krylov_space_size[i], krylov_error[i]))

print("Done!")

#%% KRYLOV APPROXIMATION ERROR - EXTENDED KRYLOV SPACE
print('------------ EXTENDED KRYLOV SPACE ------------')
# Pre-allocate some variables
nb_extended_krylov_iter = int(nb_krylov_iter/2)
extended_krylov_error = np.zeros(nb_extended_krylov_iter)
extended_krylov_space_size = np.zeros(nb_extended_krylov_iter, dtype=int)

# Preprocess the inverses
invA = spala.splu(ode.A).solve
invB = spala.splu(ode.B).solve

# Initialization of the two spaces
U = la.orth(np.column_stack([Y0.U, PGY0.U, PGRK1_minus_PGY0.U]))
left_extended_krylov_space = ExtendedKrylovSpace(A, U, invA=invA)
V = la.orth(np.column_stack([Y0.V, PGY0.V, PGRK1_minus_PGY0.V]))
right_extended_krylov_space = ExtendedKrylovSpace(B, V, invA=invB)

# Extract the two basis
Vk = left_extended_krylov_space.Q
Wk = right_extended_krylov_space.Q

# Reduced data
A_reduced = Vk.T.dot(A.dot(Vk))
B_reduced = Wk.T.dot(B.dot(Wk))
Y0_reduced = Y0.dot(Wk).dot(Vk.T, side='left')
PGY0_reduced = PGY0.dot(Wk).dot(Vk.T, side='left')
PGRK1_minus_PGY0_reduced = PGRK1_minus_PGY0.dot(Wk).dot(Vk.T, side='left')

# Solve the reduced problem
Zk = closed_form_solver(h, A_reduced, B_reduced, Y0_reduced, PGY0_reduced, PGRK1_minus_PGY0_reduced)
Y1_reduced = Vk.dot(Zk.dot(Wk.T))

# Compute the error and store the current size
extended_krylov_error[0] = la.norm(Y1_full - Y1_reduced)
extended_krylov_space_size[0] = left_extended_krylov_space.Q.shape[1]
print("Iteration: k=1, size of space: {}, current error: {}".format(extended_krylov_space_size[0], extended_krylov_error[0]))

# Loop over the iterations
for i in np.arange(1, nb_extended_krylov_iter):
    # Augment the two basis
    left_extended_krylov_space.augment_basis()
    right_extended_krylov_space.augment_basis()

    # Reduced data
    Vk = left_extended_krylov_space.Q
    Wk = right_extended_krylov_space.Q
    A_reduced = Vk.T.dot(A.dot(Vk))
    B_reduced = Wk.T.dot(B.dot(Wk))
    Y0_reduced = Y0.dot(Wk).dot(Vk.T, side='left')
    PGY0_reduced = PGY0.dot(Wk).dot(Vk.T, side='left')
    PGRK1_minus_PGY0_reduced = PGRK1_minus_PGY0.dot(Wk).dot(Vk.T, side='left')

    # Solve the reduced problem
    Zk = closed_form_solver(h, A_reduced, B_reduced, Y0_reduced, PGY0_reduced, PGRK1_minus_PGY0_reduced)
    Y1_reduced = Vk.dot(Zk.dot(Wk.T))

    # Compute the error and store the current size
    extended_krylov_error[i] = la.norm(Y1_full - Y1_reduced)
    extended_krylov_space_size[i] = left_extended_krylov_space.Q.shape[1]
    print("Iteration: k={}, size of space: {}, current error: {}".format(i+1, extended_krylov_space_size[i], extended_krylov_error[i]))

print("Done!")

# %% KRYLOV APPROXIMATION ERROR - RATIONAL KRYLOV SPACE
print('------------ RATIONAL KRYLOV SPACE ------------')
# Pre-allocate some variables
rational_krylov_error = np.zeros(nb_krylov_iter)
rational_krylov_space_size = np.zeros(nb_krylov_iter, dtype=int)

# One repeated single pole: k/sqrt(2)
poles = [nb_krylov_iter/np.sqrt(2) for _ in range(nb_krylov_iter)]

# Initialization of the two spaces
U = la.orth(np.column_stack([Y0.U, PGY0.U, PGRK1_minus_PGY0.U]))
left_rational_krylov_space = RationalKrylovSpace(A, U, poles=poles)
V = la.orth(np.column_stack([Y0.V, PGY0.V, PGRK1_minus_PGY0.V]))
right_rational_krylov_space = RationalKrylovSpace(B, V, poles=poles)

# Extract the two basis
Vk = left_rational_krylov_space.Q
Wk = right_rational_krylov_space.Q

# Reduced data
A_reduced = Vk.T.dot(A.dot(Vk))
B_reduced = Wk.T.dot(B.dot(Wk))
Y0_reduced = Y0.dot(Wk).dot(Vk.T, side='left')
PGY0_reduced = PGY0.dot(Wk).dot(Vk.T, side='left')
PGRK1_minus_PGY0_reduced = PGRK1_minus_PGY0.dot(Wk).dot(Vk.T, side='left')

# Solve the reduced problem
Zk = closed_form_solver(h, A_reduced, B_reduced, Y0_reduced, PGY0_reduced, PGRK1_minus_PGY0_reduced)
Y1_reduced = Vk.dot(Zk.dot(Wk.T))

# Compute the error and store the current size
rational_krylov_error[0] = la.norm(Y1_full - Y1_reduced)
rational_krylov_space_size[0] = left_rational_krylov_space.Q.shape[1]
print("Iteration: k=1, size of space: {}, current error: {}".format(rational_krylov_space_size[0], rational_krylov_error[0]))

# Loop over the iterations
for i in np.arange(1, nb_krylov_iter):
    # Augment the two basis
    left_rational_krylov_space.augment_basis()
    right_rational_krylov_space.augment_basis()

    # Reduced data
    Vk = left_rational_krylov_space.Q
    Wk = right_rational_krylov_space.Q
    A_reduced = Vk.T.dot(A.dot(Vk))
    B_reduced = Wk.T.dot(B.dot(Wk))
    Y0_reduced = Y0.dot(Wk).dot(Vk.T, side='left')
    PGY0_reduced = PGY0.dot(Wk).dot(Vk.T, side='left')
    PGRK1_minus_PGY0_reduced = PGRK1_minus_PGY0.dot(Wk).dot(Vk.T, side='left')

    # Solve the reduced problem
    Zk = closed_form_solver(h, A_reduced, B_reduced, Y0_reduced, PGY0_reduced, PGRK1_minus_PGY0_reduced)
    Y1_reduced = Vk.dot(Zk.dot(Wk.T))

    # Compute the error and store the current size
    rational_krylov_error[i] = la.norm(Y1_full - Y1_reduced)
    rational_krylov_space_size[i] = left_rational_krylov_space.Q.shape[1]
    print("Iteration: k={}, size of space: {}, current error: {}".format(i+1, rational_krylov_space_size[i], rational_krylov_error[i]))

print("Done!")

# %% KRYLOV APPROXIMATION ERROR - THEORETICAL BOUND
# Bound derived in the paper
def bound(k, t, mu, norm_Y0, norm_PGY0, norm_D, factor):
    # Bound
    bd = 4 * np.sqrt(2) / (factor**(k-1)) * (np.exp(t * mu) * norm_Y0 + (np.exp(t * mu) - 1)/mu * norm_PGY0 + (np.exp(t * mu) - 1 - t * mu)/(t * mu**2) * norm_D)
    return bd

# Eigenvalues of A and B
eigA = la.eig(A.todense(), right=False)
eigB = la.eig(B.todense(), right=False)
# Mus
muA = np.max(eigA)
muB = np.max(eigB)
mu = np.max([muA, muB])
# Norms
norm_Y0 = np.linalg.norm(Y0.todense(), 'fro')
norm_PGY0 = np.linalg.norm(PGY0.todense(), 'fro')
norm_D = np.linalg.norm(D, 'fro')

bound1 = bound(np.arange(nb_krylov_iter)+1, h, mu, norm_Y0, norm_PGY0, norm_D, factor=3)
bound2 = bound(np.arange(nb_krylov_iter)+1, h, mu, norm_Y0, norm_PGY0, norm_D, factor=9.037)

# %% KRYLOV APPROXIMATION ERROR - PLOT
# Plot the results
fig = plt.figure()
plt.semilogy(krylov_space_size, krylov_error, 'o-', label="Polynomial Krylov")
plt.semilogy(extended_krylov_space_size, extended_krylov_error, 'o-', label="Extended Krylov")
plt.semilogy(rational_krylov_space_size, rational_krylov_error, 'o-', label="Rational Krylov (repeated poles)")
# plt.semilogy(rational_krylov_space_size, rational_krylov_error_opti, 'o-', label="Rational Krylov (optimal poles)")
plt.semilogy(rational_krylov_space_size, bound1, '--', label="Bound with factor 3 (Theorem 5.19)")
plt.semilogy(rational_krylov_space_size, bound2, '--', label="Asymptotic bound with factor 9.037")
plt.axhline(1e-12, color='k', linestyle='-', label="Tolerance")
plt.xlabel("Size of the approximation space")
plt.xticks(rational_krylov_space_size)
plt.ylabel("Relative error in Frobenius norm")
plt.ylim([1e-13, 1e5])
plt.legend(loc='upper right')
plt.show()

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
fig.savefig(f'figures/{X0.shape}_krylov_error_two_T_{t_span[1]}_rank_{rank}_nb_iter_{nb_krylov_iter}_{timestamp}.pdf', bbox_inches='tight')


# %%
