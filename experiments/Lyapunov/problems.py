"""
File for toy problems with a Lyapunov structure.

Author: Benjamin Carrel, University of Geneva, 2022
"""

# %% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
from low_rank_toolbox import SVD
from matrix_ode_toolbox import LyapunovOde, SylvesterLikeOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp

#%% Heat equations on a square [0,1]x[0,1]
## Dirichlet BC
def make_lyapunov_heat_square_dirichlet(size):
    """
    Generate a Lyapunov ODE that models a 2D heat propagation on the square [0,1]x[0,1] with Dirichlet BC.

    The data is randomly generated with a prescribed seed, so it is consistent with itself.

    This piece of code can be used as a template for generating similar problems.

    Parameters
    ----------
    size: int
        The size of the ODE 

    Returns
    -------
    ode: LyapunovOde
        The Lyapunov ode structure with the data generated
    X0: Matrix
        The initial value that can be used out of the box
    """
    x_space = np.linspace(0, 1, num=size)
    dx = x_space[1] - x_space[0]

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, dirichlet BC
    A = (1/dx)**2 * sps.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format="csc")

    ## SOURCE: C is a random low-rank symmetric matrix with exponential decay
    np.random.seed(2222)
    sing_vals_C = np.logspace(0, -16, num=5)
    M = np.random.rand(size, 5)
    Q, _ = la.qr(M, mode='reduced')
    C = SVD(Q, sing_vals_C, Q)

    ## DEFINE THE ODE
    ode = LyapunovOde(A, C)

    ## INITIAL VALUE: X_0 is a random low-rank symmetric matrix with exponential decay
    sing_vals_iv = np.logspace(0, -20, num=21)
    M = np.random.rand(size, 21)
    Q, _ = la.qr(M, mode='reduced')
    iv = SVD(Q, sing_vals_iv, Q)
    X0 = solve_matrix_ivp(ode, (0, 0.01), iv, solver='closed_form').todense() # skip numerical instabilities due to the randomness

    return ode, X0

def make_lyapunov_heat_square_with_time_dependent_source(size: int = 128, q: int = 5):
    "A heat problem with time dependent source. Example for showing convergence."
    ## DISCRETIZATION
    xs = np.linspace(0, 1, size)
    ys = xs

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, dirichlet BC
    dx = xs[1] - xs[0]
    A = (1/dx)**2 * sps.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format="csc")

    ## SOURCE
    # Construction of C
    nb = int((q - 1) / 2)
    ones = np.ones((1, size))
    e = np.zeros((nb, size))
    f = np.zeros((nb, size))
    for k in np.arange(nb):
        e[k] = np.sqrt(2) * np.cos(2 * np.pi * k * xs)
        f[k] = np.sqrt(2) * np.sin(2 * np.pi * k * xs)
    c = np.concatenate((ones, e, f))
    C = SVD.truncated_svd(c.T.dot(c))
    def G(t, X):
        return (C * np.exp(4*t)).todense()
    
    ## DEFINE THE ODE
    ode = SylvesterLikeOde(A, A, G)

    ## INITIAL VALUE: X_0 is a random low-rank matrix with exponential decay
    sing_vals_iv = np.logspace(-1, -20, num=20)
    X0 = SVD.generate_random((size, size), sing_vals_iv, seed=2222, is_symmetric=False)

    ## REALISTIC INITIAL VALUE: SOLVE THE ODE FOR A SHORT TIME
    X0 = solve_matrix_ivp(ode, (0, 0.01), X0, solver="scipy")

    return ode, X0


def make_lyapunov_heat_square_with_time_dependent_adaptive(size: int = 128, q: int = 5):
    "A heat problem with time dependent source. Example for rank adaptive methods."
    ## DISCRETIZATION
    xs = np.linspace(0, 1, size)
    ys = xs

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, dirichlet BC
    dx = xs[1] - xs[0]
    A = (1/dx)**2 * sps.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format="csc")

    # Construction of C
    nb = int((q - 1) / 2)
    ones = np.ones((1, size))
    e = np.zeros((nb, size))
    f = np.zeros((nb, size))
    for k in np.arange(nb):
        e[k] = np.sqrt(2) * np.cos(2 * np.pi * k * xs)
        f[k] = np.sqrt(2) * np.sin(2 * np.pi * k * xs)
    c = np.concatenate((ones, e, f))
    C = SVD.truncated_svd(c.T.dot(c))

    np.random.seed(2222)
    sing_vals_C = np.logspace(0, -15, num=8)
    M = np.random.rand(size, 8)
    Q, _ = np.linalg.qr(M, mode='reduced')
    D = SVD(Q, sing_vals_C, Q)
    def G(t, X):
        sing_vals = np.logspace(0, -14, num=8)
        X1 = SVD(Q, sing_vals, Q)
        sing_vals_mid = np.logspace(0, -7, num=8)
        X2 = SVD(Q, sing_vals_mid, Q)
        # between 0 and 0.2: C = A X1 + X1 A
        if t < 0.2:
            return X1.dot(A, side='left') + X1.dot(A)
        # between 0.2 and 0.4: continuous path between X1 and X2
        elif t < 0.4:
            return X1.dot(A, side='left') + X1.dot(A) + (t - 0.2) / 0.2 * (X2.dot(A, side='left') + X2.dot(A) - X1.dot(A, side='left') - X1.dot(A)) 
        # between 0.4 and 0.6: C = A X2 + X2 A
        elif t < 0.6:
            return X2.dot(A, side='left') + X2.dot(A)
        # between 0.6 and 0.8: continuous path between X2 and X1
        elif t < 0.8:
            return X2.dot(A, side='left') + X2.dot(A) + (t - 0.6) / 0.2 * (X1.dot(A, side='left') + X1.dot(A) - X2.dot(A, side='left') - X2.dot(A)) 
        # between 0.8 and 1: C = A X1 + X1 A
        else:
            return X1.dot(A, side='left') + X1.dot(A)
    
    ## DEFINE THE ODE
    ode = SylvesterLikeOde(A, A, G)

    ## INITIAL VALUE: X_0 is a random low-rank matrix with exponential decay
    sing_vals_iv = np.logspace(-1, -20, num=20)
    X0 = SVD.generate_random((size, size), sing_vals_iv, seed=2222, is_symmetric=False)

    ## REALISTIC INITIAL VALUE: SOLVE THE ODE FOR A SHORT TIME
    X0 = solve_matrix_ivp(ode, (0, 0.01), X0, solver="scipy")

    return ode, X0