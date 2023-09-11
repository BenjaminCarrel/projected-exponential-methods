"""
File for problems.
Those problems are used in the experiments.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% Imports
import numpy as np
import scipy.sparse as sps
import numpy.linalg as la
from low_rank_toolbox import SVD
from matrix_ode_toolbox import SylvesterLikeOde, RiccatiOde, LyapunovOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp


#%% Lyapunov ODEs

## Lyapunov ODE with Dirichlet BC and constant source
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


## Lyapunov ODE with Dirichlet BC and time dependent source
def make_lyapunov_heat_square_with_time_dependent_source(size: int = 128, q: int = 5):
    """
    A heat problem with time dependent source.
    Problem inspired from the paper:
    "EXPLICIT EXPONENTIAL RUNGE-KUTTA METHODS FOR SEMILINEAR PARABOLIC PROBLEMS"
    by MARLIS HOCHBRUCK AND ALEXANDER OSTERMANN

    Exponential Runge shows an order reduction on this problem (in 1D).
    """
    ## DISCRETIZATION
    xs = np.linspace(0, 1, size)
    ys = xs

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, dirichlet BC
    dx = xs[1] - xs[0]
    A = (1/dx)**2 * sps.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format="csc")

    ## SOURCE: G(t) = 4 e^t + [x(1-x) + y(1-y)] e^t
    # Xs, Ys = np.meshgrid(xs, ys)
    # M = np.random.rand(size, 5)
    # Q, _ = np.linalg.qr(M, mode='reduced')
    # C = SVD(Q, np.logspace(-4, -20, num=5), Q)
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
    sing_vals_D = np.logspace(0, -15, num=8)
    D = SVD.generate_random((size, size), sing_vals_D, seed=2222, is_symmetric=True)
    def G(t, X):
        # return 4 * np.exp(t) + (Xs * (1 - Xs) + Ys * (1 - Ys)) * np.exp(t)
        # return 4 * np.exp(t) + (Xs * (1 - Xs)) * np.exp(t)
        # Source: linear combination of low-rank matrices
        # return C + X.dot(X)
        return C * np.exp(-4*t)
            # return (C * (np.exp(4*t) + np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t))).todense() + (D * (t**4 + t**3 + t**2 + t + 1)).todense()
    
    ## DEFINE THE ODE
    ode = SylvesterLikeOde(A, A, G)

    ## INITIAL VALUE: X_0 is a random low-rank matrix with exponential decay
    sing_vals_iv = np.logspace(-1, -20, num=20)
    X0 = SVD.generate_random((size, size), sing_vals_iv, seed=2222, is_symmetric=False)

    ## REALISTIC INITIAL VALUE: SOLVE THE ODE FOR A SHORT TIME
    X0 = solve_matrix_ivp(ode, (0, 0.01), X0, solver="scipy")

    return ode, X0


def make_lyapunov_heat_square_with_time_dependent_special(size: int = 128, q: int = 5):
    """
    A heat problem with time dependent source.
    Problem inspired from the paper:
    "EXPLICIT EXPONENTIAL RUNGE-KUTTA METHODS FOR SEMILINEAR PARABOLIC PROBLEMS"
    by MARLIS HOCHBRUCK AND ALEXANDER OSTERMANN

    Exponential Runge shows an order reduction on this problem (in 1D).
    """
    ## DISCRETIZATION
    xs = np.linspace(0, 1, size)
    ys = xs

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, dirichlet BC
    dx = xs[1] - xs[0]
    A = (1/dx)**2 * sps.diags([1, -2, 1], [-1, 0, 1], shape=(size, size), format="csc")

    ## SOURCE: G(t) = 4 e^t + [x(1-x) + y(1-y)] e^t
    # Xs, Ys = np.meshgrid(xs, ys)
    # M = np.random.rand(size, 5)
    # Q, _ = np.linalg.qr(M, mode='reduced')
    # C = SVD(Q, np.logspace(-4, -20, num=5), Q)
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

#%% Riccati ODE
def make_riccati_ostermann(m: int = 200, q: int = 9):
    """
    Make a Riccati problem as described in Ostermann (2019).

    See "Convergence of a Low-Rank Lie--Trotter Splitting for Stiff Matrix Differential Equations" by Ostermann, 2019.

    Parameters
    ----------
    m : int
        Size of the discretization.
    q : int
        Rank of the source term.

    Returns
    -------
    ode : RiccatiOde
        The Riccati equation in ODE structure.
    X0 : np.ndarray
        The initial value.
    """
    # Paper's parameters
    # m, q = 200, 9

    # Parameters
    nb = int((q - 1) / 2)
    lam = 1

    # Spatial discretization of [0, 1]^2
    x = np.zeros(m)
    x_minus = np.zeros(m)
    x_plus = np.zeros(m)
    for j in np.arange(m):
        x[j] = (j + 1) / (m + 1)
        x_minus[j] = (j + 1) / (m + 1) - 1 / (2 * m + 2)
        x_plus[j] = (j + 1) / (m + 1) + 1 / (2 * m + 2)

    # Construction of A : discrete operator of d_x(alpha(xs) d_x) - lam*I (finite volume method) 
    alpha_minus = 2 + np.cos(2 * np.pi * x_minus)
    alpha_plus = 2 + np.cos(2 * np.pi * x_plus)
    data = (m + 1) ** 2 * \
        np.array([alpha_plus, -alpha_minus - alpha_plus, alpha_minus])
    D = sps.spdiags(data, diags=[-1, 0, 1], m=m, n=m)
    Lam = lam * sps.eye(m)
    A = D - Lam
    # A = A.todense()

    # Construction of D
    D = sps.eye(m)
    #D = SVD.truncated_svd(d.dot(d.T))

    # Construction of C
    ones = np.ones((1, m))
    e = np.zeros((nb, m))
    f = np.zeros((nb, m))
    for k in np.arange(nb):
        e[k] = np.sqrt(2) * np.cos(2 * np.pi * k * x)
        f[k] = np.sqrt(2) * np.sin(2 * np.pi * k * x)
    c = np.concatenate((ones, e, f))
    C = SVD.truncated_svd(c.T.dot(c))

    # Construction the Riccati ODE
    riccati_ode = RiccatiOde(A, C, D)

    # Initial condition: zero -> solve to h0 -> truncated SVD
    X0 = np.zeros(A.shape)
    h0 = 0.01
    X0 = solve_matrix_ivp(riccati_ode, (0, h0), X0, dense_output=True)
    
    return riccati_ode, X0
