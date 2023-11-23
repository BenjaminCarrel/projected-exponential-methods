"""
File for generating the Allen-Cahn problem.

Author: Benjamin Carrel, University of Geneva
"""

# %% IMPORTATIONS
import numpy as np
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox import SylvesterLikeOde
from scipy import sparse


def make_allen_cahn(size: int):
    """
    Allen-Cahn equation
        X' = AX + XA + X - X^3
        X(0) = X0
    where A is the 1D Laplacian (times epsilon) as stencil 1/dx^2 [1 -2 1] in csc format, periodic BC
    """
    ## DISCRETIZATION
    xs = np.linspace(0, 2 * np.pi, size)
    dx = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs)

    ## OPERATOR: A is the 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format, periodic BC
    epsilon = 0.01
    # A = epsilon * laplacian_1d_dx2(size, dx, periodic=True)
    A = epsilon * laplacian_1d_dx4(size, dx, periodic=True)

    ## SOURCE: G(t) = X - X^3
    def G(t, X):
        if isinstance(X, LowRankMatrix):
            return X - X.hadamard(X).hadamard(X)
        else:
            return X - X**3
    
    ## DEFINE THE ODE
    ode = SylvesterLikeOde(A, A, G)

    ## INITIAL VALUE
    u = lambda x, y: (np.exp(-np.tan(x)**2) + np.exp(-np.tan(y)**2)) * np.sin(x) * np.sin(y) / (1 + np.exp(np.abs(1/np.sin(-x/2))) + np.exp(np.abs(1/np.sin(-y/2))))
    f = lambda x, y: u(x,y) # - u(x, 2*y) + u(3*x + np.pi, 3*y + np.pi) - 2*u(4*x, 4*y) + 2 * u(5*x, 5*y)
    X0 = f(X, Y)
    X0 = SVD.truncated_svd(X0)

    return ode, X0

## Laplacian with error O(dx^2)
def laplacian_1d_dx2(n, dx, periodic=False):
    """
    Discrete Laplacian matrix in 1D (error O(dx^2))
    """
    DD = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csc') / (dx ** 2)
    if periodic:
        DD[0, -1] = 1 / (dx ** 2)
        DD[-1, 0] = 1 / (dx ** 2)
    return DD

## Laplacian with error O(dx^4)
def laplacian_1d_dx4(n, dx, periodic=False):
    """
    Discrete Laplacian matrix in 1D (error O(dx^4))
    """
    DD = sparse.diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(n, n), format='csc') / (12 * dx ** 2)
    if periodic:
        DD[0, -2] = -1 / (12 * dx ** 2)
        DD[0, -1] = 16 / (12 * dx ** 2)
        DD[1, -1] = -1 / (12 * dx ** 2)
        DD[-1, 0] = 16 / (12 * dx ** 2)
        DD[-1, 1] = -1 / (12 * dx ** 2)
        DD[-2, 0] = -1 / (12 * dx ** 2)
    return DD