"""
Riccati problems.

Author: Benjamin Carrel, University of Geneva
"""

#%% Importations
import numpy as np
import scipy.sparse as sps
from low_rank_toolbox import SVD
from matrix_ode_toolbox import RiccatiOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp



# %% Riccati from Ostermann (2019)
def make_riccati_ostermann(m: int, q: int) -> tuple[RiccatiOde, np.ndarray]:
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
