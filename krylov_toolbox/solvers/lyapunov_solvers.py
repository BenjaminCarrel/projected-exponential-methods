"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains solvers for the Lyapunov equation.
"""

#%% Imports
import numpy as np
from numpy import ndarray
import scipy.linalg as la
from scipy.sparse import spmatrix
import scipy.sparse.linalg as spsla
from low_rank_toolbox import LowRankMatrix, QuasiSVD
from krylov_toolbox import KrylovSpace, ExtendedKrylovSpace, RationalKrylovSpace

machine_precision = np.finfo(float).eps
Matrix = ndarray | spmatrix | LowRankMatrix

#%% FUNCTIONS

def solve_small_lyapunov(A: ndarray, C: ndarray) -> ndarray:
    "Solve the Lyapunov equation AX + XA = C for small matrices."
    return la.solve_lyapunov(A, C)

def solve_sparse_low_rank_lyapunov(A: spmatrix,
                                   C: LowRankMatrix,
                                   tol: float = 1e-12,
                                   max_iter: int = None,
                                   **kwargs) -> QuasiSVD:
    """Low-rank solver for the Lyapunov equation.
    Large, sparse matrix A.
    Find X such that AX + XA = C.
    The method is similar to the one for the Sylvester equation, but twice as fast.
    NOTE: C must be symmetric.

    Parameters
    ----------
    A : spmatrix
        The matrix A of shape (n, n)
    C : LowRankMatrix
        The low-rank matrix C of shape (n, n)
    tol : float, optional
        The tolerance, by default 1e-12
    max_iter : int, optional
        The maximum number of iterations, by default 100
    kwargs : dict
        Additional arguments for the Krylov solver

    Keyword Arguments
    -----------------
    inverted : bool
        If True, use inverted Krylov space, by default False.
    invA : callable
        Function that applies the inverse of A, by default None.
    poles_A : array_like
        If given, a rational Krylov space is used for the left space, with the given poles. By default None.

    Returns
    -------
    QuasiSVD
        The low-rank solution X
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(C, LowRankMatrix), "C must be a low-rank matrix"
    assert tol > machine_precision, "tol must be larger than machine precision"
    if max_iter is None:
        max_iter = int(A.shape[0] / C.rank)
    assert max_iter > 1, "max_iter must be greater than 1"

    # Note: C must be symmetric for the method to work. It is not efficient so prefer sylvester solvers otherwise.
    Cd = C.to_dense()
    assert np.allclose(Cd, Cd.T), "C must be symmetric"

    # Check keyword arguments
    extended = kwargs.get("extended", True)
    invA = kwargs.get("invA", None)
    poles_A = kwargs.get("poles_A", None)

    if extended and poles_A is not None:
        raise ValueError("Cannot use rational Krylov space with inverted Krylov space")


    # Precompute some quantities
    normA = spsla.norm(A)
    normC = C.norm()

    # Define the Krylov space
    if extended:
        if invA is None:
            invA = lambda x: spsla.spsolve(A, x)
        krylov_space = ExtendedKrylovSpace(A, C._matrices[0], invA)
    elif poles_A is not None:
        krylov_space = RationalKrylovSpace(A, C._matrices[0], poles_A)
    else:
        krylov_space = KrylovSpace(A, C._matrices[0])
        print('Warning: standard Krylov space may not converge')
        
    # Current basis
    Uk = krylov_space.Q

    # SOLVE SMALL PROJECTED LYAPUNOV IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED LYAPUNOV Ak Y + Y Ak = Ck
        Ak = Uk.T.dot(A.dot(Uk))
        Ck = (C.dot(Uk)).dot(Uk.T, side="opposite", dense_output=True)  # Vt @ RHS @ V
        Yk = la.solve_lyapunov(Ak, Ck)

        # CHECK CONVERGENCE
        Xk = QuasiSVD(Uk, Yk, Uk)
        AXk = Xk.dot_sparse(A, side="opposite")
        XkA = Xk.dot_sparse(A)
        # computation of crit could be more efficient, but SVD so its OK for now.
        crit = (C - AXk - XkA).norm() / \
            (2 * normA * la.norm(Yk) + normC)
        # print(crit)
        if crit < tol:
            return Xk.to_svd().truncate() # truncate up to machine precision since the criterion overestimates the error
        else:
            krylov_space.augment_basis()
            Uk = krylov_space.Q

    print('No convergence before max_iter')
    X = QuasiSVD(Uk, Yk, Uk)
    return X


def solve_lyapunov(A: ndarray | spmatrix,
                   C: ndarray | LowRankMatrix,
                   **kwargs) -> ndarray | LowRankMatrix:
    """Solve the lyapunov equation AX + XA = C.
    Automatically choose the fastest method based on the size of A and C.

    Parameters
    ----------
    A : Matrix
        The matrix A of shape (n, n)
    C : Matrix
        The matrix C of shape (n, n)
    kwargs : dict
        Additional arguments for the Krylov solver
    """
    # SEPARATE ALL CASES
    if isinstance(A, spmatrix):
        X = solve_sparse_low_rank_lyapunov(A, C, **kwargs)
    elif isinstance(A, ndarray):
        if isinstance(C, LowRankMatrix):
            C = C.to_dense()
        X = solve_small_lyapunov(A, C)
    return X




