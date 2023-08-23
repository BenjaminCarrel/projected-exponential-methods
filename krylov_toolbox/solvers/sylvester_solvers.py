"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains solvers for the Sylvester equation.
"""

#%% Imports
import numpy as np
from numpy import ndarray
import scipy.linalg as la
from scipy.sparse import spmatrix
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from low_rank_toolbox import LowRankMatrix, QuasiSVD
from krylov_toolbox import KrylovSpace, ExtendedKrylovSpace, RationalKrylovSpace

machine_precision = np.finfo(float).eps
Matrix = ndarray | spmatrix | LowRankMatrix

#%% Functions

def solve_small_sylvester(A: ndarray, B: ndarray, C: Matrix) -> ndarray:
    """
    Solve the Sylvester equation for small systems. 
    Find X such that AX + XB = C. Wrapper to scipy.linalg.solve_sylvester.
    For larger systems, use solve_sparse_low_rank_sylvester or solve_sylvester_large_A_small_B.
    """
    # Check inputs
    assert isinstance(A, ndarray), "A must be a dense matrix"
    assert isinstance(B, ndarray), "B must be a dense matrix"
    assert isinstance(C, Matrix), "C must be a matrix"
    if not isinstance(C, ndarray):
        C = C.todense()
    return la.solve_sylvester(A, B, C)

def solve_sylvester_large_A_small_B(A: spmatrix,
                                    B: ndarray,
                                    C: ndarray) -> ndarray:
    """Solve the Sylvester equation when A is large and B is small.
    Find X such that AX + XB = C.
    Simoncini, V., 2016. Computational Methods for Linear Matrix Equations. SIAM Rev. 58, 377-441. https://doi.org/10.1137/130912839
    
    Parameters
    ----------
    A : spmatrix
        Sparse matrix of shape (m, m)
    B : ndarray
        Matrix of shape (n, n)
    C : ndarray
        Matrix of shape (m, n)
    
    Returns
    -------
    ndarray
        Dense solution of the Sylvester equation of shape (m, n)
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(B, ndarray), "B must be a dense matrix"
    assert isinstance(C, ndarray), "C must be a dense matrix"

    S, W = la.eigh(B)
    C_hat = C.dot(W)
    X_hat = np.zeros(C.shape)
    I = sps.eye(*A.shape)
    for i in np.arange(len(S)):
        X_hat[:, i] = spsla.spsolve(A + S[i] * I, C_hat[:, i])
    return X_hat @ W.T

def solve_sparse_low_rank_sylvester(A: spmatrix,
                                    B: spmatrix,
                                    C: LowRankMatrix,
                                    tol: float = 1e-12,
                                    max_iter: int = None,
                                    **kwargs) -> QuasiSVD:
    """Low-rank solver for Sylvester equation.
    The matrices A and B are large.
    Find X such that AX + XB = C.
    The method is based on Krylov Space for finding the solution up to a criterion.
    Simoncini, V., 2016. Computational Methods for Linear Matrix Equations. SIAM Rev. 58, 377-441. https://doi.org/10.1137/130912839


    Parameters
    ----------
    A : spmatrix
        Sparse matrix A of the sylvester equation (m, m)
    B : spmatrix
        Sparse matrix B of the sylvester equation (n, n)
    C : LowRankMatrix
        Low-rank matrix C of the sylvester equation (m, n)
    tol : float, optional
        Tolerance for the stopping criterion, by default 1e-12
    max_iter : int, optional
        Maximum number of iterations, by default 100
    kwargs : dict
        Additional arguments for the Krylov solver

    Keyword Arguments
    -----------------
    extended : bool
        If True, use inverted Krylov space, by default False.
    invA : callable
        Function that applies the inverse of A, by default None.
    invB : callable
        Function that applies the inverse of B, by default None.
    poles_A : array_like
        If given, a rational Krylov space is used for the left space, with the given poles. By default None.
    poles_B : array_like
        If given, a rational Krylov space is used for the right space, with the given poles. By default None.

    Returns
    -------
    QuasiSVD
        QuasiSVD (low-rank matrix) of X of the sylvester equation (m, n)
    """
    # Check inputs
    assert isinstance(A, spmatrix), "A must be a sparse matrix"
    assert isinstance(B, spmatrix), "B must be a sparse matrix"
    assert isinstance(C, LowRankMatrix), "C must be a low-rank matrix"
    assert tol > machine_precision, "tol must be larger than machine precision"
    if max_iter is None:
        max_iter = max(int(A.shape[0] / C.rank), 10)
    assert max_iter > 1, "max_iter must be greater than 1"

    # Check keyword arguments
    extended = kwargs.get("extended", True)
    invA = kwargs.get("invA", None)
    invB = kwargs.get("invB", None)
    poles_A = kwargs.get("poles_A", None)
    poles_B = kwargs.get("poles_B", None)

    if extended and (poles_A is not None or poles_B is not None):
        raise ValueError("Cannot use rational Krylov space with inverted Krylov space")

    # Precompute some quantities
    normA = spsla.norm(A)
    normB = spsla.norm(B)
    normC = C.norm()
    U, V = C._matrices[0], C._matrices[-1].T

    # Define the Krylov space
    ## Left space
    if extended:
        if invA is None:
            invA = lambda x: spsla.spsolve(A, x)
        left_space = ExtendedKrylovSpace(A, U, invA)
    elif poles_A is not None:
        left_space = RationalKrylovSpace(A, U, poles_A)
        right_space = RationalKrylovSpace(B, V, poles_B)
    else:
        print('Warning: by using the default Krylov space, the algorithm may not converge. Consider using extended krylov space or rational krylov space.')
        left_space = KrylovSpace(A, U)


    ## Right space
    if extended:
        if invB is None:
            invB = lambda x: spsla.spsolve(B, x)
        right_space = ExtendedKrylovSpace(B, V, invB)
    elif poles_B is not None:
        right_space = RationalKrylovSpace(B, V, poles_B)
    else:
        right_space = KrylovSpace(B, V)
    
    # Current basis
    Uk = left_space.Q
    Vk = right_space.Q

    # SOLVE SMALL PROJECTED SYLVESTER IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED SYLVESTER Ak Y + Y Bk = Ck
        Ak = Uk.T.dot(A.dot(Uk))
        Bk = Vk.T.dot(B.dot(Vk))
        Ck = (C.dot(Vk)).dot(Uk.T, side="opposite", dense_output=True)  # Vt @ Ck @ W (small dense matrix)
        Yk = la.solve_sylvester(Ak, Bk, Ck)

        # CHECK CONVERGENCE
        Xk = QuasiSVD(Uk, Yk, Vk)
        AXk = Xk.dot(A, side="opposite")
        XkB = Xk.dot(B)
        # computation of crit could be more efficient, but SVD so its OK for now.
        crit = (AXk + XkB - C).norm() / \
            ((normA + normB) * la.norm(Yk) + normC)
        # print(crit)
        if crit < tol or k == max_iter - 1:
            # truncate to machine precision since the criterion overestimates the error
            # NOTE: the user might want to change this, but works fine in most cases and reduces the cost
            return Xk.to_svd().truncate() 

        else:
            left_space.augment_basis()
            Uk = left_space.Q
            right_space.augment_basis()
            Vk = right_space.Q

    print('No convergence before max_iter')
    X = QuasiSVD(Uk, Yk, Vk)
    return X

def solve_sylvester(A: ndarray | spmatrix,
                    B: ndarray | spmatrix,
                    C: ndarray | LowRankMatrix,
                    **kwargs) -> ndarray | QuasiSVD:
    """Solve the sylvester equation AX + XB = C.
    Automatically choose the adapted method for solving the equation efficiently.

    Parameters
    ----------
    A : Matrix
        Matrix A of the sylvester equation (m, m)
    B : Matrix
        Matrix B of the sylvester equation (n, n)
    C : Matrix
        Matrix C of the sylvester equation (m, n)
    kwargs : dict
        Additional arguments when A and B are sparse matrices. See solve_sylvester_sparse for more details.

    Returns
    -------
    Matrix
        Solution of the sylvester equation
    """
    # CALL THE BEST SOLVER
    if isinstance(A, spmatrix):
        if isinstance(B, spmatrix):
            # A AND B SPARSE, C IS LOW-RANK -> X IS LOW-RANK
            X = solve_sparse_low_rank_sylvester(A, B, C, **kwargs)
        else:
            # A SPARSE AND B SMALL, C IS DENSE -> X IS DENSE
            X = solve_sylvester_large_A_small_B(A, B, C)
    else:
        if isinstance(B, spmatrix):
            # NEVER USED SO FAR
            X = NotImplementedError
        else:
            # A AND B SMALL, C IS DENSE -> X IS DENSE
            X = solve_small_sylvester(A, B, C)
    return X
