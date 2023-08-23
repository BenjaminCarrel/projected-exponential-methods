"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the definition of the InvertedKrylovSpace class.
"""


# %% Imports
from numpy import ndarray
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix
from .krylov_space import KrylovSpace

# %% Class definition
class InvertedKrylovSpace(KrylovSpace):
    """
    Inverted Krylov space.
    
    IK_m = span{A^(-1)X, A^(-2)X, ..., A^(-m)X}

    This class is a wrapper around the Krylov space class. It calls the Krylov space class with the matrix A^(-1) for the matvec function.
    """

    def __init__(self, A: spmatrix, X: ndarray, invA: callable = None, **extra_args) -> None:
        """
        Parameters
        ----------
        A : spmatrix
            The matrix A of the linear system.
        X : ndarray
            The basis of the Krylov space.
        invA: callable
            The function that computes the action of A^(-1) on a vector, or a matrix.
        """
        # Check if invA is provided
        if invA is None:
            spluA = spsla.splu(A)
            invA = lambda x: spluA.solve(x).reshape(x.shape) # the reshape is needed for the case where x is a vector (because of the QR)

        # Define the matvec function
        def matvec(v):
            return invA(v)

        # Call the KrylovSpace class
        X = invA(X) # the inverted Krylov space starts from A^(-1)X
        super().__init__(A, X, matvec=matvec, **extra_args)