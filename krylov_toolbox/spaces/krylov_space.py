"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the KrylovSpace class.
"""

#%% Imports
from __future__ import annotations
import warnings
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from numpy import ndarray
from scipy.sparse import spmatrix, diags, csc_matrix
from .space_structure import SpaceStructure

#%% Class definition
class KrylovSpace(SpaceStructure):
    """
    Class for (block) Krylov spaces.
    The definition of a Krylov space of size $m$ is the following:
        K_m(A,x) = span{x, A x, A^2 x, ..., A^(m-1) x}
    where $A$ is a sparse matrix, and $x$ is a vector or a matrix.

    How to use
    ----------
    1. Initialize the Krylov space with the matrix $A$ and the vector $x$.
    2. Augment the basis of the Krylov space as needed with the method `augment_basis`.
    3. The basis is stored in the attribute 'basis', or 'Q' for short.

    Attributes
    ----------
    A : spmatrix
        Matrix of shape (n,n)
    X : ndarray
        Vector of shape (n,1) or (n,r)
    m : int
        Size of the Krylov space
    symmetric : bool
        True if A is symmetric, False otherwise
    Q : ndarray
        Matrix of shape (n,m) or (n,m*r) containing the basis of the Krylov space
    basis : ndarray
        Pointer to Q
    """

    #%% INITIALIZATION
    def __init__(self, A: spmatrix, X: ndarray, **extra_args) -> None:
        """
        Initialize a Krylov Space where X is a vector or a matrix

        Parameters
        ----------
        A : spmatrix
            Sparse matrix of shape (n,n)
        X : ndarray
            Vector or matrix of shape (n,) or (n,r)
        extra_args : dict
            Extra arguments

        Extra arguments
        ---------------
        is_symmetric : bool
            True if A is symmetric, False otherwise
        matvec: callable
            Function for the matrix-vector product
        """
        # Call parent class
        super().__init__(A, X, **extra_args)

        # Check for a function to compute the matrix-vector product
        if "matvec" in extra_args:
            self.matvec = extra_args["matvec"]
        else:
            self.matvec = lambda x: A.dot(x)

        # Symmetric case -> Lanczos algorithm
        if self.is_symmetric:
            self._alpha = np.empty(self.n, dtype=object)
            self._beta = np.empty(self.n, dtype=object)
            self.Q, self._beta[0] = la.qr(X, mode="reduced")
        # Non symmetric case -> Arnoldi algorithm
        else:
            self.Q, self.H = la.qr(X, mode="reduced")


    #%% PROPERTIES
    @property
    def basis(self):
        return self.Q

    @property
    def size(self):
        return self.m * self.r

    #%% AUGMENT BASIS
    def augment_basis(self) -> None:
        """
        Augment the basis of the space.
        If A is symmetric, the Arnoldi algorithm is used.
        If A is not symmetric, the Lanczos algorithm is used.
        """
        # Check the next size does not exceed the dimension of the matrix
        if self.n < self.r * (self.m + 1):
            # warn user and do nothing
            warnings.warn("The next basis would exceed the dimension of the matrix.")
            return

        # Initialize
        r = self.r
        self.m += 1
        m = self.m
        AQ = self.matvec(self.Q[:, (m - 2) * r : (m - 1) * r])

        # Symmetric case (Lanczos)
        if self.is_symmetric:
            Q = np.zeros((self.n, m * r), dtype=self.Q.dtype)
            Q[:, : (m - 1) * r] = self.Q
            self._alpha[m-1] = Q[:, (m - 2) * r : (m - 1) * r].T.dot(AQ)
            AQ -= Q[:, (m - 2) * r : (m - 1) * r].dot(self._alpha[m-1])
            if m > 2:
                AQ-= Q[:, (m - 3) * r : (m - 2) * r].dot(self._beta[m-2].T)
            Q[:, (m - 1) * r : m * r], self._beta[m-1] = la.qr(AQ, mode="reduced")
            self.Q = Q

        # Non-symmetric case (scipy's qr_insert)
        else:
            self.Q, self.H = sla.qr_insert(self.Q, self.H, AQ, (m - 1) * r, which="col")

        

        



    

