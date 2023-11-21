"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the RationalKrylovSpace class and methods
"""

# %% Imports
import numpy as np
from numpy import ndarray
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix
from .space_structure import SpaceStructure


class RationalKrylovSpace(SpaceStructure):
    """
    Rational Krylov space

        RK_m(A, X) = span{X, (A- p1 * I)^{-1} X, ..., prod_{i=1}^(m-1) (A- p_i * I)^{-1} X}
        for given poles p_i

    How to use
    ----------
    1. Initialize the rational Krylov space with the matrix A, the vector X and the poles p.
    2. Augment the basis of the rational Krylov space as needed with the method `augment_basis`.
    3. The basis is stored in the attribute 'basis', or 'Q' for short.

    Attributes
    ----------
    A : spmatrix
        Sparse matrix of the linear system of shape (n, n)
    X : ndarray
        Initial vector or matrix of shape (n, r)
    poles : list
        Poles of the rational Krylov space in a list of length m-1
    m : int
        Size of the rational Krylov space
    Q : ndarray
        Basis of the rational Krylov space of shape (n, m*r)
    basis : ndarray
        Pointer to Q
    """

    #%% INITIALIZATION
    def __init__(self, A: spmatrix, X: ndarray, poles: list, inverse_only: bool = False, **extra_args) -> None:
        """
        Initialize a rational Krylov space

        Parameters
        ----------
        A : spmatrix
            Sparse matrix of shape (n, n)
        X : ndarray
            Vector or matrix of shape (n, r)
        poles : list
            Poles of the rational Krylov space in a list of length m-1
        inverse_only : bool
            True will compute only the inverse of the poles (1/q), False will compute the inverse and the product (p/q).
        extra_args : dict
            Extra arguments

        Extra arguments
        ---------------
        symmetric : bool
            True if A is symmetric, False otherwise
        """
        # Call parent class
        super().__init__(A, X, **extra_args)
        # Check and store specific parameters
        self.poles = np.array(poles)
        self.max_iter = len(poles)
        # dtype depends on the dtype of the matrix A, X and poles
        for pole in poles:
            if np.iscomplex(pole):
                self.dtype = np.promote_types(self.dtype, np.complex128)
        # For rational Krylov, the symmetric flag is not used
        Q, H = la.qr(X, mode="economic")
        self.Q, self.H = np.array(Q, dtype=self.dtype), np.array(H, dtype=self.dtype)
        if inverse_only:
            self.small_matvec = lambda x: x
        else:
            self.small_matvec = lambda x: A.dot(x)

    #%% PROPERTIES
    @property
    def basis(self) -> int:
        return self.Q

    @property
    def size(self) -> int:
        return self.m * self.r

    #%% BASIS AUGMENTATION
    def augment_basis(self):
        """
        Augment the basis of the rational Krylov space
        """
        # Check if the basis is already full
        if self.m * self.r >= self.n:
            raise ValueError("The space is exceeding the dimension of the problem")
        # Check the poles
        if self.m-2 >= len(self.poles):
            raise ValueError("The number of poles is smaller than the size of the space")
        
         # Initialize
        A = self.A
        r = self.r
        self.m += 1
        m = self.m
        Q = np.zeros((self.n, m * r), dtype=self.dtype)
        Q[:, : (m - 1) * r] = self.Q

        # Solve the next linear system
        matvec = lambda x: spsla.spsolve((self.A - self.poles[self.m-2] * sps.eye(self.n, format='csc')), self.small_matvec(x)).reshape(x.shape)
        Wm = matvec(Q[:, (m - 2) * r : (m - 1) * r])

        # Update-orthogonalization
        self.Q, self.H = la.qr_insert(self.Q, self.H, Wm, (m-1)*r, 'col')

        
