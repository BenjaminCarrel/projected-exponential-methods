"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the definition of the InvertedKrylovSpace class.
"""

# %% Imports
from __future__ import annotations
import numpy as np
from numpy import ndarray
import scipy.linalg as la
import scipy.sparse.linalg as spsla
from scipy.sparse import spmatrix
from .space_structure import SpaceStructure
from .krylov_space import KrylovSpace
from .inverted_krylov_space import InvertedKrylovSpace



# %% Class definition
class ExtendedKrylovSpace(SpaceStructure):
    """
    Extended Krylov space.

    EK_m = span{X, AX, A^2X, ..., A^(m-1)X} + span{A^(-1)X, A^(-2)X, ..., A^(-m)X}

    How to use
    ----------
    1. Create an instance of the class: EK = InvertedKrylovSpace(A, X, invA: optional)
    2. Augment the basis as needed: EK.augment_basis(). This will compute A^(k-1)X and A^(-k)X where k is the current size of the basis.
    3. The basis is stored in EK.basis, or EK.Q.


    Attributes
    ----------
    A : spmatrix
        The matrix A of the linear system.
    X : ndarray
        Vector of shape (n,1) or matrix of shape (n,r) that defines the basis of the Krylov space.
    m : int
        The size of the Krylov space.
    invA: callable
        The function that computes the action of A^(-1) on a vector, or a matrix.
    Q : ndarray
        The basis of the extended Krylov space.
    Q1 : ndarray
        The basis of the Krylov space.
    Q2 : ndarray
        The basis of the inverted Krylov space.
    basis : ndarray
        Pointer to Q.
    """

    def __init__(self, A: spmatrix, X: ndarray, invA: callable = None, **extra_args):
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
        # Call parent constructor
        super().__init__(A, X, **extra_args)

        # Check specific inputs
        assert invA is None or callable(invA), "invA must be a function."
        if invA is None:
            spluA = spsla.splu(A)
            invA = spluA.solve
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Store specific inputs
        self.invA = invA

        # Krylov space
        self.krylov_space = KrylovSpace(A, X, is_symmetric=self.is_symmetric)

        # Inverted Krylov space
        self.inverted_krylov_space = InvertedKrylovSpace(A, X, invA=invA, is_symmetric=self.is_symmetric)

    # %% Properties
    @property
    def Q1(self) -> ndarray:
        return self.krylov_space.Q
    
    @property
    def H1(self) -> ndarray:
        return self.krylov_space.H

    @property
    def Q2(self) -> ndarray:
        return self.inverted_krylov_space.Q
    
    @property
    def H2(self) -> ndarray:
        return self.inverted_krylov_space.H
    
    @property
    def Q(self) -> ndarray:
        # # Re-order the two basis
        # Q = np.zeros((self.n, 2 * self.size * self.r))
        # counter1 = 0
        # counter2 = 0
        # tot = 0
        # for _ in np.arange(self.size):
        #     Q[:, tot : tot + self.r] = self.Q1[:, counter1 : counter1 + self.r]
        #     Q[:, tot + self.r : tot + 2 * self.r] = self.Q2[:, counter2 : counter2 + self.r]
        #     counter1 += self.r
        #     counter2 += self.r
        #     tot += 2 * self.r
        # Brutal way
        Q, _ = la.qr_insert(self.Q1, self.H1, self.Q2, self.m * self.r, which="col")
        return Q

    @property
    def basis(self) -> ndarray:
        return self.Q

    @property
    def size(self) -> int:
        """Return the size of the extended Krylov space."""
        return self.krylov_space.size + self.inverted_krylov_space.size

    # %% Methods
    def augment_basis(self):
        """Augment the basis of the extended Krylov space."""
        self.krylov_space.augment_basis()
        self.inverted_krylov_space.augment_basis()
        

    


        




