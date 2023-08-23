"""
Author: Benjamin Carrel, University of Geneva, 2022

This module contains the definition of the SpaceStructure class.
"""

# %% Imports
from __future__ import annotations
import numpy as np
from numpy import ndarray
import scipy.linalg as la
from scipy.sparse import spmatrix


# %% Class definition
class SpaceStructure:
    """Space structure.
    
    General space structure class. This class is meant to be inherited by other classes that define specific space structures, like Krylov spaces, rational Krylov spaces, etc.

    In particular, this class defines the following attributes:
    - A: the matrix A (typically from a linear system A Y = X).
    - X: the vector or matrix that defines the basis of the space.
    - size: the size of the space.
    - basis: the basis of the space.
    - extra_args: a dictionary that contains extra arguments that can be passed to the class.
    """

    def __init__(self, A: spmatrix, X: ndarray, **extra_args) -> None:
        """
        Parameters
        ----------
        A : spmatrix
            The matrix A of the linear system.
        X : ndarray
            The basis of the space.
        extra_args: dict
            A dictionary that contains extra arguments that can be passed to the class.
        """
        # Check inputs
        self.check_inputs(A, X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Store inputs
        self.A = A
        self.X = X
        self.extra_args = extra_args
        self.n, self.r = X.shape
        self.m = 1
        self.k = self.m

        # Check for symmetry
        if 'is_symmetric' in extra_args:
            self.is_symmetric = extra_args['is_symmetric']
        else:
            if not abs(A - A.T).nnz:
                self.is_symmetric = True
            else:
                self.is_symmetric = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} of size {self.size} with basis of shape {self.basis.shape}"

    #%% PROPERTIES
    @property
    def size(self) -> int:
        """The size of the space. Overload this method in the child class."""
        return NotImplementedError("The size method is not implemented in the parent class.")

    @property
    def reduced_A(self) -> ndarray:
        """The reduced matrix A."""
        return self.basis.T.dot(self.A.dot(self.basis))

    @property
    def Am(self) -> ndarray:
        """Shortcut for the reduced matrix A."""
        return self.reduced_A
    
    @property
    def Ak(self) -> ndarray:
        """Shortcut for the reduced matrix A."""
        return self.reduced_A
    
    # %% CLASS METHODS
    @classmethod
    def check_inputs(cls, A, X):
        assert isinstance(A, spmatrix), "A must be a sparse matrix"
        assert isinstance(X, ndarray), "X must be a numpy array"
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        if A.shape[0] != X.shape[0]:
            raise ValueError("A and X must have the same number of rows")
        pass

    # %% Methods to be overloaded in the child class
    @property
    def basis(self) -> ndarray:
        """The basis of the space. Overload this property in the child class."""
        return NotImplementedError("The basis property is not implemented in the parent class.")

    def augment_basis(self):
        """Augment the space with a new basis. Overload this method in the child class."""
        return NotImplementedError("The augment method is not implemented in the parent class.")



