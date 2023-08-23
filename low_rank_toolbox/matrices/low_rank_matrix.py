# Authors: Benjamin Carrel and Rik Vorhaar
#         University of Geneva, 2022
# File for generic low-rank matrix format
# Path: low_rank_toolbox/matrices/low_rank_matrix.py

# Import packages
from __future__ import annotations
from copy import deepcopy
from warnings import warn
import numpy as np
from typing import List, Optional, Sequence, Tuple, Type, Union
from numpy import ndarray
import scipy.sparse.linalg as spala
from scipy.sparse import spmatrix
from scipy.linalg import block_diag
import scipy.linalg as la



# %% Define class LowRankMatrix
class LowRankMatrix:
    """
    Meta class for dealing with low rank matrices in different formats.

    Do not use this class directly, but rather use its subclasses.

    We always decompose a matrix as a product of smaller matrices. These smaller
    matrices are stored in ``self._matrices``.
    """

    _format = "generic"

    def __init__(
        self,
        *matrices: Sequence[ndarray],
        **extra_data,
    ):
        # Convert so values can be changed.
        self._matrices = list(matrices)
        self._extra_data = extra_data

    @property
    def rank(self) -> int:
        return min(min(M.shape) for M in self._matrices)

    @property
    def length(self) -> int:
        "Number of matrices"
        return len(self._matrices)

    @property
    def shape(self) -> tuple:
        return (self._matrices[0].shape[0], self._matrices[-1].shape[-1])

    @property
    def deepshape(self) -> tuple:
        return tuple(
            M.shape[0] for M in self._matrices if len(M.shape) == 2
        ) + (self._matrices[-1].shape[-1],)

    @property
    def dtype(self):
        return self._matrices[0].dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self):
        # reverse order of matrices and transpose each element
        new_matrix = self.copy()
        new_matrix._matrices = [M.T for M in reversed(self._matrices)]
        return new_matrix
    
    @property
    def H(self):
        # reverse order of matrices and conjugate transpose each element
        new_matrix = self.copy()
        new_matrix._matrices = [M.T.conj() for M in reversed(self._matrices)]
        return new_matrix

    def is_symmetric(self) -> bool:
        """Check if the matrix is symmetric. """
        # Check if square
        if self.shape[0] != self.shape[1]:
            return False
        # Check if symmetric
        dense = self.full()
        return np.allclose(dense, dense.T)

    def transpose(self) -> LowRankMatrix:
        """Transpose a low rank matrix."""
        return self.T
    
    ## CLASS METHODS
    @classmethod
    def from_matrix(cls, matrix: ndarray) -> LowRankMatrix:
        """Create a low-rank matrix from a full matrix.
        This does nothing except putting the matrix in the LowRankMatrix format.
        You can use convert() to convert to a different format. 
        Change this by overloading this method."""
        return LowRankMatrix(matrix)

    @classmethod
    def from_full(cls, matrix: ndarray):
        return cls.from_matrix(matrix)

    @classmethod
    def from_dense(cls, matrix: ndarray):
        return cls.from_matrix(matrix)

    @classmethod
    def from_low_rank(cls, low_rank_matrix: LowRankMatrix) -> LowRankMatrix:
        """Overload this method for each subclass"""
        return LowRankMatrix(*low_rank_matrix._matrices)

    def norm(self, ord: str | int = 'fro') -> float:
        """Default implementation, overload this for some subclasses"""
        if ord == 'fro':
            return np.sqrt(np.trace(self.T.dot(self, dense_output=True)))
        else:
            return np.linalg.norm(self.full(), ord=ord)

    def __repr__(self) -> str:
        return (
            f"{self.shape} low-rank matrix rank {self.rank}"
            f" and type {self.__class__._format}."
        )

    def __getitem__(self, indices) -> float:
        return self.gather(indices)

    def copy(self) -> self.__class__:
        return deepcopy(self)

    def __add__(self, other: Union[LowRankMatrix, ndarray]) -> LowRankMatrix:
        """Generic addition method"""
        # other is a LowRankMatrix
        if isinstance(other, LowRankMatrix):
            sum = self.full() + other.full()
        else:
            sum = self.full() + other
        return sum

    def __imul__(self, other: float) -> LowRankMatrix:
        self._matrices[0] *= other
        return self

    def __mul__(self, other: float) -> LowRankMatrix:
        """Scalar multiplication"""
        new_mat = self.copy()
        new_mat.__imul__(other)
        return new_mat

    __rmul__ = __mul__

    def __neg__(self) -> LowRankMatrix:
        return -1 * self

    def __sub__(self, other: LowRankMatrix) -> LowRankMatrix:
        return self + (-1) * other

    def convert(self, format: Type[LowRankMatrix]) -> LowRankMatrix:
        """Convert the low rank matrix to a different format. Generic method,
        this may not be the fastest in every situation."""
        return format.from_low_rank(self)

    def full(self) -> ndarray:
        " Multiply all factors in optimal order "
        return np.linalg.multi_dot(self._matrices)

    def todense(self) -> ndarray:
        return self.full()

    def to_dense(self) -> ndarray:
        return self.full()

    def to_full(self) -> ndarray:
        return self.full()

    def flatten(self) -> ndarray:
        return self.full().flatten()

    def gather(self, indices: ndarray) -> ndarray:
        """Access entries of the full matrix indexed by ``indices``.

        This is faster and more memory-efficient than forming full matrix. Very
        useful for e.g. matrix completion tasks, or estimating reconstruction
        error on large matrices.
        """
        A = self._matrices[0][indices[0], :]
        Z = self._matrices[-1][:, indices[1]]
        return np.linalg.multi_dot([A, *self._matrices[1:-1], Z])

    ## STANDARD MATRIX MULTIPLICATION
    def dot(self,
            other: Union[LowRankMatrix, ndarray, spmatrix],
            side: str = 'right',
            dense_output: bool = False) -> Union[ndarray, LowRankMatrix]:
        """Matrix and vector multiplication

        Parameters
        ----------
        other : LowRankMatrix, ndarray, spmatrix
            Matrix or vector to multiply with.
        side : str, optional
            Whether to multiply on the left or right, by default 'right'.
        dense_output : bool, optional
            Whether to return a dense matrix or a low-rank matrix, by default False.
        """
        # MATRIX-VECTOR CASE
        if len(other.shape) == 1:
            dense_output = True
            
        # SPARSE MATRIX CASE
        if isinstance(other, spmatrix):
            return self.dot_sparse(other, side, dense_output)

        # DENSE OUTPUT
        if dense_output:
            if isinstance(other, LowRankMatrix):
                if side == 'right' or side == 'usual': # usual is for backwards compatibility
                    return np.linalg.multi_dot(self._matrices + other._matrices)
                elif side == 'left' or side == 'opposite': # opposite is for backwards compatibility
                    return np.linalg.multi_dot(other._matrices + self._matrices)
                else:
                    raise ValueError('Incorrect side. Choose "right" or "left".')
            else:
                if side == 'right' or side == 'usual':
                    return np.linalg.multi_dot(self._matrices + [other])
                elif side == 'left' or side == 'opposite':
                    return np.linalg.multi_dot([other] + self._matrices)
                else:
                    raise ValueError('Incorrect side. Choose "right" or "left".')
        
        # LOW RANK OUTPUT (default)
        if isinstance(other, LowRankMatrix):
            if side == 'right' or side == 'usual':
                return LowRankMatrix(
                    *self._matrices, *other._matrices, **self._extra_data
                )
            elif side == 'left' or side == 'opposite':
                return LowRankMatrix(
                    *other._matrices, *self._matrices, **self._extra_data
                )
            else:
                raise ValueError('Incorrect side. Choose "right" or "left".')
        else:
            if side == 'right' or side == 'usual':
                return LowRankMatrix(
                    *self._matrices, other, **self._extra_data
                )
            elif side == 'left' or side == 'opposite':
                return LowRankMatrix(
                    other, *self._matrices, **self._extra_data
                )
            else:
                raise ValueError('Incorrect side. Choose "right" or "left".')

    __matmul__ = dot

    def multi_dot(self, others: Sequence[LowRankMatrix | ndarray]) -> LowRankMatrix:
        """Matrix multiplication of a sequence of matrices.

        Parameters
        ----------
        others : Sequence[LowRankMatrix | ndarray]
            Sequence of matrices to multiply.

        Returns
        -------
        LowRankMatrix
            Low rank matrix representing the product.
        """
        output = self.copy()
        for other in others:
            output = output.dot(other)
        return output

    ## MULTIPLICATION WITH A SPARSE MATRIX
    def dot_sparse(self,
                   sparse_other: spmatrix,
                   side: str = 'usual',
                   dense_output: bool = False) -> Union[ndarray, LowRankMatrix]:
        """
        Efficient sparse multiplication
        usual: output = matrix @ sparse_other
        opposite: output = sparse_other @ matrix
        """
        sparse_other = sparse_other.tocsc() # sanity check
        new_mat = self.copy()
        if side == 'right' or side == 'usual':
            new_mat._matrices[-1] = (sparse_other.T.dot(new_mat._matrices[-1].T)).T
        elif side == 'opposite' or side == 'left':
            new_mat._matrices[0] = sparse_other.dot(new_mat._matrices[0])
        else:
            raise ValueError('incorrect side')
        if dense_output:
            return new_mat.full()
        return  new_mat

    ## EXPONENTIAL ACTION OF A SPARSE MATRIX ON THE LOW-RANK MATRIX
    def expm_multiply(self,
                      A: spmatrix,
                      h: float,
                      side: str = 'left',
                      dense_output: bool = False,
                      **extra_args) -> Union[ndarray, LowRankMatrix]:
        """
        Efficient action of sparse matrix exponential
        left: output = exp(h*A) @ matrix
        right: output = matrix @ exp(h*A)
        """
        A = A.tocsc()  # sanity check
        new_mat = self.copy()
        if side == 'left':
            new_mat._matrices[0] = spala.expm_multiply(A, self._matrices[0], start=0, stop=h, num=2, endpoint=True, **extra_args)[-1]
        elif side == 'right':
            new_mat._matrices[-1] = spala.expm_multiply(A.T, self._matrices[-1].T, start=0, stop=h, num=2, endpoint=True, **extra_args)[-1].T
        else:
            raise ValueError('incorrect side')
        if dense_output:
            return new_mat.to_dense()
        return new_mat


    @staticmethod
    def create_matrix_alias(index: int, transpose=False, conjugate=False) -> property:
        if transpose and conjugate:
            def getter(self):
                return self._matrices[index].T.conj()
            def setter(self, value):
                self._matrices[index] = value.T.conj()
        elif transpose:
            def getter(self):
                return self._matrices[index].T
            def setter(self, value):
                self._matrices[index] = value.T
        elif conjugate:
            def getter(self):
                return self._matrices[index].conj()
            def setter(self, value):
                self._matrices[index] = value.conj()
        else:
            def getter(self) -> ndarray:
                return self._matrices[index]
            def setter(self, value: ndarray):
                self._matrices[index] = value
        return property(getter, setter)

    @staticmethod
    def create_data_alias(key: "str") -> property:
        def getter(self) -> ndarray:
            return self._extra_data[key]

        def setter(self, value: ndarray):
            self._extra_data[key] = value

        return property(getter, setter)
