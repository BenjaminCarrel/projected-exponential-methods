# Authors: Benjamin Carrel and Rik Vorhaar
#          University of Geneva, 2022
# File for SVD low-rank matrix related classes and functions
# Path: low_rank_toolbox/matrices/svd.py

# Imports
from __future__ import annotations

from .low_rank_matrix import LowRankMatrix
from numpy import ndarray
import numpy as np
import scipy.linalg as la
from typing import List
import warnings

Matrix = ndarray | LowRankMatrix

# Parameters
automatic_truncation = True
default_atol = 100 * np.finfo(float).eps


#%% Define class QuasiSVD
class QuasiSVD(LowRankMatrix):
    """
    Quasi Singular Value Decomposition (Quasi-SVD)
    X = U @ S @ V.T
    where U, V are orthonormal and S is non-singular (not necessarly diagonal)
    If X is low-rank, it is much more efficient to store only U, S, V.
    Behaves like a numpy array but preserves the low-rank structure.
    # NOTE: after multiplication, the orthogonality of U and V is lost in general.
    Therefore, the class provides a method to convert from LowRankMatrix.
    It is also possible to efficiently convert a QuasiSVD into an SVD.
    """

    # Class attributes
    _format = "QuasiSVD"

    # Aliases for the matrices
    U = LowRankMatrix.create_matrix_alias(0)
    S = LowRankMatrix.create_matrix_alias(1)
    V = LowRankMatrix.create_matrix_alias(2, transpose=True, conjugate=True)
    Vh = LowRankMatrix.create_matrix_alias(2)
    Vt = LowRankMatrix.create_matrix_alias(2, conjugate=True)
    Ut = LowRankMatrix.create_matrix_alias(0, transpose=True)
    Uh = LowRankMatrix.create_matrix_alias(0, transpose=True, conjugate=True)

    def __init__(self, U: ndarray, S: ndarray, V: ndarray, **extra_data):
        """
        Create a low-rank matrix stored by its SVD: Y = U @ S @ V.T

        NOTE: U and V must be orthonormal
        NOTE: S is not necessarly diagonal and can be rectangular
        NOTE: The user must give V and not V.T or V.H

        Parameters
        ----------
        U : ndarray
            Left singular vectors, shape (m, r)
        S : ndarray
            Non-singular matrix, shape (r, q)
        V : ndarray
            Right singular vectors, shape (n, q)
        """
        # Check inputs
        # if V.shape[0] == V.shape[1]:
        #     warnings.warn("V is square, double check that it is not transposed")
        assert U.dtype == V.dtype, "U and V must have the same dtype"
        # Call the parent constructor
        super().__init__(U, S, V.T.conj(), **extra_data)

    ## SPECIFIC PROPERTIES
    @property
    def is_symmetric(self) -> bool:
        return np.allclose(self.U, self.Vt.T)

    @property
    def check_orthogonality(self) -> bool:
        "Check if U and V are orthogonal"
        # Orthogonality of U and V
        assert np.allclose(self.Uh @ self.U, np.eye(self.U.shape[1])), "U is not orthogonal"
        assert np.allclose(self.Vh @ self.V, np.eye(self.V.shape[1])), "V is not orthogonal"
        # Non-singularity of S
        assert np.linalg.cond(self.S) < 1 / np.finfo(float).eps, "S is singular"
        return True

    @property
    def sing_vals(self) -> ndarray:
        return la.svdvals(self.S)

    def is_symmetric(self) -> bool:
        # Check squareness
        if self.shape[0] != self.shape[1]:
            return False
        # Check symmetry
        return np.allclose(self.U, self.V)

    ## CLASS METHODS
    @classmethod
    def multi_add(cls, matrices: List[QuasiSVD], truncate: bool = automatic_truncation) -> SVD:
        """
        Addition of multiple QuasiSVD matrices.

        The rank of the output is the sum of the ranks of the inputs, at maximum.
        The default behavior depends on the value of automatic_truncation.
        If automatic_truncation is True, the output is truncated to ensure non-singularity of S.
        In that case, adding matrices one by one is often more efficient.
        If automatic_truncation is False, the output is not truncated and can be singular.
        In that case, adding matrices all at once is more efficient.
        
        Parameters
        ----------
        matrices : List[QuasiSVD]
            Matrices to add
        truncate : bool, optional
            Truncate the output to ensure non-singularity of S, by default automatic_truncation.
        """
        # Check inputs
        assert all(isinstance(matrix, QuasiSVD) for matrix in matrices), "All matrices must be QuasiSVD"
        assert all(matrix.shape == matrices[0].shape for matrix in matrices), "All matrices must have the same shape"
        # Add the matrices
        U_stack = np.column_stack([*[matrix.U for matrix in matrices]])
        V_stack = np.column_stack([*[matrix.V for matrix in matrices]])
        S_stack = la.block_diag(*[matrix.S for matrix in matrices])
        # Necessary steps to get orthogonality of U and V
        Q1, R1 = la.qr(U_stack, mode='economic')
        Q2, R2 = la.qr(V_stack, mode='economic')
        M = np.linalg.multi_dot([R1, S_stack, R2.T.conj()])
        u, s, vh = la.svd(M, full_matrices=False)
        new_U = Q1.dot(u)
        new_s = s
        new_V = Q2.dot(vh.T.conj())
        if truncate:
            # Truncation guarantees non-singularity of S
            # atol is necessary for subtraction of identical matrices
            # but it loses precision
            return SVD(new_U, new_s, new_V).truncate(atol = default_atol) 
        else:     
            return SVD(new_U, new_s, new_V)

    @classmethod
    def multi_dot(cls, matrices: List[QuasiSVD], truncate: bool = automatic_truncation) -> SVD:
        """
        Matrix multiplication of several QuasiSVD matrices.

        The rank of the output is the minimum of the ranks of the first and last inputs, at maximum.
        The default behavior depends on the value of automatic_truncation.
        If automatic_truncation is True, the output is truncated to ensure non-singularity of S.
        If automatic_truncation is False, the output is not truncated and can be singular.
        The matrices are multiplied all at once, so this method is more efficient than multiplying them one by one. 
        
        Parameters
        ----------
        matrices : List[QuasiSVD]
            Matrices to multiply
        """
        # Check inputs
        assert all(isinstance(matrix, QuasiSVD) for matrix in matrices), "All matrices must be QuasiSVD"
        # Multiply the matrices
        U = matrices[0].U
        V = matrices[-1].V
        M = np.linalg.multi_dot([matrices[0].S, matrices[0].Vh] + [matrix._matrices for matrix in matrices[1:-1]] + [matrices[-1].U, matrices[-1].S])
        if truncate:
            return QuasiSVD(U, M, V).truncate(atol=default_atol)
        else:
            return QuasiSVD(U, M, V)

    def to_svd(self) -> SVD:
        "Convert a QuasiSVD into an SVD"
        output = SVD.truncated_svd(self.S)
        output.U = self.U @ output.U
        output.V = self.V @ output.V
        return output
    
    def truncate(self, r: int = None, rtol: float = default_atol, atol: float = None) -> SVD:
        "The QuasiSVD is transformed into an SVD and then truncated"
        return self.to_svd().truncate(r=r, rtol=rtol, atol=atol, inplace=True)

    def norm(self, ord: str | int = 'fro') -> float:
        """Calculate norm. Default is Frobenius norm"""
        return la.norm(self.S, ord=ord)

    def project_onto_tangent_space(self, other: Matrix, truncate: bool = automatic_truncation) -> SVD:
        """
        Projection of other onto the tangent space at self.

        The rank of the output is two times the rank of self, at maximum.
        The default behavior depends on the value of automatic_truncation.
        If automatic_truncation is True, the output is truncated to ensure non-singularity of S.
        If automatic_truncation is False, the output is not truncated and can be singular.
        
        The formula is given by:
            P_X Y = UUh Y - UUh Y VVh + Y VVh
        where X = U S Vh is the SVD of matrix self and Y is the matrix other to project.

        Parameters 
        ----------
        other : ndarray or LowRankMatrix
            Matrix to project
        """
        # STEP 1 : FACTORIZATION
        if isinstance(other, LowRankMatrix):
            YV = other.dot(self.V, dense_output=True)
            UhY = other.dot(self.Uh, side='opposite', dense_output=True)
        else:
            YV = other.dot(self.V)
            UhY = self.Uh.dot(other)
        UhYVVh = np.linalg.multi_dot([self.Uh, YV, self.Vh])
        M1 = np.column_stack([self.U, YV])
        M2 = np.row_stack([UhY - UhYVVh, self.Vh])
        # STEP 2 : DOUBLE QR  (n times 2k)
        Q1, R1 = la.qr(M1, mode='economic')
        Q2, R2 = la.qr(M2.T.conj(), mode='economic')
        if truncate:
            return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2).truncate(atol=default_atol)
        else:
            return QuasiSVD(Q1, R1.dot(R2.T.conj()), Q2)

    ## STANDARD OPERATIONS
    def __add__(self, other: QuasiSVD | ndarray) -> SVD | ndarray:
        """Special addition for QuasiSVD"""
        if isinstance(other, QuasiSVD):
            return QuasiSVD.multi_add([self, other])
        else:
            return super().__add__(other)
        
    def __sub__(self, other: QuasiSVD | ndarray) -> SVD | ndarray:
        """Sepcial subtraction for QuasiSVD"""
        if isinstance(other, QuasiSVD):
            return QuasiSVD.multi_add([self, -other])
        else:
            return super().__sub__(other)


    def __imul__(self, other: float | Matrix) -> Matrix:
        """In-place scalar multiplication, or hadamard product if other is a matrix"""
        if isinstance(other, Matrix):
            self = self.hadamard(other)
        else:
            if type(other) == np.complex128 or type(other) == complex:
                self.S = np.asarray(self.S, dtype=np.complex128)
            np.multiply(self.S, other, out=self.S)
        return self

    def __mul__(self, other: float | Matrix) -> Matrix:
        """Scalar multiplication, or hadamard product if other is a matrix"""
        new_mat = self.copy()
        new_mat *= other
        return new_mat
        

    def dot(self,
            other: QuasiSVD | Matrix,
            side: str = 'right',
            dense_output: bool = False) -> SVD | Matrix:
        """Matrix multiplication between SVD and other.
        The output is an SVD or a Matrix, depending on the type of other.
        If two QuasiSVD are multiplied, the new rank is the minimum of the two ranks.

        Parameters
        ----------
        other : QuasiSVD or Matrix
            Matrix to multiply
        side : str, optional
            'left' or 'right', by default 'right'
        dense_output : bool, optional
            If True, return a dense matrix. False by default.

        Returns
        -------
        SVD or Matrix
            Result of the matrix multiplication
        """
        if isinstance(other, QuasiSVD) and dense_output == False:
            if side == 'right' or side == 'usual':
                return QuasiSVD.multi_dot([self, other])
            elif side == 'opposite' or side == 'left':
                return QuasiSVD.multi_dot([other, self])
            else:
                raise ValueError('Incorrect side. Choose "right" or "left".')
        else:
            return super().dot(other, side, dense_output)

    def hadamard(self, other: QuasiSVD | Matrix, truncate: bool = automatic_truncation) -> QuasiSVD | Matrix:   
        """Hadamard product between two QuasiSVD matrices
        
        The new rank is the multiplication of the two ranks, at maximum.
        The default behavior depends on the value of automatic_truncation.
        If automatic_truncation is True, the output is truncated to ensure non-singularity of S.
        If automatic_truncation is False, the output is not truncated and can be singular.
        NOTE: if the rank is too large, dense matrices are used for the computation, but the output is still an SVD.

        Parameters
        ----------
        other : QuasiSVD or Matrix
            Matrix to multiply
        """
        if isinstance(other, QuasiSVD):
            # If the new rank is too large, it is more efficient to use the full matrix
            if self.rank * other.rank >= min(self.shape):
                # print(f"Large rank ({self.rank * other.rank}) due to Hadamard product. Using full matrix ({self.shape}).")
                warnings.warn(f"Large rank due to Hadamard product. Using full matrix.")
                output = np.multiply(self.full(), other.full())
                output = SVD.from_dense(output) # convert to SVD, otherwise it is inconsistent
            else:    
                # The new matrices U and V are obtained from transposed Khatri-Rao products
                new_U = la.khatri_rao(self.Uh, other.Uh).T.conj()
                new_V = la.khatri_rao(self.Vh, other.Vh).T.conj()
                # The new singular values are obtained from the Kronecker product
                new_S = np.kron(self.S, other.S)
                output = QuasiSVD(new_U, new_S, new_V)
                if truncate:
                    output = output.truncate(atol=default_atol)
        elif isinstance(other, LowRankMatrix):
            warnings.warn("Hadamard product between QuasiSVD and LowRankMatrix is not efficient.")
            output = np.multiply(self.full(), other.full())
        else:
            warnings.warn("Hadamard product between QuasiSVD and ndarray is not efficient.")
            output = np.multiply(self.full(), other)
        return output
    
#%% Define class SVD
class SVD(QuasiSVD):
    """
    Singular Value Decomposition (SVD)
    Inherited from QuasiSVD
    X = U @ S @ V.T
    where U, V are orthonormal and S is diagonal
    If X is low-rank, it is much more efficient to store only U, S, V.
    Behaves like a numpy ndarray but preserves the low-rank structure.
    Note that, after multiplication, the orthogonality of U and V is lost in general.
    Therefore, the class provides a method to convert from LowRankMatrix.
    It is also possible to efficiently convert a QuasiSVD into an SVD.
    """

    ## ATTRIBUTES
    _format = "SVD"

    def __init__(self, U: ndarray, s: ndarray, V: ndarray, **extra_data):
        """
        Create a low-rank matrix stored by its SVD: Y = U @ S @ V.T
        Parameters
        ----------
        U : ndarray
            Left singular vectors, shape (m, r)
        s : ndarray
            Singular values, shape (r,)
        V : ndarray
            Right singular vectors, shape (n, r)
        """
        # Sanity check
        assert s.ndim == 1, "s is not a vector"
        super().__init__(U, np.diag(s), V, **extra_data)

    ## SPECIFIC PROPERTIES
    @property
    def sing_vals(self) -> ndarray:
        return np.diag(self.S)
    
    def norm(self, ord: str | int = 'fro') -> float:
        if ord == 'fro':
            return la.norm(self.sing_vals)
        elif ord == 2:
            return self.sing_vals[0]
        else:
            return super().norm(ord)

    
    ## CLASS METHODS
    @classmethod
    def singular_values(cls, X: SVD | Matrix) -> SVD | Matrix:
        """Compute the singular values of X"""
        if isinstance(X, SVD):
            return X.sing_vals
        elif isinstance(X, LowRankMatrix):
            return SVD.from_low_rank_matrix(X).sing_vals
        else:
            return la.svd(X, compute_uv=False)

    @classmethod
    def from_quasiSVD(cls, mat: QuasiSVD) -> SVD:
        """Create an SVD from a QuasiSVD"""
        return mat.to_svd()

    @classmethod
    def from_low_rank(cls, mat: LowRankMatrix, **extra_data) -> SVD:
        """Create a SVD from a LowRankMatrix"""
        # QR decomposition of the first matrix
        Q1, R1 = la.qr(mat._matrices[0], mode='economic')
        # QR decomposition of the last matrix
        Q2, R2 = la.qr(mat._matrices[-1].T.conj(), mode='economic')
        # SVD of the middle matrix
        middle = np.linalg.multi_dot([R1] + mat._matrices[1:-1] + [R2.T.conj()])
        U, s, Vh = la.svd(middle, full_matrices=False)
        # Create the SVD
        U = Q1.dot(U)
        V = Q2.dot(Vh.T.conj())
        return cls(U, s, V, **extra_data)

    @classmethod
    def from_matrix(cls, mat: ndarray | LowRankMatrix, **extra_data) -> SVD:
        """Create an SVD from a matrix"""
        if isinstance(mat, SVD):
            return mat
        elif isinstance(mat, QuasiSVD):
            return mat.to_svd()
        elif isinstance(mat, LowRankMatrix):
            return cls.from_low_rank(mat, **extra_data)
        else:
            return cls.reduced_svd(mat, **extra_data)

    @classmethod
    def full_svd(cls, mat: ndarray, **extra_data) -> SVD:
        """Compute a full SVD"""
        u, s, vh = la.svd(mat, full_matrices=True)
        return SVD(u, s, vh.T.conj(), **extra_data)

    @classmethod
    def reduced_svd(cls, mat: ndarray, **extra_data) -> SVD:
        """Compute a reduced SVD of rank r"""
        u, s, vh = la.svd(mat, full_matrices=False)
        return cls(u, s, vh.T.conj(), **extra_data)

    @classmethod
    def truncated_svd(cls, mat: Matrix, r: int = None, rtol: float = None, atol: float = default_atol, **extra_data) -> SVD:
        """Compute a truncated SVD of rank r

        First compute a reduced SVD, then truncate the singular values.
        
        Parameters
        ----------
        mat : Matrix
            Input matrix
        r : int, optional
            Target rank, by default None
        rtol : float, optional
            Relative tolerance, by default None
        atol : float, optional
            Absolute tolerance, by default machine_precision

        Returns
        -------
        SVD
            Truncated SVD
        """
        X = cls.from_matrix(mat, **extra_data)
        return X.truncate(r=r, rtol=rtol, atol=atol)

    @classmethod
    def generate_random(cls, shape: tuple, sing_vals: ndarray, seed: int = 1234, is_symmetric: bool = False, **extra_data) -> SVD:
        """Generate a random SVD with given singular values.
        
        Parameters
        ----------
        shape : tuple
            Shape of the matrix
        sing_vals : ndarray
            Singular values
        seed : int, optional
            Random seed, by default 1234
        is_symmetric : bool, optional
            Whether the generated matrix is symmetric, by default False

        Returns
        -------
        SVD
            SVD matrix generated randomly
            """
        np.random.seed(seed) # for reproducibility
        (m, n) = shape
        r = len(sing_vals)
        if is_symmetric:
            A = np.random.rand(m, r)
            Q, _ = la.qr(A, mode='economic')
            return SVD(Q, sing_vals, Q, **extra_data)
        else:
            A = np.random.rand(m, r)
            Q1, _ = la.qr(A, mode='economic')
            B = np.random.rand(n, r)
            Q2, _ = la.qr(B, mode='economic')
            return SVD(Q1, sing_vals, Q2, **extra_data)

    def truncate(self,
                 r: int = None,
                 rtol: float = None,
                 atol: float = default_atol,
                 inplace: bool = False) -> SVD:
        """
        Truncate the SVD.
        The rank is prioritized over the tolerance.
        The relative tolerance is prioritized over absolute tolerance. 
        NOTE: By default, the truncation is done with respect to the machine precision (default_atol).

        Parameters
        ----------
        r : int, optional
            Rank, by default None
        rtol : float, optional
            Relative tolerance, by default None.
            Uses the largest singular value as reference.
        atol : float, optional
            Absolute tolerance, by default default_atol.
        inplace : bool, optional
            If True, modify the matrix inplace, by default False
        """
        # If all are None, do nothing
        if r is None and rtol is None and atol is None:
            return self
        
        # Compute the rank associated to the tolerance
        if r is None:
            if rtol is not None:
                r = np.sum(self.sing_vals > self.sing_vals[0] * rtol)
            else:
                r = np.sum(self.sing_vals > atol)
        
        # Truncate
        (m,n) = self.shape
        if r == 0: # trivial case
            U = np.zeros((m, 0))
            s = np.zeros(0)
            V = np.zeros((n, 0))
        else: # general case
            U = self.U[:, :r]
            s = self.sing_vals[:r]
            V = self.V[:, :r]
        if inplace:
            self.U = U
            self.S = np.diag(s)
            self.V = V
            self.r = r
            return self
        else:
            return SVD(U, s, V)

    def hadamard(self, other: Matrix | SVD, truncate: bool = automatic_truncation) -> Matrix | SVD:
        """Faster version of the Hadamard product for SVDs."""
        if isinstance(other, SVD) and not self.rank * other.rank >= min(self.shape):
            # The new matrices U and V are obtained from transposed Khatri-Rao products
            new_U = la.khatri_rao(self.Uh, other.Uh).T.conj()
            new_V = la.khatri_rao(self.Vh, other.Vh).T.conj()
            # The new singular values are obtained from the Kronecker product
            new_S = np.kron(self.sing_vals, other.sing_vals)
            output = SVD(new_U, new_S, new_V)
            if truncate:
                output = output.truncate(atol=default_atol)
        else:
            output = super().hadamard(other)
        return output