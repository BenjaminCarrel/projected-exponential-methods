"""
Author: Benjamin Carrel, University of Geneva, 2022

Riccati differential equation structure. Subclass of MatrixODE.
"""


# %% IMPORTATIONS
import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
from typing import Union

Matrix = ndarray | LowRankMatrix | spmatrix

# %% CLASS RICCATI
class RiccatiOde(MatrixOde):
    r"""Subclass of MatrixOde. Specific to the Riccati differential equation.

    Riccati differential equation : Q'(t) = A^T Q(t) + Q(t) A + C - Q(t) D Q(t)
    
    A is a sparse matrix
    C and D are symmetric, low-rank matrices
    """
    # ATTRIBUTES
    name = 'Riccati'
    A = MatrixOde.create_parameter_alias(0)
    B = A
    C = MatrixOde.create_parameter_alias(1)
    D = MatrixOde.create_parameter_alias(2)

    # %% INIT FUNCTION
    def __init__(self, A: spmatrix, C: LowRankMatrix, D: Matrix):
        # Check inputs
        assert isinstance(A, spmatrix), "A must be a sparse matrix"
        assert isinstance(C, LowRankMatrix), "C must be a low-rank matrix"
        assert isinstance(D, Matrix), "D must be a matrix"

        # Convert to csc format
        A = A.tocsc()
        if isinstance(D, spmatrix):
            D = D.tocsc()

        super().__init__(A, C, D)
        

    # %% PREPROCESS ODE
    def preprocess_ode(self):
        A, C, D = self.A, self.C, self.D
        ode_type, mats_uv = self.ode_type, self.mats_uv
        # PRE-COMPUTATIONS
        if ode_type == "F":
            self.Ar = A.T
            self.Br = A
            self.Cr = C
            self.Dr = D
        elif ode_type == "K":
            (V,) = mats_uv
            self.Ar = A.T
            self.Br = V.T.dot(A.dot(V))
            self.Cr = C.dot(V, dense_output=True)
            self.Dr = D.T.dot(V).T
        elif ode_type == "L":
            (U,) = mats_uv
            self.Ar = A
            self.Br = U.T.dot(A.T.dot(U))
            self.Cr = C.dot(U, dense_output=True)
            self.Dr = D.T.dot(U).T
        elif ode_type == 'S':
            U, V = mats_uv
            self.Ar = U.T.dot(A.T.dot(U))
            self.Br = V.T.dot(A.dot(V))
            self.Cr = C.dot(U.T, side='left').dot(V, dense_output=True)
            self.Dr = V.T.dot(D.dot(U))
        elif ode_type == 'minus_S':
            U, V = mats_uv
            self.Ar = -U.T.dot(A.T.dot(U))
            self.Br = -V.T.dot(A.dot(V))
            self.Cr = -C.dot(U.T, side='left').dot(V, dense_output=True)
            self.Dr = -V.T.dot(D.dot(U))
        else:
            raise ValueError("ode_type must be 'F', 'K', 'L', 'S' or 'minus_S'")
        

    # %% VECTOR FIELDS
    def ode(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            dX = X.dot(self.Ar.T, side='left') + X.dot(self.Br) + self.Cr - X.dot(X.dot(self.Dr, side='left'))
        else:
            if isinstance(self.Dr, LowRankMatrix):
                dX = self.Cr + self.Ar.T.dot(X) + self.Br.T.dot(X.T).T - X.dot(self.Dr.dot(X, dense_output=True))
            else:
                dX = self.Cr + self.Ar.T.dot(X) + self.Br.T.dot(X.T).T - X.dot(self.Dr.dot(X))
        return dX
    
    def ode_F(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            dX = X.dot(self.A.T, side='left') + X.dot(self.B) + self.C - X.dot(X.dot(self.D, side='left'))
        else:
            dX = self.C + self.A.T.dot(X) + self.B.T.dot(X.T).T - X.dot(self.D.dot(X, dense_output=True))
        return dX

    def linear_field(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            dX = X.dot(self.Ar.T, side='left') + X.dot(self.Br)
        else:
            dX = self.Ar.T.dot(X) + self.Br.T.dot(X.T).T
        return dX

    def non_linear_field(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            dX = self.Cr - X.dot(X.dot(self.Dr, side='left'))
        else:
            dX = self.Cr - X.dot(self.Dr.dot(X))
        return dX
