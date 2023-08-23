"""
Author: Benjamin Carrel, University of Geneva, 2022

Sylvester ODE structure. Subclass of MatrixOde.
"""


# %% IMPORTATIONS
from __future__ import annotations
from matrix_ode_toolbox.structures.matrix_ode import Matrix
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
from typing import Callable

Matrix = ndarray | spmatrix | LowRankMatrix

# %% CLASS SYLVESTER
class SylvesterOde(MatrixOde):
    """
    Subclass of MatrixOde. Specific to the Sylvester equation.

    Sylvester differential equation : X'(t) = A X(t) + X(t) B + C.
    Initial value given by X(t_0) = X0.
    
    Typically, A and B are sparse matrices, and C is a low-rank matrix.
    """

    #%% ATTRIBUTES
    name = 'Sylvester'
    A = MatrixOde.create_parameter_alias(0)
    B = MatrixOde.create_parameter_alias(1)
    C = MatrixOde.create_parameter_alias(2)

    #%% FUNDAMENTALS
    def __init__(self, A: Matrix, B: Matrix, C: LowRankMatrix, **kwargs):
        """Sylvester differential equation: X'(t) = A X(t) + X(t) B + C."""
        # Check inputs
        assert isinstance(A, Matrix), "A must be a sparse matrix"
        assert isinstance(B, Matrix), "B must be a sparse matrix"
        assert isinstance(C, LowRankMatrix), "C must be a LowRankMatrix"

        # INITIALIZATION
        super().__init__(A, B, C, **kwargs)

    def preprocess_ode(self):
        "Preprocess the ODE -> compute the factors of the selected ODE"
        super().preprocess_ode()
        A, B, C = self.A, self.B, self.C
        ode_type, mats_uv = self.ode_type, self.mats_uv
        if ode_type == "F":
            self.Ar, self.Br, self.Cr = A, B, C
            return self
        elif ode_type == "K":
            (V,) = mats_uv
            self.Ar = A
            self.Br = V.T.dot(B.dot(V))
            self.Cr = C.dot(V).todense()
        elif ode_type == "L": # don't forget: A and B are switched due to the transpose
            (U,) = mats_uv
            self.Ar = B
            self.Br = U.T.dot(A.dot(U))
            self.Cr = C.dot(U).todense()
        elif ode_type == "S" or ode_type == "minus_S":
            (U, V) = mats_uv
            self.Ar = U.T.dot(A.dot(U))
            self.Br = V.T.dot(B.dot(V))
            self.Cr = C.dot(V).dot(U.T, side='opposite').todense()



    #%% VECTOR FIELDS
    def ode_F(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            dY = X.dot(self.A, side='left') + X.dot(self.B, side='right') + self.C
        else:
            dY = self.C + self.A.dot(X) + self.B.T.dot(X.T).T
        return dY
    
    def ode_K(self, t: float, K: Matrix) -> ndarray:
        return self.Ar.dot(K) + K.dot(self.Br.T) + self.Cr
    
    def ode_S(self, t: float, S: Matrix) -> ndarray:
        return self.Ar.dot(S) + S.dot(self.Br) + self.Cr
    
    def ode_L(self, t: float, L: Matrix) -> ndarray:
        return self.Ar.dot(L) + L.dot(self.Br.T) + self.Cr

    def linear_field(self, t: float, X: Matrix) -> Matrix:
        "Linear field of the equation"
        if isinstance(X, LowRankMatrix):
            dY = X.dot(self.Ar, side='left') + X.dot(self.Br, side='right')
        else:
            dY = self.Ar.dot(X) + self.Br.T.dot(X.T).T
        return dY

    def non_linear_field(self, t: float, X: Matrix) -> Matrix:
        "Non linear field of the equation"
        return self.Cr
