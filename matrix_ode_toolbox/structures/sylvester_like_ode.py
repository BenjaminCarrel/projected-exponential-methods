"""
Author: Benjamin Carrel, University of Geneva, 2023

Sylvester-like ODE structure. Subclass of MatrixOde.
"""

# %% IMPORTATIONS
from __future__ import annotations
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .matrix_ode import MatrixOde
from typing import Callable

Matrix = ndarray | spmatrix | LowRankMatrix

# %% CLASS SYLVESTER-LIKE
class SylvesterLikeOde(MatrixOde):
    """
    Class for Sylvester-like equations. Subclass of MatrixOde.

    Sylvester-like differential equation : 
    X'(t) = A X(t) + X(t) B + G(t, X(t)).
    Initial value given by X(t_0) = X0.

    Typically, A and B are sparse matrices, and G is a non-linear function.

    The linear field is assumed to be stiff, and the non-linear field is assumed to be non-stiff. To change this, edit the stiff_field and non_stiff_field methods.
    """

    #%% ATTRIBUTES
    name = 'Sylvester-like'
    A = MatrixOde.create_parameter_alias(0)
    B = MatrixOde.create_parameter_alias(1)
    G = MatrixOde.create_parameter_alias(2)

    def __init__(self, A: Matrix, B: Matrix, G: Callable, **kwargs):
        """Sylvester-like differential equation: X'(t) = A X(t) + X(t) B + G(X(t))."""
        # Check inputs
        assert isinstance(A, Matrix), "A must be a sparse matrix"
        assert isinstance(B, Matrix), "B must be a sparse matrix"
        assert callable(G), "G must be a function"

        # INITIALIZATION
        super().__init__(A, B, G, **kwargs)

    def ode_F(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            return X.dot(self.A, side='opposite') + X.dot(self.B) + self.G(t, X)
        else:
            return self.G(t, X) + self.A.dot(X) + self.B.T.dot(X.T).T
    
    def preprocess_ode(self):
        "Preprocess the ODE -> compute the factors of the selected ODE"
        super().preprocess_ode()
        A, B, G = self.A, self.B, self.G
        ode_type, mats_uv = self.ode_type, self.mats_uv
        if ode_type == "F":
            self.Ar, self.Br, self.Gr = A, B, G
            return self
        elif ode_type == "K":
            (V,) = mats_uv
            self.Ar = A
            self.Br = V.T.dot(B.dot(V))
            self.Gr = lambda t, X: G(t, X.dot(V.T)).dot(V)
        elif ode_type == "L": # don't forget: A and B are switched due to the transpose
            (U,) = mats_uv
            self.Ar = B
            self.Br = U.T.dot(A.dot(U))
            self.Gr = lambda t, X: G(t, X.dot(U.T)).dot(U)
        elif ode_type == "S":
            (U, V) = mats_uv
            self.Ar = U.T.dot(A.dot(U))
            self.Br = V.T.dot(B.dot(V))
            self.Gr = lambda t, X: U.T.dot(G(t, U.dot(X.dot(V.T))).dot(V))
        elif ode_type == "minus_S":
            (U, V) = mats_uv
            self.Ar = - U.T.dot(A.dot(U))
            self.Br = - V.T.dot(B.dot(V))
            self.Gr = lambda t, X: - U.T.dot(G(t, U.dot(X.dot(V.T))).dot(V))

    def linear_field(self, t: float, X: Matrix) -> Matrix:
        if isinstance(X, LowRankMatrix):
            return X.dot(self.A, side='left') + X.dot(self.B, side='right')
        else:
            return self.Ar.dot(X) + self.Br.T.dot(X.T).T

    def non_linear_field(self, t: float, X: Matrix) -> Matrix:
        return self.Gr(t, X)

    def stiff_field(self, t: float, X: Matrix) -> Matrix:
        return self.linear_field(t, X)

    def non_stiff_field(self, t: float, Y: Matrix) -> Matrix:
        return self.non_linear_field(t, Y)
    

    