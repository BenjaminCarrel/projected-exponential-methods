"""
Author: Benjamin Carrel, University of Geneva, 2022

Lyapunov ODE structure. Subclass of SylvesterOde.
"""


# %% IMPORTATIONS
from __future__ import annotations
from scipy.sparse import spmatrix
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from .sylvester_ode import SylvesterOde
from .matrix_ode import MatrixOde

Matrix = ndarray | spmatrix | LowRankMatrix

# %% CLASS SYLVESTER
class LyapunovOde(SylvesterOde):
    """
    Subclass of SylvesterOde. Specific to the Lyapunov equation.
    The class is a essentially a wrapper for the SylvesterOde class.

    Lyapunov differential equation : X'(t) = A X(t) + X(t) A + C.
    Initial value given by X(t_0) = X0.
    
    Typically, A is a sparse matrix, and C is a low-rank, symmetric matrix.
    """

    #%% ATTRIBUTES
    A = MatrixOde.create_parameter_alias(0)
    B = A
    C = MatrixOde.create_parameter_alias(2)

    #%% FUNDAMENTALS
    def __init__(self, A: Matrix, C: LowRankMatrix, **kwargs):
        """Lyapunov differential equation: X'(t) = A X(t) + X(t) A + C."""
        # Check inputs
        assert isinstance(A, Matrix), "A must be a sparse matrix"
        assert isinstance(C, LowRankMatrix), "C must be a LowRankMatrix"
        assert C.is_symmetric(), "C must be symmetric"

        # INITIALIZATION
        super().__init__(A, A, C, **kwargs)

    @property
    def name(self) -> str:
        """Return the name of the ODE."""
        if self.ode_type == 'F':
            return 'Lyapunov'
        else:
            return 'Sylvester'

    def copy(self):
        return LyapunovOde(self.A, self.C, ode_type=self.ode_type, mats_uv=self.mats_uv)



