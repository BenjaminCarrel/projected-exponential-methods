"""
Author: Benjamin Carrel, University of Geneva, 2022

Adaptive projected exponential Runge-Kutta methods for solving Sylvester-like matrix differential equations.
"""

# %% Imports
from matrix_ode_toolbox.dlra.adaptive_dlra_solver import AdaptiveDlraSolver
from matrix_ode_toolbox import SylvesterLikeOde
from .. import ProjectedExponentialRungeKutta
from low_rank_toolbox import LowRankMatrix, SVD
from numpy import ndarray

Matrix = ndarray | LowRankMatrix


#%% Class Adaptive Projected Exponential Runge-Kutta
class AdaptiveProjectedExponentialMethods(AdaptiveDlraSolver):
    """
    Adaptive Projected exponential Runge-Kutta methods.
    The matrix ODE must be of the form
    X' = AX + XB + G(t, X)
    See Carrel & Vandereycken, 2023.
    """

    name = 'Adaptive PERK'

    def __init__(self, 
                 matrix_ode: SylvesterLikeOde, 
                 nb_substeps: int = 1, 
                 order: int = 1,
                 krylov_kwargs: dict = {},
                 strict_order_conditions: bool = True,
                 rtol: float = 1e-8, 
                 atol: float = 1e-8,
                 **extra_args) -> None:
        # Initialization
        super().__init__(matrix_ode, nb_substeps, rtol, atol)
        self.order = order
        self.strict_order_conditions = strict_order_conditions

        # Wrapper around projected exponential Runge-Kutta...
        solver = ProjectedExponentialRungeKutta(matrix_ode, nb_substeps, order, krylov_kwargs, strict_order_conditions, **extra_args)
        # ... but with modified retraction method
        def adaptive_retraction(X: SVD) -> SVD:
            return X.truncate(rtol=rtol, atol=atol)
        solver.retraction = adaptive_retraction
        # Overwrite the stepper
        self.stepper = solver.stepper
        # Extract some data
        self.krylov_size = solver.krylov_size
        self.krylov_kind = solver.krylov_kind
        self.size_factor = solver.size_factor

    def info(self) -> str:
        """Return the info string."""
        info = f'Adaptive PERK \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- {self.order} stages \n'
        info += f'-- Strict order conditions: {self.strict_order_conditions} \n'
        info += f'-- {self.krylov_kind} Krylov space and {self.krylov_size} iteration(s) \n'
        info += f'-- Largest matrix shape: ({2 * self.size_factor * self.krylov_size * self.order}*rank, {2 * self.size_factor * self.krylov_size * self.order}*rank) \n'
        info += f'-- Relative tolerance = {self.rtol} \n'
        info += f'-- Absolute tolerance = {self.atol}'
        return info

