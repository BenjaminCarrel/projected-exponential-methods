"""
Author: Benjamin Carrel, University of Geneva, 2022

Exponential Runge-Kutta methods for solving matrix ODEs.
"""

# Imports
import numpy as np
from numpy import ndarray
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolver
from low_rank_toolbox import LowRankMatrix
from scipy.integrate import solve_ivp

Matrix = LowRankMatrix | ndarray

#%% Class for exponential Runge-Kutta methods
class ExponentialRungeKutta(MatrixOdeSolver):
    """
    Exponential Runge-Kutta methods for solving matrix ODEs.

    Applied only to ODEs of the form X' = L(X) + G(t, X)
    where L is linear (typically stiff) and G is non linear.
    """

    name = 'Exponential Runge-Kutta'

    def __init__(self, 
                 matrix_ode: MatrixOde,
                 nb_substeps: int = 1,
                 order: int = 2):
        super().__init__(matrix_ode, nb_substeps)
        self.order = order

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Exponential Runge-Kutta \n'
        info += f'-- {self.order} stage(s) \n'
        info += f'-- {self.nb_substeps} substep(s)'
        return info

    def solve(self, t_span: tuple, X0: Matrix):
        "Overload of the solve method to perform the substepping"
        # VARIABLES
        t0, t1 = t_span
        ts = np.linspace(t0, t1, self.nb_substeps + 1, endpoint=True)
        X = X0
        # LOOP
        for i in np.arange(self.nb_substeps):
            X = self.stepper(ts[i:i+2], X)
        return X

    def stepper(self, t_span: tuple, X0: Matrix):
        """
        This method is a wrapper around the correct order stepper.
        """
        if self.order == 1:
            return self.stepper_1(t_span, X0)
        elif self.order == 2:
            return self.stepper_2(t_span, X0)
        else:
            raise ValueError(f'Order {self.order} not implemented.')

    def solve_equivalent_ode(self, t_span: tuple, X0: Matrix, GX: Matrix):
        """
        The equivalent ode is
        X' = L(t, X) + G(t, X), X(0) = X0
        It is useful for order 1 and order 2 methods.
        """
        # Create the equivalent ode
        def equiv_ode(t: float, x: ndarray):
            X = x.reshape(X0.shape)
            return (self.matrix_ode.linear_field(t, X) + GX).flatten()
        
        # Solve it
        x1 = solve_ivp(fun=equiv_ode, t_span=t_span, y0=X0.flatten(), dense_output=True).y[:, -1]
        X1 = x1.reshape(X0.shape)
        return X1


    def stepper_1(self, t_span: tuple, X0: Matrix):
        """
        This stepper implements the first order exponential Runge-Kutta method.
        If the matrix ODE is of the form X' = L(X) + G(t, X), then this method is
        X1 = exp(h L) X0 + h phi1(h L) G(0, X0)
        which is equivalent to solve the ivp
        X' = L(X) + G(0, X_0), X(0) = X0
        on the time domain [0, h].
        """
        h = t_span[1] - t_span[0]
        h_span = (0, h)
        GX0 = self.matrix_ode.non_linear_field(t_span[0], X0)
        X1 = self.solve_equivalent_ode(h_span, X0, GX=GX0)
        return X1

    def stepper_2(self, t_span: tuple, X0: Matrix):
        """
        This stepper implements the second order exponential Runge-Kutta method.
        """
        h_half = (t_span[1] - t_span[0]) / 2
        t_subspan = (t_span[0], t_span[0] + h_half)
        GX0 = self.matrix_ode.non_linear_field(t_span[0], X0)
        X_half = self.solve_equivalent_ode(t_subspan, X0, GX=GX0)
        GX_half = self.matrix_ode.non_linear_field(t_span[0] + h_half, X_half)
        X1 = self.solve_equivalent_ode(t_span, X0, GX=GX_half)
        return X1
    



