"""
Author: Benjamin Carrel, University of Geneva, 2022

This file contains the class MatrixOdeSolver, which is a general class for solving matrix ODEs
"""

#%% Imports
import numpy as np
from numpy import ndarray
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox import MatrixOde

Matrix = ndarray | LowRankMatrix


#%% Class MatrixOdeSolver
class MatrixOdeSolver:
    """
    Matrix ODE solver class.

    How to define a new solver:
    1. Define a new class that inherits from MatrixOdeSolver.
    2. Overload the __init__ method to add the necessary arguments.
    3. Overload the stepper method to define the solver.

    See the documentation for more details. See ExplicitRungeKutta for an example (path: matrix_ode_toolbox/integrate/explicit_runge_kutta.py).
    """

    #%% ATTRIBUTES
    name: str = "General"

    #%% Static methods
    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int):
        """
        Initialize the matrix solver.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE to solve
        nb_substeps : int
            The number of substeps to use in solve()
        """
        self.matrix_ode = matrix_ode
        self.nb_substeps = nb_substeps

    @property
    def info(self) -> str:
        "Return the info string."
        info = f"{self.name} \n"
        info += f"-- {self.nb_substeps} substep(s)"
        return info

    def __repr__(self):
        return f"{self.name} solver for {self.matrix_ode}"

    def solve(self, t_span: tuple, X0: Matrix):
        """
        Solve the matrix ODE from t0 to t1, with initial value X0.
        Applies the stepper method nb_substeps times, and returns the final value.
        """
        # VARIABLES
        t0, t1 = t_span
        ts = np.linspace(t0, t1, self.nb_substeps + 1, endpoint=True)
        X = X0
        # LOOP
        for i in np.arange(self.nb_substeps):
            X = self.stepper(ts[i:i+2], X)
        return X

    #%% Methods to be overloaded
    def stepper(self, t_span: tuple, X0: Matrix) -> Matrix:
        """
        Compute the next step of the solution.
        Overload this method to define a new solver.

        Parameters
        ----------
        t_span : tuple
            The time interval (t0, t1)
        X0 : Matrix
            The initial value at t0
        extra_args : dict
            Extra arguments

        Returns
        -------
        Matrix
            The solution at t1
        """
        raise NotImplementedError(
            "Stepper is not implemented for the generic solver. Stepper must be overloaded."
        )

