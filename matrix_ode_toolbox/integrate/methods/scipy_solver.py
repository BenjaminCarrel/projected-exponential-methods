# Author: Benjamin Carrel
#         University of Geneva, 2022

# This file contains the class MatrixScipySolver, which is a wrapper around the scipy solver.

# Imports
from scipy.integrate import solve_ivp
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolver
from low_rank_toolbox import LowRankMatrix
from numpy import ndarray

Matrix = ndarray | LowRankMatrix

# Class
class ScipySolver(MatrixOdeSolver):
    """
    Scipy solver. This is a wrapper around the scipy solver solve_ivp.
    """

    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int = 1, **solve_ivp_args):
        super().__init__(matrix_ode, nb_substeps)
        # Get the method and remove it from the arguments
        self.scipy_method = solve_ivp_args.pop('scipy_method', 'RK45') # Default method for high precision
        # self.scipy_method = solve_ivp_args.pop('scipy_method', 'Radau') # Default method for stiff problems
        # self.scipy_method = solve_ivp_args.pop('scipy_method', 'LSODA') # Automatic detection of stiffness, seems to be the best
        # Get the atol and rtol
        self.atol = solve_ivp_args.pop('atol', 1e-12)
        self.rtol = solve_ivp_args.pop('rtol', 1e-12)
        # Store other arguments
        self.solve_ivp_args = solve_ivp_args
        self.name = f'Scipy [{self.scipy_method} - {nb_substeps} substeps]'

    @property
    def info(self) -> str:
        info = f'Scipy solver \n'
        info += f'-- {self.scipy_method} method \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- Relative tolerance: {self.rtol} \n'
        info += f'-- Absolute tolerance: {self.atol}'
        return info

    def stepper(self, t_span: tuple, X0: Matrix):
        """
        This stepper is a wrapper around the scipy solver solve_ivp.

        Parameters
        ----------
        t_span : tuple
            The time span
        initial_value : Matrix
            The initial value
        solve_ivp_args : dict
            Extra arguments for solve_ivp
        """
        # Flatten the initial value
        shape = X0.shape
        x0 = X0.flatten()

        # SOLVE
        sol = solve_ivp(self.matrix_ode.vec_ode,
                        t_span=t_span, 
                        y0=x0, 
                        method=self.scipy_method, 
                        dense_output=True,
                        rtol=self.rtol,
                        atol=self.atol,
                        **self.solve_ivp_args,
                         args = (shape,))
        x1 = sol.y[:, -1]
        return x1.reshape(shape)



