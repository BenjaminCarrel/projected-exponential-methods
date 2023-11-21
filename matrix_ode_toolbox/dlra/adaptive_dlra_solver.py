'''
Author: Benjamin Carrel, University of Geneva, 2022

General class for implementing adaptive DLRA solvers.
'''

#%% Imports
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox import MatrixOde
from .dlra_solver import DlraSolver

Matrix = ndarray | LowRankMatrix


#%% Common class for the DLRA solvers
class AdaptiveDlraSolver(DlraSolver):
    ''''
    Class for the adaptive DLRA solvers. It inherits from the DlraSolver class.

    How to implement a new adaptive DLRA method:
    1. Create a new class that inherits from AdaptiveDlraSolver
    2. Create a specific init method that takes the necessary parameters
    3. Implement the method stepper
    Well done! You can now use your method in the solve_adaptive_dlra function.
    '''

    #%% ATTRIBUTES
    name = 'Adaptive DLRA - General'

    #%% Static methods
    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int = 1, rtol: float = 1e-8, atol: float = 1e-8):
        "Initialize the solver."
        self.matrix_ode = matrix_ode.copy()
        self.nb_substeps = nb_substeps
        self.rtol = rtol
        self.atol = atol

    @property
    def info(self) -> str:
        """Return the info string."""
        info = f'General adaptive DLRA \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- Relative tolerance: {self.rtol} \n'
        info += f'-- Absolute tolerance: {self.atol}'
        return info

