'''
Author: Benjamin Carrel, University of Geneva, 2022

General class for implementing DLRA solvers.
'''

#%% Imports
import numpy as np
from numpy import ndarray
from low_rank_toolbox import LowRankMatrix
from matrix_ode_toolbox import MatrixOde


Matrix = ndarray | LowRankMatrix


#%% Common class for the DLRA solvers
class DlraSolver:
    ''''
    Class for the DLRA solvers

    How to implement a new DLRA method:
    1. Create a new class that inherits from DlraSolver
    2. Create a specific init method that takes the necessary parameters
    3. Implement the method stepper
    Well done! You can now use your method in the solve_dlra function.
    '''

    name = 'Generic DLRA'

    #%% Static methods
    def __init__(self, matrix_ode: MatrixOde, nb_substeps: int = 1):
        "Initialize the solver."
        self.matrix_ode = matrix_ode.copy()
        self.nb_substeps = nb_substeps

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Generic DLRA \n'
        info += f'{self.nb_substeps} substep(s)'
        return info

    def __repr__(self) -> str:
        return self.info

    def solve(self, t_span: tuple, Y0: LowRankMatrix):
        "Solve the DLRA by calling the stepper method."
        # Initialization
        t0, tf = t_span
        ts = np.linspace(t0, tf, self.nb_substeps + 1, endpoint=True)
        Y = Y0
        # Loop over the substeps
        for i in np.arange(self.nb_substeps):
            previous_rank = Y.rank
            Y = self.stepper(ts[i:i+2], Y)
            # Check if the rank has changed
            if Y.rank != previous_rank:
                print(f'Rank has changed from {previous_rank} to {Y.rank} at t = {ts[i+1]}')
        return Y

    #%% Methods to be overloaded
    def stepper(self, t_subspan: tuple, Y0: LowRankMatrix):
        "Perform one step of the DLRA."
        return NotImplementedError('The stepper method is not implemented. Overload it in the child class.')

