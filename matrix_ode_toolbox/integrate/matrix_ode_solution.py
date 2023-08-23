"""
Author: Benjamin Carrel, University of Geneva, 2023

This file contains the class MatrixOdeSolution, which is used to store the solution of a matrix ODE.
"""

#%% Imports
import numpy as np
from numpy import ndarray
from matrix_ode_toolbox import MatrixOde
from low_rank_toolbox import LowRankMatrix

Matrix = ndarray | LowRankMatrix


#%% Class MatrixOdeSolution
class MatrixOdeSolution:
    """
    Class used to store the solution of a matrix ODE.
    It contains additional informations like the time grid points, the time of computation, etc.
    """

    def __init__(self,
                matrix_ode: MatrixOde,
                ts: ndarray,
                Xs: ndarray,
                computation_time: ndarray = None):
        """
        Store the solution of a matrix ODE.

        Parameters
        ----------
        matrix_ode : MatrixOde
            The matrix ODE.
        ts : ndarray
            The time grid points.
        Xs : ndarray
            The solution at the time grid points.
        computation_time : ndarray, optional
            The computation time at each time grid point, by default None.
        """
        self.matrix_ode = matrix_ode
        self.ts = ts
        self.t0 = ts[0]
        self.tf = ts[-1]
        self.t1 = ts[-1]
        self.X0 = Xs[0]
        self.X1 = Xs[-1]
        self.Xf = Xs[-1]
        self.Xs = Xs
        self.computation_time = computation_time

    def __getitem__(self, key):
        "Return the solution at the time grid point corresponding to the key."
        return self.Xs[self.ts == key]

    def __len__(self):
        return len(self.ts)

    def __str__(self):
        return ('Solution of a matrix ODE. \n' +
        f'ODE structure: {self.matrix_ode} \n' + 
        f'Time grid: t0={self.ts[0]} and t1={self.ts[-1]} with {self.ts.__len__()} grid points. \n' +
        f'Total computation time: {np.sum(self.computation_time)}.')

    def __repr__(self):
        return self.__str__()

    def todense(self):
        "Convert the solution to dense format."
        # convert X to dense iff X is a LowRankMatrix
        return np.array([X.todense() if isinstance(X, LowRankMatrix) else X for X in self.Xs])

    def visualise(self, meshgrid: tuple = None, **kwargs):
        """
        Visualise the solution over time

        Parameters
        ----------
        meshgrid : tuple, optional
            The meshgrid used to plot the solution, by default None.
        kwargs : dict
            Additional arguments to pass to the plot
        """
        return NotImplementedError('Not implemented yet. Implement it if necessary. You may want to add a plot function to the class MatrixOde, since the plot is specific to the ODE.')
