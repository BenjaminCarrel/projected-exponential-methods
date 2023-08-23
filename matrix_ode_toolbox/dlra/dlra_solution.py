"""
Author: Benjamin Carrel, University of Geneva, 2023

This file contains the class DlraSolution, which is used to store the solution of the DLRA.
"""

#%% Imports
from numpy import ndarray
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import MatrixOdeSolution


#%% Class DlraSolution
class DlraSolution(MatrixOdeSolution):
    """
    Class for the solution of the DLRA.
    It is a subclass of the MatrixOdeSolution class.
    It stores the solution of the DLRA in low-rank format, and additional informations such as the time grid points, the time of computation, etc.
    It can be extended to add visualization methods, manifold interpolation, etc.
    """

    # Redefine the Xs to Ys
    def __init__(self,
                matrix_ode: MatrixOde,
                ts: ndarray,
                Ys: ndarray,
                computation_time: ndarray = None):
            """
            Store the solution of the DLRA.
    
            Parameters
            ----------
            matrix_ode : MatrixOde
                The matrix ODE.
            ts : ndarray
                The time grid points.
            Ys : ndarray
                The solution at the time grid points.
            computation_time : ndarray, optional
                The computation time at each time grid point, by default None.
            """
            super().__init__(matrix_ode, ts, Ys, computation_time)
            self.Ys = Ys