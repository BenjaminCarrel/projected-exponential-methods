"""
Author: Benjamin Carrel, University of Geneva, 2022

Unconventional method(s) for the DLRA.
See Ceruti and Lubich, 2020.
"""

# %% Imports
from low_rank_toolbox import QuasiSVD
import scipy.linalg as la
from matrix_ode_toolbox.dlra import DlraSolver
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp

#%% Class unconventional
class Unconventional(DlraSolver):
    """
    Class for the unconventional DLRA method.
    See Ceruti and Lubich, 2020.
    """

    name = 'Unconventional'
    
    def __init__(self, 
                matrix_ode: MatrixOde,
                nb_substeps: int = 1, 
                substep_kwargs: dict = {'solver': 'automatic', 'nb_substeps': 1},
                **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.substep_kwargs = substep_kwargs

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Unconventional (Ceruti & Lubich 2020) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver"
        return info

    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        """
        Unconventional DLRA method.
        """
        # CHECK INPUTS
        assert len(t_subspan) == 2, 't_subspan must be a tuple of length 2.'
        assert isinstance(Y0, QuasiSVD), 'Y0 must be a QuasiSVD (or SVD).'

        # INITIALISATION
        U0, S0, V0 = Y0.U, Y0.S, Y0.V
        problem: MatrixOde = self.matrix_ode

        # K-STEP
        K0 = U0.dot(S0)
        problem.select_ode('K', (V0,))
        K1 = solve_matrix_ivp(problem, t_subspan, K0, **self.substep_kwargs)
        U1, _ = la.qr(K1, mode='economic')
        M = U1.T.conj().dot(U0)

        # L-STEP
        L0 = V0.dot(S0.T.conj())
        problem.select_ode('L', (U0,))
        L1 = solve_matrix_ivp(problem, t_subspan, L0, **self.substep_kwargs)
        V1, _ = la.qr(L1, mode='economic')
        N = V1.T.conj().dot(V0)

        # S-STEP
        S0 = M.dot(S0.dot(N.T.conj()))
        problem.select_ode('S', (U1, V1))
        S1 = solve_matrix_ivp(problem, t_subspan, S0, **self.substep_kwargs)

        # SOLUTION
        return QuasiSVD(U1, S1, V1)




