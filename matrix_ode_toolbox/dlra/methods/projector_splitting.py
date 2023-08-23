"""
Author: Benjamin Carrel, University of Geneva, 2022

Projector splitting based methods.
See Lubich & Oseledets, 2013.
"""

#%% Imports
import scipy.linalg as la
from low_rank_toolbox import QuasiSVD
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox import MatrixOde
from matrix_ode_toolbox.dlra import DlraSolver

#%% Projector-splitting based methods
class ProjectorSplitting(DlraSolver):
    """
    Projector-splitting (KSL) based methods for the DLRA.
    See Lubich & Osedelets 2013.
    """

    name = 'Projector splitting (KSL)'


    def __init__(self, 
                matrix_ode: MatrixOde, 
                nb_substeps: int = 1,
                order: int = 2,
                substep_kwargs: dict = {'solver': 'automatic', 'nb_substeps': 1},
                **extra_args) -> None:
        super().__init__(matrix_ode, nb_substeps)
        self.order = order
        self.substep_kwargs = substep_kwargs
        if order == 1:
            self.splitting_name = 'Lie-Trotter'
        elif order == 2:
            self.splitting_name = 'Strang'
        else:
            raise ValueError("order must be 1 or 2.")

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Projector splitting (KSL) (Lubich & Oseledets 2013) \n'
        info += f'-- Order {self.order} splitting ({self.splitting_name}) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f"-- {self.substep_kwargs['solver']} as substep solver"
        return info


    def stepper(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        # Check inputs
        assert len(t_subspan) == 2, "t_subspan must be a tuple of length 2."
        assert isinstance(Y0, QuasiSVD), "Y0 must be a QuasiSVD (or QuasiSVD)."
        if self.order == 1:
            return self.KSL1(t_subspan, Y0)
        elif self.order == 2:
            return self.KSL2(t_subspan, Y0)

    def KSL1(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        """
        First-order KSL method.
        """
        # INITIALISATION
        U0, S0, V0 = Y0.U, Y0.S, Y0.V
        problem: MatrixOde = self.matrix_ode
        # K-STEP
        K0 = U0.dot(S0)
        problem.select_ode('K', mats_uv=(V0,))
        K1 = solve_matrix_ivp(problem, t_subspan, K0, **self.substep_kwargs)
        U1, S_hat = la.qr(K1, mode='economic')
        # S-STEP
        problem.select_ode('minus_S', mats_uv=(U1, V0))
        S0_tilde = solve_matrix_ivp(problem, t_subspan, S_hat, **self.substep_kwargs)
        # L-STEP
        L0 = V0.dot(S0_tilde.T.conj())
        problem.select_ode('L', mats_uv=(U1,))
        L1 = solve_matrix_ivp(problem, t_subspan, L0, **self.substep_kwargs)
        V1, S1h = la.qr(L1, mode='economic')
        S1 = S1h.T.conj()
        # SOLUTION
        Y1 = QuasiSVD(U1, S1, V1)
        return Y1

    def KSL2(self, t_subspan: tuple, Y0: QuasiSVD) -> QuasiSVD:
        """
        Second-order KSL method.
        """
        # INITIALISATION
        U0, S0, V0 = Y0.U, Y0.S, Y0.V
        problem = self.matrix_ode
        (t0, t1) = t_subspan
        t_half = (t1 - t0) / 2
        # FORWARD PASS
        K0 = U0.dot(S0)
        problem.select_ode('K', mats_uv=(V0,))
        K1 = solve_matrix_ivp(problem, (t0, t0+t_half), K0, **self.substep_kwargs)
        U1, S_hat = la.qr(K1, mode='economic')
        problem.select_ode('minus_S', mats_uv=(U1, V0))
        S1 = solve_matrix_ivp(problem, (t0, t0+t_half), S_hat, **self.substep_kwargs)
        ## DOUBLE (III)
        L0 = V0.dot(S1.T.conj())
        problem.select_ode('L', mats_uv=(U1,))
        L2 = solve_matrix_ivp(problem, (t0, t1), L0, **self.substep_kwargs)
        V2, Sh_hat = la.qr(L2, mode='economic')
        # BACKWARD PASS
        problem.select_ode('minus_S', (U1, V2))
        S1_bis = solve_matrix_ivp(problem, (t0+t_half, t1), Sh_hat.T.conj(), **self.substep_kwargs)
        K1 = U1.dot(S1_bis)
        problem.select_ode('K', mats_uv=(V2,))
        K2 = solve_matrix_ivp(problem, (t0+t_half, t1), K1, **self.substep_kwargs)
        U2, S2 = la.qr(K2, mode='economic')
        # SOLUTION
        Y1 = QuasiSVD(U2, S2, V2)
        return Y1
        

