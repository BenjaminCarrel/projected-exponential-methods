"""
Author: Benjamin Carrel, University of Geneva, 2022

Integrators based on a low-rank splitting.
Applied only to ODEs of the form
X' = A X + X B + G(t,X)
See Ostermann et al. 2019.
"""

#%% Imports
from numpy import ndarray
from matrix_ode_toolbox import SylvesterOde, RiccatiOde, SylvesterLikeOde
from matrix_ode_toolbox.dlra import DlraSolver
from low_rank_toolbox import LowRankMatrix
from .projector_splitting import ProjectorSplitting

Matrix = LowRankMatrix | ndarray


#%% Class LowRankSlitting
class LowRankSplitting(DlraSolver):
    """
    Class low-rank splitting, inherits from DlraSolver.
    See Ostermann et al. 2019.
    """

    name = 'Low-rank splitting'

    def __init__(self, 
                matrix_ode: SylvesterOde | RiccatiOde | SylvesterLikeOde, 
                nb_substeps: int = 1, 
                order: int = 2,
                dlra_substepper: DlraSolver = ProjectorSplitting,
                substep_kwargs: dict = {'solver': 'scipy'},
                **extra_args) -> None:
        # Check the input
        if order not in [1, 2]:
            raise ValueError("The order of the splitting method must be 1 or 2.")
        if not isinstance(matrix_ode, (SylvesterOde, RiccatiOde, SylvesterLikeOde)):
            raise TypeError("The matrix ODE is not supported yet.")
        
        # Initialize the DlraSolver
        super().__init__(matrix_ode, nb_substeps)
        self.order = order
        if order==1:
            self.splitting_name = "Lie-Trotter"
            self.splitting = LieTrotter
        elif order==2:
            self.splitting_name = "Strang"
            self.splitting = Strang

        # Define the non-stiff solver
        if isinstance(matrix_ode, SylvesterOde):
            non_stiff_problem = SylvesterOde(matrix_ode.A*0, matrix_ode.B*0, matrix_ode.C)
        elif isinstance(matrix_ode, RiccatiOde):
            non_stiff_problem = RiccatiOde(matrix_ode.A*0, matrix_ode.C, matrix_ode.D)
        elif isinstance(matrix_ode, SylvesterLikeOde):
            non_stiff_problem = SylvesterLikeOde(matrix_ode.A*0, matrix_ode.B*0, matrix_ode.G)
        self.dlra_substepper = dlra_substepper(non_stiff_problem, substep_kwargs=substep_kwargs, **extra_args)

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Low-rank splitting (Ostermann et al. 2019) \n'
        info += f'-- Order {self.order} splitting ({self.splitting_name}) \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- {self.dlra_substepper.name} as low-rank solver'
        return info


    def stepper(self, t_subspan: tuple, Y0: LowRankMatrix):
        "Stepper for the low-rank splitting."
        return self.splitting(t_subspan, Y0, self._non_stiff_solver, self._stiff_solver)

    def _stiff_solver(self, t_subspan: tuple, Y0: LowRankMatrix) -> LowRankMatrix:
        """
        Closed form solution of the stiff, linear part of the ODE. X' = AX + XB.
        """
        A = self.matrix_ode.A
        B = self.matrix_ode.B
        h = t_subspan[1] - t_subspan[0]
        Y_tilde = Y0.expm_multiply(A, h, side='left')
        Y1 = Y_tilde.expm_multiply(B, h, side='right')
        return Y1

    def _non_stiff_solver(self, t_subspan: tuple, Y0: LowRankMatrix) -> LowRankMatrix:
        """
        DlraSolver of the non-stiff part of the ODE. Y' = P(Y)[G(t,Y)].
        """
        Y1 = self.dlra_substepper.solve(t_subspan, Y0)
        return Y1
        

# %% SPLITTING METHODS
def LieTrotter(t_span: tuple, 
               initial_value: Matrix, 
               solver1: object, 
               solver2: object) -> Matrix:
    """Lie-Trotter splitting.
    For an ODE with a sum
    Y' = F(Y) + G(Y),
    the Lie-Trotter splitting computes
    Y1 = phi_F^h \circ phi_G^h (Y0)
    resulting in an order 1 method in h.

    Args:
        t_span (tuple): time interval (0, h)
        initial_value (Union[ndarray, LowRankMatrix]): initial value
        solver1 (object): first solver function with input (t_span, initial_value)
        solver2 (object): second solver function with input (t_span, initial_value)
    """
    Y0 = initial_value
    Y_half = solver1(t_span, Y0)
    Y1 = solver2(t_span, Y_half)
    return Y1

def Strang(t_span: tuple, 
           initial_value: Matrix, 
           solver1: object, 
           solver2: object) -> Matrix:
    """Strang splitting.
    For an ODE with a sum
    Y' = F(Y) + G(Y),
    the Strang splitting computes
    Y1 = phi_F^{h/2} \circ phi_G^h \circ phi_F^{h/2} (Y0)
    resulting in an order 2 method in h.

    Args:
        t_span (tuple): time interval (0, h)
        initial_value (Union[ndarray, LowRankMatrix]): initial value
        solver1 (object): first solver function with input (t_span, initial_value)
        solver2 (object): second solver function with input (t_span, initial_value)
    """
    t0 = t_span[0]
    t1 = t_span[1]
    h = t1 - t0
    Y0 = initial_value
    Y_one = solver1((t0, t0 + h/2), Y0)
    Y_two = solver2(t_span, Y_one)
    Y1 = solver1((t0 + h/2, t1), Y_two)
    return Y1
