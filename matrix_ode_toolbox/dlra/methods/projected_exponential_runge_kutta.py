"""
Author: Benjamin Carrel, University of Geneva, 2022

Projected exponential Runge-Kutta methods for solving Sylvester-like matrix differential equations.
"""

# %% Imports
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spala
from matrix_ode_toolbox.dlra import DlraSolver
from matrix_ode_toolbox import SylvesterLikeOde
from low_rank_toolbox import LowRankMatrix, SVD, QuasiSVD
from krylov_toolbox import ExtendedKrylovSpace, RationalKrylovSpace
from numpy import ndarray


Matrix = ndarray | LowRankMatrix


#%% Class Projected Exponential Runge-Kutta
class ProjectedExponentialRungeKutta(DlraSolver):
    """
    Projected exponential Runge-Kutta methods.
    The methods aim at solving the Sylvester-like ODE
        X'(t) = A X(t) + X(t) B + G(t, X),
        X(0) = X0.
    See Carrel & Vandereycken (to appear).
    """

    name = 'Projected exponential Runge-Kutta (PERK)'
    
    def __init__(self,
                 matrix_ode: SylvesterLikeOde,
                 nb_substeps: int = 1,
                 order: int = 1,
                 krylov_kwargs: dict = {'kind': 'extended', 'size': 5},
                 strict_order_conditions: bool = True,
                 **extra_args) -> None:
        
        super().__init__(matrix_ode, nb_substeps)
        self.order = order
        self.strict_order_conditions = strict_order_conditions
        self.krylov_kwargs = krylov_kwargs.copy()

        # Process the Krylov kwargs
        self.krylov_size = krylov_kwargs['size']
        self.krylov_kind = krylov_kwargs['kind']
        del self.krylov_kwargs['size']
        del self.krylov_kwargs['kind']
        self.left_krylov_args = self.krylov_kwargs.copy()
        self.right_krylov_args = self.krylov_kwargs.copy()
        self.size_factor = 1
        if self.krylov_kind == 'extended':
            # Preprocess inverses of A and B
            if not 'invA' in krylov_kwargs:
                self.left_krylov_args['invA'] = spala.splu(self.matrix_ode.A).solve
            else:
                self.left_krylov_args['invA'] = krylov_kwargs['invA']
            if not 'invB' in krylov_kwargs:
                self.right_krylov_args['invB'] = spala.splu(self.matrix_ode.B).solve
            else:
                self.right_krylov_args['invB'] = krylov_kwargs['invB']
            self.Krylov_space = ExtendedKrylovSpace
            self.size_factor = 2
        elif self.krylov_kind == 'rational':
            # Preprocess the poles here
            self.left_krylov_args['poles'] = krylov_kwargs['poles']
            self.right_krylov_args['poles'] = krylov_kwargs['poles']
            self.Krylov_space = RationalKrylovSpace
        else:
            raise ValueError(f'Krylov kind {self.krylov_kind} not implemented.')
        
        # If the rank decreases, we need to store the original rank
        self.rank = None

        

    @property
    def info(self) -> str:
        "Return the info string."
        info = f'Projected exponential Runge-Kutta (PERK) \n'
        info += f'-- {self.order} stage(s) \n'
        info += f'-- Strict order conditions: {self.strict_order_conditions} \n'
        info += f'-- {self.nb_substeps} substep(s) \n'
        info += f'-- {self.krylov_kind} Krylov space and {self.krylov_size} iteration(s) \n'
        info += f'-- Largest matrix shape: ({2 * self.size_factor * self.krylov_size * self.order}*rank, {2 * self.size_factor * self.krylov_size * self.order}*rank)'
        return info

    def stepper(self, t_subspan: tuple, Y0: SVD) -> SVD: # change manually the method for testing
        "Wrapper around the correct order stepper."
        if self.rank is None:
            self.rank = Y0.rank
        if self.order == 1:
            return self.projected_exponential_Euler(t_subspan, Y0)
        elif self.order == 2:
            if self.strict_order_conditions:
                return self.projected_exponential_Runge_strict(t_subspan, Y0) 
            else:
                return self.projected_exponential_Runge_non_strict(t_subspan, Y0)
        else:
            raise ValueError(f'Order {self.order} not implemented.') 
        
    def retraction(self, X: SVD) -> SVD:
        "Truncate the SVD to the current rank. Overwrite this for an adaptive rank."
        return X.truncate(self.rank)
        
    def compute_Krylov_basis(self, matrices: list):
        """
        Compute the left and right basis.
        Vk is the basis of K_k(A, V0) where V0 is the left basis of matrices
        Wk is the basis of K_k(B, W0) where W0 is the right basis of matrices

        Parameters
        ----------
        matrices : list
            List of matrices.

        Returns
        -------
        Vk : ndarray
            Left basis.
        Wk : ndarray
            Right basis.
        """

        # Orthogonalize the two basis
        V0 = np.concatenate([matrices[i].U for i in np.arange(len(matrices))], axis=1)
        V0 = la.orth(V0)
        W0 = np.concatenate([matrices[i].V for i in np.arange(len(matrices))], axis=1)
        W0 = la.orth(W0)

        # Create the two Krylov spaces
        left_space = self.Krylov_space(self.matrix_ode.A, V0, **self.left_krylov_args)
        right_space = self.Krylov_space(self.matrix_ode.B, W0, **self.right_krylov_args)

        # Iterate
        for _ in np.arange(self.krylov_size - 1):
            left_space.augment_basis()
            right_space.augment_basis()

        # Return the basis
        Vk = left_space.basis
        Wk = right_space.basis

        return Vk, Wk
    
    def projected_exponential_Euler(self, t_subspan: tuple, Y0: SVD) -> SVD:
        """
        Projected exponential Euler.
        Solves a matrix ODE of the form X' = AX + XB + G(X).
        Exponential Euler: Y1 = exp(hL) (Y0) + h phi_1(hL) (G(Y0))
        Projected Exponential Euler: Y1 = R( exp(hL) (Y0) + h phi_1(hL) (P(Y0)[G(Y0)]) )
        where h*phi_1(hL) = L^-1 ( exp(hL) - I )
        """
        # VARIABLES
        t0, t1 = t_subspan
        h = t1 - t0

        # COMPUTE THE PROJECTION: P(Y0)[G(Y0)]
        GY0 = self.matrix_ode.non_stiff_field(t0, Y0)
        PGY0 = Y0.project_onto_tangent_space(GY0)

        # COMPUTE THE INNER TERM: exp(hL) (Y0) + h phi(hL) (P(Y0)[G(Y0)])
        ## Two Krylov spaces (left and right)
        Qk, Wk = self.compute_Krylov_basis([Y0, PGY0])
        ## Reduced matrices
        A_reduced = Qk.T.conj().dot(self.matrix_ode.A.dot(Qk))
        B_reduced = Wk.T.conj().dot(self.matrix_ode.B.dot(Wk))
        PGY0_reduced = PGY0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        Y0_reduced = Y0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        ## Solve reduced ODE (closed form) 
        C = la.solve_sylvester(A_reduced, B_reduced, PGY0_reduced) # NOTE: the Sylvester operator must be invertible
        Z = Y0_reduced + C
        Z = spala.expm_multiply(A_reduced, Z, start=0, stop=h, endpoint=True, num=2)[-1]
        Z = spala.expm_multiply(B_reduced.T.conj(), Z.T.conj(), start=0, stop=h, endpoint=True, num=2)[-1].T.conj()
        S1 = Z - C

        # ASSEMBLE AND RETRACT
        Y1 = QuasiSVD(Qk, S1, Wk)
        Y1 = self.retraction(Y1)
        return Y1
    
    def projected_exponential_Runge_non_strict(self, t_subspan: tuple, Y0: SVD) -> SVD:
        """
        Projected exponential Runge method.
        This method is not strictly order 2, see (Luan & Ostermann, 2014).
        Solves a matrix ODE of the form X' = AX + XB + G(X).
        Midpoint: Y_half = R( exp(h/2 L) (Y0) + h/2 phi(h/2 L) (P(Y0)[G(Y0)]) )
        Y1 = R( exp(h L) (Y0) + h phi(h L) (P(Y_half)[G(Y_half)]) )
        """
        # VARIABLES
        t0, t1 = t_subspan
        h = t1 - t0
        h_half = h/2

        # STEP 1: CALL PERK1
        Y_half = self.projected_exponential_Euler((t0, t0 + h_half), Y0)

        # STEP 2: COMPUTE THE PROJECTION: P(Y_half)[G(Y_half)]
        GY_half = self.matrix_ode.non_stiff_field(t0 + h_half, Y_half)
        PGY_half = Y_half.project_onto_tangent_space(GY_half)

        # COMPUTE THE INNER TERM: exp(h L) (Y0) + h phi(h L) (P(Y_half)[G(Y_half)])
        ## Two Krylov spaces (left and right)
        Qk, Wk = self.compute_Krylov_basis([Y0, PGY_half])
        ## Reduced matrices
        A_reduced = Qk.T.conj().dot(self.matrix_ode.A.dot(Qk))
        B_reduced = Wk.T.conj().dot(self.matrix_ode.B.dot(Wk))
        PGY_half_reduced = PGY_half.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        Y0_reduced = Y0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        ## Solve reduced ODE (with closed form)
        C = la.solve_sylvester(A_reduced, B_reduced, PGY_half_reduced) # NOTE: the Sylvester operator must be invertible
        Z = Y0_reduced + C
        Z = spala.expm_multiply(A_reduced, Z, start=0, stop=h, endpoint=True, num=2)[-1]
        Z = spala.expm_multiply(B_reduced.T.conj(), Z.T.conj(), start=0, stop=h, endpoint=True, num=2)[-1].T.conj()
        S1 = Z - C

        # ASSEMBLE AND RETRACT
        Y1 = QuasiSVD(Qk, S1, Wk)
        Y1 = self.retraction(Y1)
        return Y1
    
    def projected_exponential_Runge_strict(self, t_subspan: tuple, Y0: SVD) -> SVD:
        """
        Projected exponential Runge method. Version 2, which verifies the stiff order conditions.
        Solves an equation of the form X' = AX + XB + G(X) = L(X) + G(X)
        The scheme is:
        K1 = exp(h L) (Y0) + h * phi_1(h L) (P(Y0)[G(Y0)])
        RK1 = R(K1)
        Y1 = K1 + h * phi_2(h L) (P(RK1)[G(RK1)] - P(Y0)[G(Y0)])
        """
        # VARIABLES
        t0, t1 = t_subspan
        h = t1 - t0
        c2 = 1
        # print('Y0 rank:', Y0.rank)

        # STEP 1: PERK1
        GY0 = self.matrix_ode.non_stiff_field(t0, Y0)
        PGY0 = Y0.project_onto_tangent_space(GY0)
        # print('PGY0 rank:', PGY0.rank)
        # print('Theoretical max size:', 2 * (self.krylov_size +1) * 2 * Y0.rank)
        Qk, Wk = self.compute_Krylov_basis([Y0, PGY0])
        # print('Final dimension of the two bases:', Qk.shape, Wk.shape)
        A_reduced = Qk.T.conj().dot(self.matrix_ode.A.dot(Qk))
        B_reduced = Wk.T.conj().dot(self.matrix_ode.B.dot(Wk))
        PGY0_reduced = PGY0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        Y0_reduced = Y0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        C = la.solve_sylvester(A_reduced, B_reduced, PGY0_reduced) # NOTE: the Sylvester operator must be invertible
        Z = Y0_reduced + C
        Z = spala.expm_multiply(A_reduced, Z, start=0, stop=c2 * h, endpoint=True, num=2)[-1]
        Z = spala.expm_multiply(B_reduced.T.conj(), Z.T.conj(), start=0, stop=c2 * h, endpoint=True, num=2)[-1].T.conj()
        S1 = Z - C
        # print('Shape of first reduced ODE:', S1.shape)
        K1 = QuasiSVD(Qk, S1, Wk)
        RK1 = self.retraction(K1)

        # STEP 2
        ## Compute the projection: P(RK1)[G(RK1)]
        GRK1 = self.matrix_ode.non_stiff_field(t0 + c2 * h, RK1)
        PGRK1 = RK1.project_onto_tangent_space(GRK1)
        PGRK1_minus_PGRY0 = PGRK1 - PGY0
        ## Two Krylov spaces
        Qk, Wk = self.compute_Krylov_basis([Y0, PGY0, PGRK1_minus_PGRY0])
        # print('Theoretical max size:', 2 * (self.krylov_size+1) * (2 * Y0.rank + PGRK1_minus_PGRY0.rank))
        ## Reduced matrices
        A_reduced = Qk.T.conj().dot(self.matrix_ode.A.dot(Qk))
        B_reduced = Wk.T.conj().dot(self.matrix_ode.B.dot(Wk))
        Y0_reduced = Y0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        PGY0_reduced = PGY0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        PGRK1_minus_PGRY0_reduced = PGRK1_minus_PGRY0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        ## Solve reduced ODE (closed form)
        D = la.solve_sylvester(A_reduced * c2, B_reduced * c2, PGRK1_minus_PGRY0_reduced)
        D_hat = la.solve_sylvester(A_reduced * h, B_reduced * h, D)
        C = la.solve_sylvester(A_reduced, B_reduced, PGY0_reduced)
        Z = Y0_reduced + C + D_hat
        Z = spala.expm_multiply(A_reduced, Z, start=0, stop=h, endpoint=True, num=2)[-1]
        Z = spala.expm_multiply(B_reduced.T.conj(), Z.T.conj(), start=0, stop=h, endpoint=True, num=2)[-1].T.conj()
        S2 = Z - C - D_hat - D
        # print('Shape of second reduced ODE:', S2.shape)

        # ASSEMBLE AND TRUNCATE
        Y1 = QuasiSVD(Qk, S2, Wk)
        Y1 = self.retraction(Y1)
        return Y1

