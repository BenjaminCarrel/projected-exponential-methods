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
from krylov_toolbox import KrylovSpace, ExtendedKrylovSpace, RationalKrylovSpace
from matrix_ode_toolbox.phi import sylvester_combined_phi_k
from numpy import ndarray


Matrix = ndarray | LowRankMatrix


#%% Class Projected Exponential Runge-Kutta
class ProjectedExponentialRungeKutta(DlraSolver):
    """
    Projected exponential Runge-Kutta methods.
    The methods aim at solving the Sylvester-like ODE
        X'(t) = A X(t) + X(t) B + G(t, X),
        X(0) = X0.
    See Carrel & Vandereycken (to be published, see README).
    """

    name = 'Projected exponential Runge-Kutta (PERK)'
    
    def __init__(self,
                 matrix_ode: SylvesterLikeOde,
                 nb_substeps: int = 1,
                 order: int = 1,
                 krylov_kwargs: dict = {'kind': 'extended', 'size': 1},
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
        elif self.krylov_kind == 'polynomial':
            self.Krylov_space = KrylovSpace
        else:
            raise ValueError(f'Krylov kind {self.krylov_kind} not implemented.')
        # Check closed form argument
        if 'use_closed_form' in extra_args:
            self.use_closed_form = extra_args['use_closed_form']
            del extra_args['use_closed_form']
        else:
            self.use_closed_form = True # Fast but Sylvester operator must be invertible
            print('Warning: closed form solver is used by default. The Sylvester operator must be invertible. If your operator is not invertible, set use_closed_form=False.')
        # pop extra agrs for scipy solver
        if 'scipy_method' in extra_args:
            self.scipy_method = extra_args['scipy_method']
            del extra_args['scipy_method']
        else:
            self.scipy_method = 'LSODA' # default solver is scipy's LSODA for automatic stiffness detection
        
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
    
    def exp_sylvester(self, h: float, A: Matrix, B: Matrix, X: Matrix) -> Matrix:
        "Shortcut for computing the matrix exponential."
        Z = spala.expm_multiply(A, X, start=0, stop=h, endpoint=True, num=2)[-1]
        Z = spala.expm_multiply(B.T.conj(), Z.T.conj(), start=0, stop=h, endpoint=True, num=2)[-1].T.conj()
        return Z
    
    def solve_sylvester_one_ode(self, h: float, A: Matrix, B: Matrix, X0: SVD, X1: SVD) -> QuasiSVD:
        "Solve the Sylvester ODE X' = AX + XB + X1, X(0) = X0."
        ## Compute the two Krylov spaces
        Qk, Wk = self.compute_Krylov_basis([X0, X1])
        ## Reduced matrices
        A_reduced = Qk.T.conj().dot(A.dot(Qk))
        B_reduced = Wk.T.conj().dot(B.dot(Wk))
        X0_reduced = X0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        X1_reduced = X1.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        ## Solve reduced ODE (closed form)
        if self.use_closed_form:
            C = la.solve_sylvester(A_reduced, B_reduced, X1_reduced) # NOTE: the Sylvester operator must be invertible
            Z = X0_reduced + C
            Z = self.exp_sylvester(h, A_reduced, B_reduced, Z)
            S1 = Z - C
            output = QuasiSVD(Qk, S1, Wk)
        ## Solve reduced ODE (default solver is scipy's LSODA for automatic stiffness detection)
        else:
            S1 = sylvester_combined_phi_k(A_reduced, B_reduced, h, k=1, X=[X0_reduced, X1_reduced], solver='scipy', scipy_method=self.scipy_method)
            output = QuasiSVD(Qk, S1, Wk)
        return output
    
    def solve_sylvester_two_ode(self, h: float, A: Matrix, B: Matrix, X0: SVD, X1: SVD, X2: SVD) -> QuasiSVD:
        """
        Solve the Sylvester ODE:
            X'(t) = AX + XB + X1 + t/h * X2, 
            X(0) = X0.
        The computation are done efficiently by reducing the problem with Krylov spaces.
        The reduced problem is solved with a closed form formula or with scipy.
        
        """
        ## Compute the two Krylov spaces
        Qk, Wk = self.compute_Krylov_basis([X0, X1, X2])
        ## Reduced matrices
        A_reduced = Qk.T.conj().dot(A.dot(Qk))
        B_reduced = Wk.T.conj().dot(B.dot(Wk))
        X0_reduced = X0.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        X1_reduced = X1.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        X2_reduced = X2.dot(Wk).dot(Qk.T.conj(), side='left').todense()
        ## Solve reduced ODE (closed form)
        if self.use_closed_form:
            # Split computations over kernel and range
            NA = la.null_space(A_reduced)
            OA = la.orth(A_reduced)
            NB = la.null_space(B_reduced)
            OB = la.orth(B_reduced)

            # Compute the projected matrices A and B
            Ar_reduced = np.linalg.multi_dot([OA, OA.T.conj(), A_reduced, OA, OA.T.conj()])
            A_zero = np.zeros(A_reduced.shape)
            Br_reduced = np.linalg.multi_dot([OB, OB.T.conj(), B_reduced, OB, OB.T.conj()])
            B_zero = np.zeros(B_reduced.shape)

            # Compute the projected matrices X0, X1 and X2
            X01 = np.linalg.multi_dot([NA, NA.T, X0_reduced, NB, NB.T])
            X02 = np.linalg.multi_dot([NA, NA.T, X0_reduced, OB, OB.T])
            X03 = np.linalg.multi_dot([OA, OA.T, X0_reduced, NB, NB.T])
            X04 = np.linalg.multi_dot([OA, OA.T, X0_reduced, OB, OB.T])
            X11 = np.linalg.multi_dot([NA, NA.T, X1_reduced, NB, NB.T])
            X12 = np.linalg.multi_dot([NA, NA.T, X1_reduced, OB, OB.T])
            X13 = np.linalg.multi_dot([OA, OA.T, X1_reduced, NB, NB.T])
            X14 = np.linalg.multi_dot([OA, OA.T, X1_reduced, OB, OB.T])
            X21 = np.linalg.multi_dot([NA, NA.T, X2_reduced, NB, NB.T])
            X22 = np.linalg.multi_dot([NA, NA.T, X2_reduced, OB, OB.T])
            X23 = np.linalg.multi_dot([OA, OA.T, X2_reduced, NB, NB.T])
            X24 = np.linalg.multi_dot([OA, OA.T, X2_reduced, OB, OB.T])

            # Define the solver
            def closed_form_solver(A, B, X0, X1, X2):
                # Solve the Sylvester equations
                D = la.solve_sylvester(A, B, X2)
                D_hat = la.solve_sylvester(A, B, 1/h * D)
                C = la.solve_sylvester(A, B, X1)
                # Assemble
                Z = X0 + C + D_hat
                Z = self.exp_sylvester(h, A, B, Z)
                S = Z - C - D_hat - D
                return S
            
            # Compute on each projected matrix
            ## On the Kernel
            S1 = X01 + X11 + h/2 * X21
            ## A = 0, B = Br_reduced
            S2 = closed_form_solver(A_zero, Br_reduced, X02, X12, X22)
            ## A = Ar_reduced, B = 0
            S3 = closed_form_solver(Ar_reduced, B_zero, X03, X13, X23)
            ## A = Ar_reduced, B = Br_reduced
            S4 = closed_form_solver(Ar_reduced, Br_reduced, X04, X14, X24)

            # Assemble
            S = S1 + S2 + S3 + S4
            output = QuasiSVD(Qk, S, Wk)
        ## Solve reduced ODE (with scipy LSODA for automatic stiffness detection)
        else:
            S2 = sylvester_combined_phi_k(A_reduced, B_reduced, h, k=2, X=[X0_reduced, X1_reduced, 1/h * X2_reduced], solver='scipy', scipy_method=self.scipy_method)
            output = QuasiSVD(Qk, S2, Wk)
        return output
    
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

        # SOLVE THE EQUIVALENT ODE
        Y1 = self.solve_sylvester_one_ode(h, self.matrix_ode.A, self.matrix_ode.B, Y0, PGY0)

        # RETRACT
        Y1 = self.retraction(Y1)
        return Y1
    
    def projected_exponential_Runge_non_strict(self, t_subspan: tuple, Y0: SVD) -> SVD:
        """
        Projected exponential Runge method (weak). Not proven to be robust order 2.
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

        # STEP 3: SOLVE THE EQUIVALENT ODE
        Y1 = self.solve_sylvester_one_ode(h, self.matrix_ode.A, self.matrix_ode.B, Y0, PGY_half)

        # RETRACT
        Y1 = self.retraction(Y1)
        return Y1
    
    def projected_exponential_Runge_strict(self, t_subspan: tuple, Y0: SVD) -> SVD:
        """
        Projected exponential Runge (strong) method with stiff order conditions. Proven to be robust order 2.
        Solves an equation of the form X' = AX + XB + G(X) = L(X) + G(X)
        The scheme is:
        K1 = exp(c2 * h L) (Y0) + c2 * h * phi_1(c2 * h L) (P(Y0)[G(Y0)])
        RK1 = R(K1)
        Y1 = exp(h L) (Y0) + h phi_1(h L) (P(Y0)[G(Y0)]) + h * phi_2(h L) (P(RK1)[G(RK1)] - P(Y0)[G(Y0)])
        """
        # VARIABLES
        t0, t1 = t_subspan
        h = t1 - t0
        c2 = 1

        # STEP 1: PERK1
        GY0 = self.matrix_ode.non_stiff_field(t0, Y0)
        PGY0 = Y0.project_onto_tangent_space(GY0)
        K1 = self.solve_sylvester_one_ode(c2 * h, self.matrix_ode.A, self.matrix_ode.B, Y0, PGY0)
        RK1 = self.retraction(K1)

        # STEP 2: COMPUTE THE PROJECTION: P(RK1)[G(RK1)]
        GRK1 = self.matrix_ode.non_stiff_field(t0 + c2 * h, RK1)
        PGRK1 = RK1.project_onto_tangent_space(GRK1)
        PGRK1_minus_PGY0 = PGRK1 - PGY0

        # STEP 3: SOLVE THE EQUIVALENT ODE
        Y1 = self.solve_sylvester_two_ode(h, self.matrix_ode.A, self.matrix_ode.B, Y0, PGY0, PGRK1_minus_PGY0)

        # RETRACT
        Y1 = self.retraction(Y1)
        return Y1


