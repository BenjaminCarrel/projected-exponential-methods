"""
Author: Benjamin Carrel, University of Geneva, 2022

Methods currently implemented:
- MatrixScipySolver: scipy based solver 
- ClosedFormSolver: closed form solver (the closed form needs to be implemented in the problem)
- RungeKuttaSolver: Runge-Kutta solver (for testing purposes)
- MatrixOptimalSolver: optimal solver (closed form if available, scipy otherwise) (default)
"""
from .scipy_solver import ScipySolver
from .closed_form import ClosedFormSolver
from .explicit_runge_kutta import ExplicitRungeKutta
from .exponential_runge_kutta import ExponentialRungeKutta
