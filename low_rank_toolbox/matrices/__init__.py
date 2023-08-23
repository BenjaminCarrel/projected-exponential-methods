"""
Authors: Benjamin Carrel and Rik Vorhaar
        University of Geneva, 2022

Currently supported low-rank matrix formats:
- Generic low-rank matrix format
- Quasi-SVD
- SVD
"""
from .low_rank_matrix import LowRankMatrix
from .svd import QuasiSVD, SVD