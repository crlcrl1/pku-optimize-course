"""
Generate random data for the group LASSO problem.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def generate_data(seed: int = 97006855) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate random data for the group LASSO problem.
    
    Parameters
    ----------
    seed : int, optional
        The random seed.
    
    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        The data matrix A, the response vector b, and the initial guess x0.
    """
    
    np.random.seed(seed)
    n = 512
    m = 256
    A = np.random.randn(m, n)
    k = round(n * 0.1)
    l = 2
    p = np.random.permutation(n)
    p = p[0: k]
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)
    b = A @ u
    x0 = np.random.randn(n, l)
    return A, b, x0
