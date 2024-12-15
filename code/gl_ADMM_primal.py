from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from util import group_lasso_loss, extract_config, test_and_plot


def ADMM_primal(x0: NDArray,
                A: NDArray,
                b: NDArray,
                mu: float,
                opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    r"""
    Solve the group LASSO problem using Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    x0 : NDArray
        Initial guess for the solution.
    A : NDArray
        Coefficients of the monomials in the objective function.
    b : NDArray
        Coefficients of the monomials in the constraints.
    mu : float
        The parameter in the geometric programming problem.
    opt : dict, optional
        Options for the solver.

        if it contains key 'tol', the value is the tolerance.
        if it contains key 'tol_inner', the value is the tolerance for the inner loop.
        if it contains key 'maxiter', the value is the maximum iteration.
        if it contains key 'log', the value is whether to print the log.

    Returns
    -------
    Tuple[NDArray, int, Dict]
        The solution, the iteration count(-1 if known), and the solver information.

        The solver information contains the following keys:

        - obj_val: the list of objective values during the iterations.
        - dual_gap: the list of dual gaps during the iterations.
    """
    m, n = A.shape
    _, l = b.shape
    iter_count = 0

    tol = extract_config(opt, 'tol', 1e-7)
    max_iter = extract_config(opt, 'maxiter', 5000)
    log = extract_config(opt, 'log', True)

    x = x0
    y = np.zeros((n, l), dtype=float)
    z = np.zeros((n, l), dtype=float)

    obj_val_list = []
    dual_gap_list = []

    rho = 5.0

    inv = np.linalg.inv(A.T @ A + rho * np.eye(n))
    step_size = (1 + np.sqrt(5)) / 2

    while iter_count <= max_iter:
        if log and iter_count % 50 == 0:
            print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, x, mu)}")

        iter_count += 1

        # update x
        x = inv @ (A.T @ b + rho * z - y)

        # update z
        z = x + y / rho
        norms = np.linalg.norm(z, ord=2, axis=1, keepdims=True)
        z -= z * np.minimum(1, (mu / rho) / norms)

        # update y
        y = y + step_size * rho * (x - z)

        obj_val = group_lasso_loss(A, b, x, mu)
        obj_val_list.append(obj_val)
        if np.linalg.norm(x - z, ord="fro") < tol:
            break

    return x, iter_count, {'obj_val': obj_val_list, 'dual_gap': dual_gap_list}


if __name__ == '__main__':
    test_and_plot(ADMM_primal, log_scale=False, benchmark=True)
