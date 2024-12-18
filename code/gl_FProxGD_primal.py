from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from util import group_lasso_loss, extract_config, run_method


def FProxGD_primal(x0: NDArray,
                   A: NDArray,
                   b: NDArray,
                   mu: float,
                   opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """
    Solve the group LASSO problem using Fast Proximal Gradient Descent.

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
        if it contains key 'maxiter', the value is the maximum iteration.
        if it contains key 'log', the value is whether to print the log.

    Returns
    -------
    Tuple[NDArray, int, Dict]
        The solution, the iteration count(-1 if known), and the solver information.

        The solver information contains the following keys:

        - 'obj_val': the list of objective values during the iterations.
        - 'grad_norm': the list of gradient norms during the iterations.
    """
    _, l = b.shape
    iter_count = 0
    x = x0

    step_size = 1 / np.linalg.norm(A, ord=2) ** 2
    f_last = np.inf
    tol = extract_config(opt, 'tol', 1e-8)
    max_iter = extract_config(opt, 'maxiter', 5000)
    log = extract_config(opt, 'log', True)
    orig_mu = mu

    mu_list = list(reversed([(6 ** i) * mu for i in range(5)]))
    last_idx = len(mu_list) - 1

    obj_val_list = []
    grad_norm_list = []
    last_x = [x.copy(), x.copy()]

    for k, mu in enumerate(mu_list):
        if iter_count > max_iter:
            break

        inner_iter = 0
        while True:
            # If log is enabled, print the log every 100 iterations.
            if log and iter_count % 20 == 0:
                print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, x, orig_mu)}")

            iter_count += 1
            inner_iter += 1

            if iter_count > max_iter or (inner_iter > 50 and k != last_idx):
                break

            if k != last_idx:
                step = step_size
            else:
                step = step_size if inner_iter <= 100 else step_size / np.sqrt(inner_iter - 100)

            y = last_x[1] + (inner_iter - 2) / (inner_iter + 1) * (last_x[1] - last_x[0])
            grad = A.T @ (A @ y - b)
            x = y - step * grad

            # Apply prox
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x -= x * np.minimum(1, mu * step / norms)

            f_new = group_lasso_loss(A, b, x, mu)
            grad_norm = np.linalg.norm(grad, ord="fro")

            obj_val_list.append(f_new)
            grad_norm_list.append(grad_norm)

            if grad_norm * step < tol or abs(f_last - f_new) < tol:
                break

            f_last = f_new
            last_x[0] = last_x[1]
            last_x[1] = x.copy()

    return x, iter_count, {'obj_val': obj_val_list, 'grad_norm': grad_norm_list}


if __name__ == '__main__':
    run_method(FProxGD_primal, benchmark=True)
