from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from util import group_lasso_loss, extract_config, run_method


def ALM_dual(x0: NDArray,
             A: NDArray,
             b: NDArray,
             mu: float,
             opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    r"""
    Solve the group LASSO problem using Augmented Lagrangian method.

    dual form of the group LASSO problem:
    :math:
    $$
    \begin{align*}
    \min_{Y\in \mathbb{R}^{m\times l}}\, &\left<b, Y\right> + \frac{1}{2} \|Y\|_F^2
    \\
    \text{s.t.}\,\, & \|A^T Y(:,i)\|_2 \leq \mu, \forall i = 1,2,\ldots,l
    \end{align*}
    $$

    we rewrite the problem as
    :math:
    $$
    \begin{align*}
    \min_{Y\in \mathbb{R}^{m\times l}}\, &\left<b, Y\right> + \frac{1}{2} \|Y\|_F^2
    \\
    \text{s.t.}\,\, & A^T Y = Z\\
    & \|Z(:,i)\|_2 \leq \mu, \forall i = 1,2,\ldots,l
    \end{align*}
    $$

    The augmented Lagrangian is
    :math:
    $$
    L(Y,Z;X) = \left<b, Y\right> + \frac{1}{2} \|Y\|_F^2 + \left<X, A^T Y - Z\right> + \frac{\sigma}{2} \|A^T Y - Z\|_F^2\\
     \text{where } \|Z(:,i)\|_2 \leq \mu, \forall i = 1,2,\ldots,l
    $$

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

    tol = extract_config(opt, 'tol', 1e-8)
    tol_inner = extract_config(opt, 'tol_inner', 1e-4)
    max_iter = extract_config(opt, 'maxiter', 5000)
    log = extract_config(opt, 'log', True)

    x = x0
    y = np.zeros((m, l), dtype=float)
    z = np.zeros((n, l), dtype=float)

    obj_val_list = []
    dual_gap_list = []

    sigma = 100.0
    total_iter = 0

    inv = np.linalg.inv(sigma * A @ A.T + np.eye(m))
    step_size = (1 + np.sqrt(5)) / 2

    while iter_count <= max_iter:
        if log:
            print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, -x, mu)}")

        iter_count += 1

        for i in range(50):
            total_iter += 1
            y = inv @ (A @ (sigma * z - x) - b)
            z_old = z.copy()
            z = x / sigma + A.T @ y
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            mask = norms < mu
            norms[mask] = mu
            z = z * (mu / norms)
            if np.linalg.norm(z - z_old, ord="fro") < tol_inner:
                break

        x = x + step_size * sigma * (A.T @ y - z)
        obj_val = group_lasso_loss(A, b, -x, mu)
        obj_val_list.append(obj_val)
        dual_val = np.sum(b * y) + 0.5 * np.linalg.norm(y, "fro") ** 2
        dual_gap_list.append(abs(obj_val - dual_val))
        if np.linalg.norm(A.T @ y - z, ord="fro") < tol:
            break

    if log:
        print(f"Total inner iterations: {total_iter}")
    return -x, iter_count, {'obj_val': obj_val_list, 'dual_gap': dual_gap_list}


if __name__ == '__main__':
    run_method(ALM_dual, benchmark=True)
