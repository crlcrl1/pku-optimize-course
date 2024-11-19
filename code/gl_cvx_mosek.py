import cvxpy as cp
import numpy as np

from random_data import generate_data
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional

def cvx_mosek(x0: NDArray,
              A: NDArray,
              b: NDArray,
              mu: float,
              opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """
    Solve the group LASSO problem using CVXPY with MOSEK.

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

    Returns
    -------
    Tuple[NDArray, int, Dict]
        The solution, the iteration count(-1 if known), and the solver information.
    """
    m, n = A.shape
    _, l = b.shape
    x = cp.Variable((n, l))
    x.value = x0
    obj = 0.5 * cp.norm(A @ x - b, "fro") ** 2 + mu * cp.sum(cp.norm(x, axis=1))
    prob = cp.Problem(cp.Minimize(obj))
    if opt is not None:
        prob.solve(solver=cp.MOSEK, **opt)
    else:
        prob.solve(solver=cp.MOSEK)
    
    return x.value, prob.solver_stats.num_iters, prob.solver_stats


def cvx_mosek_test():
    A, b, x0 = generate_data()
    mu = 1e-2
    x, iter, opt = cvx_mosek(x0, A, b, mu)
    print(x)
    print(iter)
    print(opt)
    print("Objective value: ", 0.5 * np.linalg.norm(A @ x - b, "fro") ** 2 + mu * np.sum(np.linalg.norm(x, axis=1)))


if __name__ == "__main__":
    cvx_mosek_test()