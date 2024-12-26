import cvxpy as cp

from util import run_method
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def gl_cvx_mosek(x0: NDArray,
                 A: NDArray,
                 b: NDArray,
                 mu: float,
                 _opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """
    #no_benchmark
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
    _opt : dict, optional
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
    prob.solve(solver=cp.MOSEK)

    return x.value, prob.solver_stats.num_iters, prob.solver_stats


if __name__ == "__main__":
    run_method(gl_cvx_mosek, plot=False)
