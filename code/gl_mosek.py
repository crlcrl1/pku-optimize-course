import mosek as msk
import numpy as np
import sys

from util import run_method
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def stream_printer(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def mosek(_x0: NDArray,
          A: NDArray,
          b: NDArray,
          mu: float,
          opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """
    #no_benchmark
    Solve the group LASSO problem using MOSEK directly.
    
    We rewrite the group LASSO problem as a SOCP problem:
    
    min     0.5 * t + mu * (s1 + s2 + ... + sn)
    s.t.    ||Ax - b||_F^2 <= 2 * t * t0
            t0 = 0.5
            ||x(i, :)||_2 <= si, i = 1, 2, ..., n

    Parameters
    ----------
    _x0 : NDArray
        Initial guess for the solution, not supported by MOSEK.
    A : NDArray
        Coefficients of the monomials in the objective function.
    b : NDArray
        Coefficients of the monomials in the constraints.
    mu : float
        The parameter in the geometric programming problem.
    opt : dict, optional
        Options for the solver.
        
        if it contains key 'maxtime', the value is the maximum time in seconds.
        if it contains key 'log', the value is whether to print the log.

    Returns
    -------
    Tuple[NDArray, int, Dict]
        The solution, the iteration count(-1 if known), and the solver information.
    """
    m, n = A.shape
    _, l = b.shape

    var_cnt = 2 + n + n * l

    # Create the MOSEK environment and task
    with msk.Env() as env:
        with env.Task() as task:
            if opt is not None and opt['log']:
                task.set_Stream(msk.streamtype.log, stream_printer)

            if opt is not None and 'maxtime' in opt:
                task.putdouparam(msk.dparam.optimizer_max_time, opt['maxtime'])

            task.appendvars(var_cnt)

            # Set bounds for variables
            task.putvarbound(0, msk.boundkey.lo, 0.0, np.inf)  # t >= 0
            task.putvarbound(1, msk.boundkey.fx, 0.5, 0.5)  # t0 = 0.5
            for i in range(n):
                task.putvarbound(2 + i, msk.boundkey.lo, 0.0, np.inf)  # s >= 0
            for i in range(n * l):
                task.putvarbound(2 + n + i, msk.boundkey.fr, -np.inf, np.inf)  # x(i, j) free

            # Set the objective function
            task.putcj(0, 0.5)  # 0.5 * t
            for i in range(n):
                task.putcj(2 + i, mu)  # mu * s_i

            task.appendafes(m * l + 2 + n * (l + 1))

            # Data for constraint ||Ax - b||_F^2 <= t * t0
            # index range: 0 - m * l + 2
            # left-hand side
            task.putafefentrylist([0], [0], [1.0])
            task.putafefentrylist([1], [1], [1.0])
            for i in range(m):
                for j in range(l):
                    col_idx = [2 + n + k * l + j for k in range(n)]
                    row_vals = [A[i, k] for k in range(n)]
                    task.putafefentrylist([2 + i * l + j] * n, col_idx, row_vals)
            # right-hand side
            task.putafegslice(2, m * l + 2, [-b[i, j] for i in range(m) for j in range(l)])

            # Data for constraint ||x(i, :)||_2 <= s_i, i = 1, 2, ..., n
            offset = 2 + m * l
            for i in range(n):
                row_idx = [offset + i * (l + 1) + j for j in range(l + 1)]
                col_idx = [i + 2] + [2 + n + i * l + j for j in range(l)]
                row_vals = [1.0] * (l + 1)
                task.putafefentrylist(row_idx, col_idx, row_vals)

            task.appendcons(1 + n)

            # Set constraint ||Ax - b||_F^2 <= t * t0
            r_quad_cone = task.appendrquadraticconedomain(2 + m * l)
            task.appendacc(r_quad_cone, range(m * l + 2), None)

            # Set constraint ||x(i, :)||_2 <= s_i, i = 1, 2, ..., n
            for i in range(n):
                quad_cone = task.appendquadraticconedomain(1 + l)
                task.appendacc(quad_cone, [2 + m * l + i * (l + 1) + j for j in range(l + 1)], None)

            # Solve the problem
            task.putobjsense(msk.objsense.minimize)
            task.optimize()

            # Get the solution
            result = task.getxxslice(msk.soltype.itr, 2 + n, 2 + n + n * l)
            result = np.array(result).reshape((n, l))

            # Get the iteration count
            num_it = task.getintinf(msk.iinfitem.intpnt_iter)

            # Get the solver information
            res = {'status': task.getprosta(msk.soltype.itr), 'obj': task.getprimalobj(msk.soltype.itr)}

            return result, num_it, res


if __name__ == "__main__":
    run_method(mosek, plot=False)
