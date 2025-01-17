import gurobipy as gp
import numpy as np

from util import run_method
from gurobipy import GRB
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def gl_gurobi(x0: NDArray,
              A: NDArray,
              b: NDArray,
              mu: float,
              opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """
    #no_benchmark
    Solve the group LASSO problem using Gurobi directly.

    We rewrite the group LASSO problem as a SOCP problem:

    ..math::
    $$
    min     0.5 * t + mu * (s1 + s2 + ... + sn)
    s.t.    ||Ax - b||_F^2 <= t
            ||x(i, :)||_2 <= si, i = 1, 2, ..., n
    $$

    Since it's easier to input a QCQP problem to Gurobi, we input the constraints as quadratic constraints.

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

        if it contains key 'log', the value is whether to print the log.

    Returns
    -------
    Tuple[NDArray, int, Dict]
        The solution, the iteration count(-1 if known), and the solver information.
    """
    m, n = A.shape
    _, l = b.shape

    model = gp.Model("LASSO")
    x = [
        [model.addVar(name=f"x{i}{j}", lb=-GRB.INFINITY) for j in range(l)]
        for i in range(n)
    ]
    t = model.addVar(name="t", lb=0)
    s = [model.addVar(name=f"s{i}", lb=0) for i in range(n)]

    # Set initial guess
    for i in range(n):
        for j in range(l):
            x[i][j].start = x0[i, j]

    # Set no log
    if opt is None or (opt is not None and not opt["log"]):
        model.setParam("OutputFlag", False)

    # Set constrain for t >= ||Ax - b||_F^2
    # Calculate Ax - b and store it in temp values
    temp_list = []
    for i in range(m):
        for j in range(l):
            temp = model.addVar(lb=-GRB.INFINITY, name=f"temp{i}{j}")
            temp_list.append(temp)
            model.addConstr(temp == sum([A[i, k] * x[k][j] for k in range(n)]) - b[i, j])

    quad_expr = gp.QuadExpr()
    quad_expr.addTerms([-1], [t], [t])
    quad_expr.addTerms([1] * m * l, temp_list, temp_list)
    model.addQConstr(quad_expr, sense=GRB.LESS_EQUAL, rhs=0, name="Ax-b")

    # Set constrains for ||x(i, :)||_2 <= s_i
    for i in range(n):
        quad_expr = gp.QuadExpr()
        quad_expr.addTerms([1] * l, x[i], x[i])
        quad_expr.addTerms(-1, s[i], s[i])
        model.addQConstr(quad_expr, sense=GRB.LESS_EQUAL, rhs=0, name=f"x{i}")

    # Set objective
    model.setObjective(0.5 * t + mu * gp.quicksum(s), GRB.MINIMIZE)

    model.optimize()

    # Get the result
    result = [[model.getVarByName(f"x{i}{j}").X for j in range(l)] for i in range(n)]
    result = np.array(result)

    obj_val = model.ObjVal

    return result, model.BarIterCount, {"obj": obj_val}


if __name__ == "__main__":
    run_method(gl_gurobi, plot=False)
