import numpy as np

from util import generate_data, group_lasso_loss
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def SDG_primal(x0: NDArray,
               A: NDArray,
               b: NDArray,
               mu: float,
               opt: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """
    Solve the group LASSO problem using Subgradient Descent.
    
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
    m, n = A.shape
    iter_count = 0
    x = x0
    
    step_size = 1 / np.linalg.norm(A, ord=2) ** 2
    f_last = np.inf
    tol = 1e-7 if opt is None or 'tol' not in opt else opt['tol']
    max_iter = 5000 if opt is None or 'maxiter' not in opt else opt['maxiter']
    log = True if opt is None or 'log' not in opt else opt['log']
    orig_mu = mu
    
    mu_list = list(reversed([(2 ** i) * mu for i in range(10)]))
    last_idx = len(mu_list) - 1
    
    if log:
        print(f"Start the subgradient descent algorithm with step size {step_size}.")
    
    obj_val_list = []
    grad_norm_list = []
    
    for k, mu in enumerate(mu_list):
        if iter_count > max_iter:
            break
        
        inner_iter = 0
        while True:
            # If log is enabled, print the log every 100 iterations.
            if log and iter_count % 100 == 0:
                print("Iteration:", iter_count, end=", ")
                print("Objective value:", group_lasso_loss(A, b, x, orig_mu))
            
            iter_count += 1
            inner_iter += 1
            
            if iter_count > max_iter or (inner_iter > 500 and k != last_idx):
                break
            
            # Compute the gradient.
            grad = A.T @ (A @ x - b)
            for i in range(n):
                norm = np.linalg.norm(x[i, :], ord=2)
                grad[i, :] += mu * x[i, :] / norm
            
            # Update the solution.
            x -= step_size * grad
            f_new = group_lasso_loss(A, b, x, mu)
            grad_norm = np.linalg.norm(grad, ord="fro")
            
            obj_val_list.append(f_new)
            grad_norm_list.append(grad_norm)
            
            if grad_norm * step_size < tol or abs(f_last - f_new) < tol:
                break
            
            f_last = f_new
    
    return x, iter_count, {'obj_val': obj_val_list, 'grad_norm': grad_norm_list}


def SDG_primal_test():
    import matplotlib.pyplot as plt
    
    A, b, x0 = generate_data()
    mu = 1e-2
    x, iter_count, out = SDG_primal(x0, A, b, mu, {'log': True})
    print(x)
    print(iter_count)
    print("Objective value: ", group_lasso_loss(A, b, x, mu))
    losses = out['obj_val']
    grad_norms = out['grad_norm']
    ax = plt.subplot(121)
    ax.plot(np.arange(len(losses)), losses)
    ax.set_yscale('log')
    ax.set_title("Objective value")
    ax = plt.subplot(122)
    ax.plot(np.arange(len(grad_norms)), grad_norms)
    ax.set_yscale('log')
    ax.set_title("Gradient norm")
    plt.show()


if __name__ == '__main__':
    SDG_primal_test()
