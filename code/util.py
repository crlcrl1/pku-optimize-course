import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Callable


def generate_data(seed: int = 97006855) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
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
    return A, b, x0, u


def group_lasso_loss(A: NDArray, b: NDArray, x: NDArray, mu: float):
    """
    Compute the group LASSO loss.
    """
    return 0.5 * np.linalg.norm(A @ x - b, "fro") ** 2 + mu * np.sum(np.linalg.norm(x, axis=1))


def extract_config(opt: Dict, key: str, default=None):
    return default if opt is None or key not in opt else opt[key]


def bench_mark(func: Callable) -> float:
    import time

    A, b, x0, u = generate_data()
    mu = 1e-2
    for _ in range(50):
        func(x0, A, b, mu, {'log': False})
    start = time.time()
    for _ in range(500):
        func(x0, A, b, mu, {'log': False})
    end = time.time()
    return (end - start) * 2


def sparisity(x: NDArray) -> float:
    elem_num = 1
    for i in x.shape:
        elem_num *= i
    max_elem = np.max(np.abs(x))
    return np.sum(np.abs(x) > (1e-6 * max_elem)) / elem_num


def run_method(func: Callable, plot: bool = True, log_scale: bool = True, benchmark: bool = False,
               log: bool = True, seed: int = 97006855, output: bool = True, **kwargs) -> NDArray:
    import matplotlib.pyplot as plt

    A, b, x0, u = generate_data(seed)
    mu = 1e-2
    x, iter_count, out = func(x0, A, b, mu, {'log': log})
    if output:
        print(f"Objective value: {group_lasso_loss(A, b, x, mu):.8f}")
        gurobi_ans = kwargs.get('gurobi_ans', None)
        mosek_ans = kwargs.get('mosek_ans', None)
        if mosek_ans is not None:
            print(f"Error mosek: {np.linalg.norm(x - mosek_ans, 'fro') / (1 + np.linalg.norm(mosek_ans, 'fro')):.6e}")
        if gurobi_ans is not None:
            print(
                f"Error gurobi: {np.linalg.norm(x - gurobi_ans, 'fro') / (1 + np.linalg.norm(gurobi_ans, 'fro')):.6e}")
        print(f"Error exact: {np.linalg.norm(x - u, 'fro') / (1 + np.linalg.norm(u, 'fro')):.6e}")
        print(f"Sparsity: {sparisity(x):.4f}")
        print(f"Iteration count: {iter_count}")

    if plot:
        losses = out['obj_val']
        ax = plt.subplot(121)
        ax.plot(np.arange(len(losses)), losses)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title("Objective value")
        data = out['grad_norm'] if 'grad_norm' in out else out['dual_gap']
        ax = plt.subplot(122)
        ax.plot(np.arange(len(data)), data)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title("Gradient norm" if 'grad_norm' in out else "Dual gap")
        plt.show()

    if benchmark:
        doc: str = func.__doc__
        skip_bench = doc.find("#no_benchmark") != -1

        if skip_bench:
            print(f"Benchmark for {func.__name__} is skipped.")
            return x

        avg_time = bench_mark(func)
        print(f"Benchmark: {avg_time:.4f} ms")

    return x
