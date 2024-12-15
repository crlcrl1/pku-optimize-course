import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Callable


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


def group_lasso_loss(A: NDArray, b: NDArray, x: NDArray, mu: float):
    """
    Compute the group LASSO loss.
    """
    return 0.5 * np.linalg.norm(A @ x - b, "fro") ** 2 + mu * np.sum(np.linalg.norm(x, axis=1))


def dual_loss(b: NDArray, y: NDArray) -> float:
    """
    Compute the dual loss.
    """
    return 0.5 * np.linalg.norm(y, "fro") ** 2 - np.sum(b * y)


def extract_config(opt: Dict, key: str, default=None):
    return default if opt is None or key not in opt else opt[key]


def test_and_plot(func: Callable, plot: bool = True, log_scale: bool = True, benchmark: bool = False):
    import matplotlib.pyplot as plt
    import time

    A, b, x0 = generate_data()
    mu = 1e-2
    x, iter_count, out = func(x0, A, b, mu, {'log': True})
    print(x)
    print(iter_count)
    print("Objective value: ", group_lasso_loss(A, b, x, mu))
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
        for _ in range(50):
            func(x0, A, b, mu, {'log': False})
        start = time.time()
        for _ in range(500):
            func(x0, A, b, mu, {'log': False})
        end = time.time()
        print(f"Benchmark: {(end - start) * 2} ms")
