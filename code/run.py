import argparse

from gl_ADMM_dual import gl_ADMM_dual
from gl_ADMM_primal import gl_ADMM_primal
from gl_ALM_dual import gl_ALM_dual
from gl_FProxGD_primal import gl_FProxGD_primal
from gl_GD_primal import gl_GD_primal
from gl_ProxGD_primal import gl_ProxGD_primal
from gl_SGD_primal import gl_SGD_primal
from gl_cvx_gurobi import gl_cvx_gurobi
from gl_cvx_mosek import gl_cvx_mosek
from gl_gurobi import gl_gurobi
from gl_mosek import gl_mosek
from util import run_method

METHODS = {
    "cvx_gurobi": gl_cvx_gurobi,
    "cvx_mosek": gl_cvx_mosek,
    "gurobi": gl_gurobi,
    "mosek": gl_mosek,
    "ProxGD_primal": gl_ProxGD_primal,
    "FProxGD_primal": gl_FProxGD_primal,
    "SGD_primal": gl_SGD_primal,
    "GD_primal": gl_GD_primal,
    "ADMM_primal": gl_ADMM_primal,
    "ADMM_dual": gl_ADMM_dual,
    "ALM_dual": gl_ALM_dual
}


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--method", type=str, required=False, choices=METHODS.keys(), help="The method to use.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the result.")
    parser.add_argument("--log", action="store_true", help="Whether to show the log in the plot.")
    parser.add_argument("--log-scale", action="store_true", help="Whether to use log scale in the plot.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Whether to run the benchmark (this may take a long time).")
    parser.add_argument("--seed", type=int, required=False, help="The random seed.", default=97006855)


def run(method: str, plot: bool, log_scale: bool, benchmark: bool, log: bool, seed: int):
    gurobi_ans = run_method(gl_cvx_gurobi, False, seed=seed, output=False, log=False)
    mosek_ans = run_method(gl_cvx_mosek, False, seed=seed, output=False, log=False)
    if method is None:
        for method in METHODS:
            print(f"========= Method {method} =========")
            run_method(METHODS[method], plot, log_scale, benchmark, log, seed, gurobi_ans=gurobi_ans,
                       mosek_ans=mosek_ans)
            print()
    else:
        print(f"========= Method {method} =========")
        run_method(METHODS[method], plot, log_scale, benchmark, log, seed, gurobi_ans=gurobi_ans,
                   mosek_ans=mosek_ans)


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    method = args.method
    plot = args.plot
    log_scale = args.log_scale
    benchmark = args.benchmark
    log = args.log
    seed = args.seed
    run(method, plot, log_scale, benchmark, log, seed)


if __name__ == '__main__':
    main()
