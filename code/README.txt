# Optimize Homework

This one is the same as the file `README.md` in the root directory, just to satisfy the requirement of the homework. You may need to rename this file to `README.md` to view it properly.

## Software Requirements

This program is tested on Python 3.12.7 and the files exclude `gl_cvx_mosek.py`, `gl_cvx_gurobi.py`, `gl_mosek.py`, `gl_gurobi.py` are tested on Python 3.13.1.

Packages used in this program are:

- NumPy 2.2.1, it is strongly recommended to use numpy compiled with MKL&TBB ([AUR Package](https://aur.archlinux.org/packages/python-numpy-mkl-tbb)) for better performance on Intel CPUs.
- Mosek 10.1.29
- Gurobi 12.0.0
- cvxpy 1.6.0
- matplotlib 3.9.3 (optional, only used for plotting)

## Usage

Simply run `python run.py` in folder `code` with default settings, this will run all optimization algorithms. Here are some extra command line arguments:
- `--method`: Determine which optimize algorithm to use.
- `--plot`: If this flag is set, the program will plot the result.
- `--seed`: Set the random seed for the program.
- `--log`: If this flag is set, the program will print the log of the optimization process.
- `--benchmark`: If this flag is set, the program will run the benchmark test. Note that by default, methods which use Mosek and Gurobi are disabled in the benchmark test because they are too slow. You can enable them by deleting the `#no_benchmark` tag in the function's docstring.

If you are still confused, you can run `python run.py --help` to see the help message.