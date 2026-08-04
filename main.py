import argparse
import numpy as np
import time

from ising import new_random_ising
from mcmc_utils import metropolis_ising
from graphics import animate
from datetime import timedelta
from process import filter_data, magnetization_bake
from generation import simulate

def positive_int(value):
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from exc

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")

    return parsed_value


def nonnegative_int(value):
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from exc

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")

    return parsed_value


def parse_args():
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description = "Run Ising model simulations.",
        formatter_class = HelpFormatter,
        epilog = (
            "Examples:\n"
            "  python main.py 200 1 1000\n"
            "  python main.py -N 200 -dim 1 -steps 1000\n\n"
            "Positional and flagged arguments are interchangeable. If both are provided, the flagged value wins."
        ),
    )
    parser.add_argument("N_pos", nargs = "?", type = positive_int, default = 100, metavar = "N", help = "Linear size of the lattice")
    parser.add_argument("dim_pos", nargs = "?", type = positive_int, default = 2, metavar = "dim", help = "Number of dimensions")
    parser.add_argument("steps_pos", nargs = "?", type = nonnegative_int, default = 1_000, metavar = "steps", help = "MCMC steps per temperature")

    parser.add_argument("-N", dest = "N", type = positive_int, default = argparse.SUPPRESS, help = "Linear size of the lattice")
    parser.add_argument("-dim", dest = "dim", type = positive_int, default = argparse.SUPPRESS, help = "Number of dimensions")
    parser.add_argument("-steps", dest = "steps", type = nonnegative_int, default = argparse.SUPPRESS, help = "MCMC steps per temperature")

    args = parser.parse_args()

    args.N = args.N if hasattr(args, "N") else args.N_pos
    args.dim = args.dim if hasattr(args, "dim") else args.dim_pos
    args.steps = args.steps if hasattr(args, "steps") else args.steps_pos

    args.N = args.N if args.N is not None else 100
    args.dim = args.dim if args.dim is not None else 2
    args.steps = args.steps if args.steps is not None else 1_000

    return args

def anim_mcmc_1D():
    np.random.seed(0)
    N = 50
    m = new_random_ising((N,))
    steps = N * N

    models = metropolis_ising(m, T = 2.3, steps = steps)

    print("MCMC completed.")
    if steps >= 500:
        models = models[::(steps // 500 + 1)]  # Limit to 500 frames for animation
    animate(models, fps = len(models), filename = 'tmp1D.gif')

def anim_mcmc_2D():
    np.random.seed(0)
    N = 30
    m = new_random_ising((N, N))
    steps = N * N * N

    models = metropolis_ising(m, T = 3.3, steps = steps)
    
    print("MCMC completed.")
    if steps >= 500:
        models = models[::(steps // 500 + 1)]  # Limit to 500 frames for animation
    animate(models, fps = len(models), filename = 'tmp2D.gif')


def main(N, dim, steps):

    data_file = r"tmp.hdf5".format(dim = dim, N = N)

    start = time.perf_counter()

    # anim_mcmc_1D()
    # anim_mcmc_2D()

    simulate(N, dim, steps, data_file = data_file)
    filter_data(N, dim, data_file = data_file)
    magnetization_bake(N, dim, data_file = data_file)

    end = time.perf_counter()
    print(f"Time elapsed since main.py was run = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    args = parse_args()
    main(N = args.N, dim = args.dim, steps = args.steps)