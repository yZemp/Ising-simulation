import argparse
import numpy as np
from ising import new_random_ising
from mcmc_utils import metropolis_ising
from graphics import animate
from datetime import timedelta
from io_utils import read_data
from operators import magnetization
from generation import simulate, filter_data

import time
import h5py
from matplotlib import pyplot as plt

def positive_int(value):
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from exc

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")

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
    parser.add_argument("steps_pos", nargs = "?", type = positive_int, default = 1_000, metavar = "steps", help = "MCMC steps per temperature")

    parser.add_argument("-N", dest = "N", type = positive_int, default = argparse.SUPPRESS, help = "Linear size of the lattice")
    parser.add_argument("-dim", dest = "dim", type = positive_int, default = argparse.SUPPRESS, help = "Number of dimensions")
    parser.add_argument("-steps", dest = "steps", type = positive_int, default = argparse.SUPPRESS, help = "MCMC steps per temperature")

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
    steps = int(np.power(N, 1.5))
    fps = steps / 10

    models = metropolis_ising(m, T = 1.0, steps = steps, burn_in = 0)
    print("MCMC completed.")

    animate(models, fps = fps, filename = 'tmp.gif')


def anim_mcmc_2D():
    np.random.seed(0)
    N = 20
    m = new_random_ising((N, N))
    steps = 2 * N * N
    fps = steps / 10

    models = metropolis_ising(m, T = 1.0, steps = steps, burn_in = 0)
    
    print("MCMC completed.")
    if steps >= 500:
        models = models[::(steps // 500 + 1)]  # Limit to 500 frames for animation
    animate(models, fps = len(models), filename = 'tmp.gif')


def main(N, dim, steps):

    data_file = "simulations_data/" + f"dim_{dim}_N_{N}" + "_data.hdf5"

    start = time.perf_counter()

    # anim_mcmc_1D()
    # anim_mcmc_2D()
    # simulate(N, dim, steps, data_file = data_file)
    # magnetization_graph(N, dim, steps, data_file = data_file, filename = "tmp.png")

    filter_data(N, dim)

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    args = parse_args()
    main(N = args.N, dim = args.dim, steps = args.steps)