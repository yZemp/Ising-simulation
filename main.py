import argparse
import numpy as np
from ising import new_random_ising
from mcmc_utils import metropolis_ising
from graphics import animate
from datetime import timedelta
from io_utils import read_data

import time
import h5py


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


def mcmc_sampling(N = 20, dim = 2, T = 1.0, steps = 1000, initial_model = None, seed = 0):
    '''
    Samples N-dimensional Ising states using MCMC with the Metropolis algorithm.
    Parameters:
    - N: The size of the Ising model (N x N for 2D)
    - dim: The dimensionality of the Ising model
    - T: The temperature (K is set to 1 for simplicity)
    - steps: The total number of steps to run the MCMC
    - initial_model: The initial state of the Ising model
    - seed: The random seed for reproducibility

    NOTE: The number of steps is a placeholder for more sophisticated autocorrelation studies.
    Good steps number:
    - 1D: 100_000
    - 2D: 500_000
    - 3D: 2_000_000
    '''

    if initial_model is None:
        np.random.seed(seed)
        m = new_random_ising(tuple([N] * dim))
    else:
        m = np.array(initial_model, copy = True)

    # Avoiding burn_in and thinning
    models = metropolis_ising(m, T = T, steps = steps, burn_in = 0, seed = seed)

    return models


def simulate(N, dim, steps, data_file = "tmp.hdf5"):
    '''
    Computes the simulation of the Ising model varying the temperature and stores the results in an HDF5 file.
    
    NOTE:
        This is the full mcmc simulation at every temperature step.
        Thermalization and autocorreltions are not yet considered at this point.
    '''

    temps = np.arange(0.05, 7.0, .1)
    current_model = None

    model_shape = tuple([N] * dim)
    raw_data_shape = (len(temps), steps) + model_shape
    group_name = f"dim_{dim}_N_{N}"

    # TODO: Switch to packedbits model
    with h5py.File(data_file, "a") as f:

        # Extending an existing simulation
        if group_name in f:
            print("Data of an existing simulation was found.\nOverwriting temperatures and continuing previous simulation...")

            tmp_group = f[group_name]
            temps = tmp_group['temperatures'][:]
            raw_data = tmp_group['raw_data']
            
            current_steps = raw_data.shape[1]
            new_steps_total = current_steps + steps
            
            raw_data.resize((len(temps), new_steps_total) + model_shape)
            
            for i, t in enumerate(temps):
                current_model = raw_data[i, current_steps - 1]
                
                models = mcmc_sampling(
                    N = N,
                    dim = dim,
                    T = t,
                    steps = steps,
                    initial_model = current_model,
                )
                
                # Appende i nuovi modelli simulati nello slice appena creato
                raw_data[i, current_steps:new_steps_total] = models

                # print(f"Computed {(i + 1) / len(temps) * 100:.1f}%")

        # New simulation
        else:    
            max_shape = (len(temps), None) + model_shape

            tmp_group = f.create_group(group_name)
            tmp_group.create_dataset('temperatures', data = temps)
            tmp_group.create_dataset(
            "raw_data", 
            shape = raw_data_shape, 
            maxshape = max_shape,
            dtype = np.int8,
            compression = "lzf",
            chunks = True
            )
            
            for i, t in enumerate(temps):
                models = mcmc_sampling(
                    N = N,
                    dim = dim,
                    T = t,
                    steps = steps,
                    initial_model = current_model,
                )
                current_model = models[-1]

                f.get(f"dim_{dim}_N_{N}/raw_data")[i] = models

                # print(f"Computed {(i + 1) / len(temps) * 100:.1f}%")
    
    # print(f"Simulation completed. Data saved to {data_file}.")

    return data_file



def main(N = 100, dim = 2, steps = 1_000):

    data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"

    # start = time.perf_counter()

    # anim_mcmc_1D()
    # anim_mcmc_2D()
    simulate(N, dim, steps, data_file = data_file)
    # magnetization_graph(N, dim, steps, data_file = data_file, filename = "tmp.png")

    # end = time.perf_counter()
    # print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    args = parse_args()
    main(N = args.N, dim = args.dim, steps = args.steps)