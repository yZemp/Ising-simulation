import argparse
import numpy as np
from ising import new_random_ising
from mcmc_utils import metropolis_ising
from graphics import animate
from datetime import timedelta
from io_utils import read_data
from operators import magnetization
from process import tau_exp_fit, tau_int_sokal

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




###############################################################################
# Data generation


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

    if dim == 1:
        temps = np.arange(0.1, 3.0, .05)
    if dim == 2:
        temps = np.arange(0.5, 5.0, .07)
    if dim == 3:
        temps = np.arange(1.0, 7.0, .1)
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
                
                # Append new models to the existing data
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



def filter_data(N, dim):
    '''
    Filters the raw data stored in an HDF5 producing an actual sample of Ising states.
    Saves the filtered data in the same HDF5 file.

    burn_in:
        TODO: implement using tau_exp or graphical method
    thinning:
        1 element every 2 * tau_int
        where tau_int is calculated with tau_int_sokal()
    '''

    data_file = f"dim_{dim}_N_{N}_data.hdf5"
    group_name = f"dim_{dim}_N_{N}"
    filtered_data_path = f"{group_name}/filtered_data"
    filtered_lengths_path = f"{group_name}/filtered_lengths"

    print(f"Filtering data for N = {N}, dim = {dim}...")

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"{group_name}/temperatures"])
        raw_data = file[f"{group_name}/raw_data"]
        model_shape = raw_data.shape[2:]
        raw_dtype = raw_data.dtype

    filtered_samples = []
    filtered_lengths = np.zeros(len(temperatures), dtype = np.int32)

    with h5py.File(data_file, "a") as file:
        if filtered_data_path in file:
            del file[filtered_data_path]

        if filtered_lengths_path in file:
            del file[filtered_lengths_path]

        # Sampling
        for i, T in enumerate(temperatures):
            print("----------------------------------------------------------------------")
            print(f"Filtering data {i} ({(i + 1) / len(temperatures) * 100:.1f}%)")

            raw_data = np.array(file[f"{group_name}/raw_data"][i])
            observables = np.array([magnetization(model) for model in raw_data])
            
            print("Computing tau...")
            tau_int = tau_int_sokal(observables, c = 15.0)
            print(f"Done.")    

            print("Filtering data...")
            if int(tau_int) > 0:
                # Using 20 * tau_int as burn-in and thinning every 2 * tau_int
                filtered_data = raw_data[int(20 * tau_int)::int(2 * tau_int)]
            else:
                # Using arbitrary thinning (should not matter)
                filtered_data = raw_data[::10_000]

            if filtered_data.size == 0:
                filtered_data = raw_data[-1:]
            print("Done.")

            print("Appending filtered data...")
            filtered_samples.append(filtered_data)
            filtered_lengths[i] = filtered_data.shape[0]
            print("Done.")
    
        max_length = int(np.max(filtered_lengths))
        filtered_dataset = file.create_dataset(
            filtered_data_path,
            shape = (len(temperatures), max_length) + model_shape,
            dtype = raw_dtype,
            compression = "lzf",
            chunks = True,
        )
        file.create_dataset(
            filtered_lengths_path,
            data = filtered_lengths,
        )

        for i, filtered_data in enumerate(filtered_samples):
            filtered_dataset[i, :filtered_data.shape[0]] = filtered_data

        filtered_dataset.attrs["lengths_dataset"] = "filtered_lengths"


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