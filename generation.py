import numpy as np
import h5py
import time
from datetime import timedelta
from ising import new_random_ising
from mcmc_utils import metropolis_ising
from operators import magnetization
from autocorrelation import tau_exp_fit, tau_int_sokal


###################################################################################
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

    models = metropolis_ising(m, T = T, steps = steps, seed = seed)

    return models


def simulate(N, dim, steps, data_file = "tmp.hdf5"):
    '''
    Computes the simulation of the Ising model varying the temperature and stores the results in an HDF5 file.
    
    NOTE:
        This is the full mcmc simulation at every temperature step.
        Thermalization and autocorreltions are not yet considered at this point.
    '''

    start_time = time.time()
    print(f"Generating data: Ising model MCMC for N = {N}, dim = {dim}, steps = {steps}...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

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

    # TODO: Possibly switch to packedbits model
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

                print(f"Computing MCMC for T = {t:.2f}... ({i * 100 / len(temps):.1f}%)")
                
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
                print(f"Computing MCMC for T = {t:.2f}... ({i * 100 / len(temps):.1f}%)")

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
    
    end_time = time.time()
    print(f"Done. Elapsed: {timedelta(seconds = end_time - start_time)}.")

    return data_file


###################################################################################
# Data filtering

def filter_data(N, dim, data_file = "tmp.hdf5"):
    '''
    Filters the raw data stored in an HDF5 producing an actual sample of Ising states.
    Saves the filtered data in the same HDF5 file.

    burn_in:
        TODO: implement using tau_exp or graphical method
    thinning:
        1 element every 2 * tau_int
        where tau_int is calculated with tau_int_sokal()
    '''

    if data_file is None:
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
            print(f"Filtering data {i} ({i / len(temperatures) * 100:.1f}%)")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

            raw_data = np.array(file[f"{group_name}/raw_data"][i])
            observables = np.array([magnetization(model) for model in raw_data])
            
            tau_int = tau_int_sokal(observables, c = 15.0)

            if int(tau_int) > 0:
                # Using 20 * tau_int as burn-in and thinning every 2 * tau_int
                filtered_data = raw_data[int(20 * tau_int)::int(2 * tau_int)]
            else:
                # Using arbitrary thinning (should not matter)
                filtered_data = raw_data[::10_000]

            if filtered_data.size == 0:
                filtered_data = raw_data[-1:]

            filtered_samples.append(filtered_data)
            filtered_lengths[i] = filtered_data.shape[0]
    
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
