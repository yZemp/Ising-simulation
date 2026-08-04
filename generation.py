import numpy as np
import h5py
import time
from datetime import timedelta
from typing import cast
from ising import new_random_ising
from mcmc_utils import metropolis_ising

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

            tmp_group = cast(h5py.Group, f[group_name])
            temps = cast(h5py.Dataset, tmp_group['temperatures'])[:]
            raw_data = cast(h5py.Dataset, tmp_group['raw_data'])
            
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
            raw_data = cast(h5py.Dataset, tmp_group.create_dataset(
            "raw_data", 
            shape = raw_data_shape, 
            maxshape = max_shape,
            dtype = np.int8,
            compression = "lzf",
            chunks = True
            ))
            
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

                raw_data[i] = models

                # print(f"Computed {(i + 1) / len(temps) * 100:.1f}%")
    
    end_time = time.time()
    print(f"Done. Elapsed: {timedelta(seconds = end_time - start_time)}.")

    return data_file

