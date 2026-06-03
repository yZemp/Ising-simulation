import numpy as np
import matplotlib.pyplot as plt
from ising import new_random_ising
from utils import metropolis_ising, magnetization
from graphics import animate, graph

import time
import h5py

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

    temps = np.arange(0.05, 7.0, .2)
    current_model = None

    model_shape = tuple([N] * dim)
    raw_data_shape = (len(temps), steps) + model_shape

    with h5py.File("tmp.hdf5", "w") as f:
        tmp_group = f.create_group(f"dim_{dim}_N_{N}")
        tmp_group.create_dataset('temperatures', data = temps)
        tmp_group.create_dataset(
        "raw_data", 
        shape = raw_data_shape, 
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

            print(f"Computed {(i + 1) / len(temps) * 100:.1f}%")
    
    print(f"Simulation completed. Data saved to {data_file}.")

    return data_file



def magnetization_graph(N, dim, steps, data_file = "tmp.hdf5", filename = "magnetization.png"):
    '''
    Plots the magnetization of the Ising model as a function of temperature 
    given the raw data stored in an HDF5 file.

    NOTE: This code arbitrarily filters the data for the sake of time.
    TODO: Implement proper thermalization and autocorrelation analysis.
    '''

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        raw_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"])

    filtered_data = raw_data[:, 30_000::4_000]
    mean_magnetization = np.array([np.mean([magnetization(model) for model in models]) for models in filtered_data])
    errors = np.array([np.std([magnetization(model) for model in models]) for models in filtered_data])


    graph(temperatures,
          mean_magnetization,
          yerr = errors,
          xlabel = 'T (Temperature)',
          ylabel = 'Mean magnetization',
          title = f"(*) N = {N}, dim = {dim}, sample_length = {steps}",
          filename = filename
          )



def main():
    N = 20
    dim = 2
    steps = 100_000

    data_file = "tmp.hdf5"

    # start = time.perf_counter()

    # anim_mcmc_1D()
    # anim_mcmc_2D()
    # file = simulate(N, dim, steps, data_file = data_file)
    magnetization_graph(N, dim, steps, data_file = data_file, filename = "tmp.png")

    # end = time.perf_counter()
    # print(f"Elapsed = {end - start}s")


if __name__ == "__main__":
    main()