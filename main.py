from tracemalloc import start

import numpy as np
import matplotlib.pyplot as plt
from ising import new_random_ising
from utils import maxwell_boltzmann_statistics, energy, metropolis_ising, magnetization
from graphics import animate, array_to_png
import time

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


def mcmc_sampling(N = 20, dim = 2, T = 1.0, sample_length = 10, initial_model = None, seed = 0):
    '''
    Samples N-dimensional Ising states using MCMC with the Metropolis algorithm.
    Parameters:
    - N: The size of the Ising model (N x N for 2D)
    - dim: The dimensionality of the Ising model
    - T: The temperature (K is set to 1 for simplicity)
    - sample_length: The number of samples to collect after burn-in
    - burn_in and thin are functions of the size of the model.
    '''
    # Good results with N = 20, T = 1.0
    if initial_model is None:
        np.random.seed(seed)
        m = new_random_ising(tuple([N] * dim))
    else:
        m = np.array(initial_model, copy = True)
    steps = np.power(N, dim + 1) + sample_length * np.power(N, dim)
    burn_in = int(np.power(N, dim + 1))
    thin = int(np.power(N, dim))

    models = metropolis_ising(m, T = T, steps = steps, burn_in = burn_in, seed = seed)

    # Thinning
    models = models[::thin]

    # img = array_to_png(models[-1].spins, filename = '') # visualize the last sampled configuration
    # img.save('tmp.png')

    return models

def magnetization_graph():
    '''
    Plots the magnetization of the Ising model as a function of temperature.
    '''

    N = 20
    dim = 3
    sample_length = 20

    temps = np.arange(.01, 7.0, .12)
    magns = np.zeros_like(temps)
    current_model = None
    for i, t in enumerate(temps):
        models = mcmc_sampling(
            N = N,
            dim = dim,
            T = t,
            sample_length = sample_length,
            initial_model = current_model,
        )
        current_model = models[-1]
        magns[i] = np.mean([magnetization(model) for model in models])
        print(f"Computed {(i + 1) / len(temps) * 100:.1f}%")
    
    plt.plot(temps, magns, marker = 'o')
    plt.xlabel('T (Temperature)')
    plt.ylabel('Magnetization')
    plt.title(f"N = {N}, dim = {dim}, sample_length = {sample_length}")
    plt.grid()
    plt.savefig('magnetization_graph.png')

def main():
    start = time.perf_counter()

    # anim_mcmc_1D()
    # anim_mcmc_2D()

    magnetization_graph()

    end = time.perf_counter()
    print(f"Elapsed = {end - start}s")


if __name__ == "__main__":
    main()