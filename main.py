import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import norm
from ising import IsingModel
from mcmc import metropolis, maxwell_boltzmann_statistics
from utils import energy, metropolis_ising, magnetization
from graphics import animate, array_to_png

def anim_mcmc_1D():
    np.random.seed(0)
    N = 50
    m = IsingModel(N)
    steps = int(np.power(N, 1.5))
    fps = steps / 10

    energies, models = metropolis_ising(maxwell_boltzmann_statistics, m, steps = steps, burn_in = 0)

    frames = [model.spins for model in models]
    animate(frames, fps = fps, filename = 'tmp.gif')


def anim_mcmc_2D():
    np.random.seed(0)
    N = 20
    m = IsingModel((N, N))
    steps = 2 * N * N
    fps = steps / 10

    energies, models = metropolis_ising(maxwell_boltzmann_statistics, m, steps = steps, burn_in = 0)

    frames = [model.spins for model in models]
    # frames = frames[::100] # thin the chain to reduce animation length
    animate(frames, fps = fps, filename = 'tmp.gif')


def mcmc_sampling(N = 20, dim = 2, beta = .3, sample_length = 10):
    '''
    Samples N-dimensional Ising states using MCMC with the Metropolis algorithm.
    Parameters:
    - N: The size of the Ising model (N x N for 2D)
    - dim: The dimensionality of the Ising model
    - beta: The inverse temperature (K is set to 1 for simplicity)
    - sample_length: The number of samples to collect after burn-in
    - burn_in and thin are functions of the size of the model.
    '''
    # Good results with N = 20, beta = .3
    np.random.seed(0)
    m = IsingModel(tuple([N] * dim))
    steps = np.power(N, dim + 1) + sample_length * np.power(N, dim)
    burn_in = int(np.power(N, dim + 1))
    thin = int(np.power(N, dim))

    mbs = lambda epsilon: maxwell_boltzmann_statistics(epsilon, beta = beta)

    energies, models = metropolis_ising(mbs, m, steps = steps, burn_in = burn_in)
    # Thinning
    energies = energies[::thin]
    models = models[::thin]

    # img = array_to_png(models[-1].spins, filename = '') # visualize the last sampled configuration
    # img.save('tmp.png')

    return energies, models

def magnetization_per_beta():
    # This doesn't work at all! :D

    betas = np.arange(0, 3.0, .1)
    magns = np.zeros_like(betas)
    for i, beta in enumerate(betas):
        _, models = mcmc_sampling(N = 100, dim = 1, beta = beta, sample_length = 20)
        magns[i] = np.mean([magnetization(model) for model in models])
    
    plt.plot(betas, magns, marker = 'o')
    plt.xlabel('Beta (Inverse Temperature)')
    plt.ylabel('Magnetization')
    plt.title("N = 100, dim = 1, sample_length = 20")
    plt.grid()
    plt.show()

def main():
    # anim_mcmc_1D()
    # anim_mcmc_2D()
    magnetization_per_beta()


if __name__ == "__main__":
    main()