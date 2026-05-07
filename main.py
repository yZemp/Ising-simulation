import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import norm
from ising import IsingModel
from mcmc import metropolis, maxwell_boltzmann_statistics
from utils import metropolis_ising
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


def mcmc_sampling(beta = .3, sample_length = 10):
    # Good results with N = 20, beta = .3
    np.random.seed(0)
    N = 20
    m = IsingModel((N, N))
    steps = np.power(N, 3) + sample_length * np.power(N, 2)
    burn_in = int(np.power(N, 3))
    thin = int(np.power(N, 2))

    mbs = lambda epsilon: maxwell_boltzmann_statistics(epsilon, beta = beta)

    energies, models = metropolis_ising(mbs, m, steps = steps, burn_in = burn_in)
    # Thinning
    energies = energies[::thin]
    models = models[::thin]

    img = array_to_png(models[-1].spins, filename = '') # visualize the last sampled configuration
    img.save('final_sample.png')

    # plt.hist(np.sort(energies), density = True, label = "MCMC samples' energies")    
    # plt.legend()
    # plt.show()

    return energies, models

def main():
    # anim_mcmc_1D()
    # anim_mcmc_2D()
    energies, models = mcmc_sampling(beta = .3)


if __name__ == "__main__":
    main()