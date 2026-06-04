import numpy as np
import matplotlib.pyplot as plt
from mcmc_utils import energy

import h5py


# TODO: Study the distribution of energy and see if they are maxwell boltzmann distributed.

if __name__ == "__main__":
    data_file = "tmp_mcmc.hdf5"
    N = 20
    dim = 2

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        raw_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"])

    filtered_data = raw_data[:, 30_000::1_000]
    energy_distributions = np.array([[energy(model) for model in models] for models in filtered_data])

    for i, T in enumerate(temperatures):
        if not (i % 5 == 0 or i == len(temperatures) - 1): continue
        color = (1.0, 1 - i / len(temperatures), i / len(temperatures))
        plt.hist(energy_distributions[i], density = False, color = color, histtype = 'step', label = f"T = {T:.2f}" if i == 0 or i == len(temperatures) - 1 else None)
    plt.title(f"Energy distributions at different temperatures (N = {N}, dim = {dim})")
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
