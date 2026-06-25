import numpy as np
import matplotlib.pyplot as plt
import h5py
from mcmc_utils import energy
import time
from datetime import timedelta
from iminuit import Minuit

# NOTE:
# I don't really know what to do with the informations I got with this script

def energy_distributions(N, dim, data_file = "tmp_mcmc.hdf5", filename = "energy_distributions.png", distrib_freq = 1 / 5):

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][:, 30_000::1_000])

    energy_distributions = np.array([[energy(model) for model in models] for models in filtered_data])

    for i, T in enumerate(temperatures):
        if not (i % int(1 / distrib_freq) == 0 or i == len(temperatures) - 1): continue
        color = (1.0, 1 - i / len(temperatures), i / len(temperatures))
        plt.hist(energy_distributions[i], density = False, color = color, histtype = 'step', label = f"T = {T:.2f}" if i == 0 or i == len(temperatures) - 1 else None)
    plt.title(f"Energy distributions at different temperatures (N = {N}, dim = {dim})")
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(filename)


def fit_distribution(N, dim, data_file = "tmp_mcmc.hdf5", filename = "fit_distribution.png", T_index = 30):

    with h5py.File(data_file, "r") as file:
        temperature = float(file[f"dim_{dim}_N_{N}/temperatures"][T_index])
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][T_index, 30_000::100])

    energy_distribution = np.array([energy(model) for model in filtered_data])

    # Build a density histogram and fit PDF parameters by chi2 minimization.
    counts, bin_edges = np.histogram(energy_distribution, bins="auto", density=True)
    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_samples = energy_distribution.size

    density = counts / (n_samples * widths)
    density_err = np.sqrt(np.maximum(counts, 1.0)) / (n_samples * widths)

    def normal_dist(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    def maxwell_shifted_dist(x, loc, scale):
        z = (x - loc) / scale
        out = np.zeros_like(x, dtype=float)
        mask = z >= 0.0
        out[mask] = np.sqrt(2.0 / np.pi) * (z[mask] ** 2 / scale) * np.exp(-0.5 * z[mask] ** 2)
        return out

    def chi2_normal(mu, sigma):
        model = normal_dist(centers, mu, sigma)
        return np.sum(((density - model) / density_err) ** 2)

    def chi2_maxwell(loc, scale):
        model = maxwell_shifted_dist(centers, loc, scale)
        return np.sum(((density - model) / density_err) ** 2)

    mu_init = float(np.mean(energy_distribution))
    sigma_init = float(np.std(energy_distribution))
    loc_init = float(np.min(energy_distribution))
    scale_init = max(sigma_init, 1e-6)

    m = Minuit(chi2_normal, mu=mu_init, sigma=sigma_init)
    m.errordef = Minuit.LEAST_SQUARES
    m.limits["sigma"] = (1e-8, None)
    m.migrad()

    m2 = Minuit(chi2_maxwell, loc=loc_init, scale=scale_init)
    m2.errordef = Minuit.LEAST_SQUARES
    m2.limits["scale"] = (1e-8, None)
    m2.migrad()

    plt.hist(energy_distribution, bins=bin_edges, density=True, alpha=0.35, label = f"T = {temperature:.2f}")

    x = np.linspace(min(energy_distribution), max(energy_distribution), 100)
    plt.plot(x, normal_dist(x, m.values["mu"], m.values["sigma"]), label = f"Normal fit (mu = {m.values['mu']:.2f}, sigma = {m.values['sigma']:.2f}, valid = {m.valid})", color = "red")
    plt.plot(x, maxwell_shifted_dist(x, m2.values["loc"], m2.values["scale"]), label = f"Shifted Maxwell fit (loc = {m2.values['loc']:.2f}, scale = {m2.values['scale']:.2f}, valid = {m2.valid})", color = "green")

    plt.title(f"Energy distribution at temperature (N = {N}, dim = {dim})")
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":

    start = time.perf_counter()

    N = 20
    dim = 2
    data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"

    print(f"Reading data from {data_file}...")
    # energy_distributions(N, dim, data_file)
    fit_distribution(N, dim, data_file, T_index = 10)

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")