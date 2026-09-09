from cmath import tau

import numpy as np
import h5py
from typing import cast

from operators import magnetization
from graphics import graph
from matplotlib import pyplot as plt
from iminuit import Minuit
from hdf5_utils import read_data
from scipy.optimize import curve_fit
from numba import njit
from global_variables import ALLOW_NUMBA_CACHING


###############################################################################
# Integrated Autocorrelation Time (Tau_int)
# (For decorrelating samples)

@njit(cache = ALLOW_NUMBA_CACHING)
def tau_int_sokal(observables, c = 15.0):
    '''
    Computes the integrated autocorrelation time using the self-consistent windowing method
    optimized as per Sokal's method.
    c: The windowing parameter, determining how many times tau is used as a window size.
    NOTE: higher c values yield more accurate results but require more computation time.
    '''

    N = len(observables)
    var = np.var(observables)
    
    if var == 0.0:
        return 0.0

    centered = observables - np.mean(observables)
    tau = 0.5

    for t in range(1, N):
        autocov_t = np.sum(centered[:N-t] * centered[t:]) / N
        tau += autocov_t / var

        if t >= c * tau:
            return tau

    return tau


@njit(cache = ALLOW_NUMBA_CACHING)
def tau_int_blocking(observables, burn_in = None):
    '''
    Computes the integrated autocorrelation time using the data blocking method.
    observables: ndarray of observable values along the Markov chain.
    burn_in: int, optional
        The number of initial samples to discard as burn-in.
        if None, burn_in is estimated as 20 tau_int computed with the Sokal method.
    '''

    N = len(observables)
    if N < 2:
        return 0.0

    if burn_in is None:
        tau0 = tau_int_sokal(observables, c = 5.0)
        if not np.isfinite(tau0) or tau0 <= 0.0:
            burn_in = 0
        else:
            burn_in = int(20 * tau0)

    burn_in = max(0, min(int(burn_in), N // 2))

    observables = observables[burn_in:]
    N = len(observables)
    if N < 2:
        return 0.0

    # Return 0 if the number of samples is too small to compute tau_int
    var = np.var(observables)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0

    current_blocks = observables.copy()
    tau_estimates = []
    block_size = 1
    Nb_array = [] # Number of blocks for each block size
    Nb = N # Number of blocks for the current block size (Nb_array[0])

    while Nb >= 2:
        # Estimation
        mean_blocks = np.mean(current_blocks)
        var_blocks = np.sum(np.power(current_blocks - mean_blocks, 2)) / (Nb - 1)
        tau_estimate = .5 * (block_size * (var_blocks / var) - 1)
        if not np.isfinite(tau_estimate):
            return 0.0
        tau_estimates.append(tau_estimate)
        Nb_array.append(Nb)

        # Blocking
        if Nb % 2 != 0:
            current_blocks = current_blocks[:-1]  # Discard the last block if odd number of blocks
            Nb -= 1
        Nb //= 2
        current_blocks = 0.5 * (current_blocks[::2] + current_blocks[1::2])
        block_size *= 2


    if len(tau_estimates) == 0:
        return 0.0

    # Locating plateau in the tau_estimates to determine the final tau_int value
    for i in range(len(tau_estimates) - 1):
        if Nb_array[i] < 30:
            return float(tau_estimates[i])
            
        diff = abs(tau_estimates[i + 1] - tau_estimates[i])
        err = (tau_estimates[i] + 0.5) * np.sqrt(2.0 / (Nb_array[i] - 1)) 
        
        if diff < err:
            return float(tau_estimates[i])
            
    return float(tau_estimates[-1])


def tau_int_graph(N, dim, data_file, filename = "tau_int.png"):
    '''
    Plots the integrated autocorrelation time (tau_int) with respect to magnetization
    as a function of temperature.
    '''
    
    temperatures, data = read_data(data_file, N, dim)

    print(f"Filtered data shape: {data.shape}")

    observables = np.array([[magnetization(model) for model in models_at_T] for models_at_T in data])
    taus_sokal = np.zeros_like(temperatures)
    taus_blocking = np.zeros_like(temperatures)

    for i, T in enumerate(temperatures):
        print(f"Temperature: {T:.2f}")
        try:
            taus_sokal[i] = tau_int_sokal(observables[i], c = 5.0)
            taus_blocking[i] = tau_int_blocking(observables[i])
        except Exception as e:
            print(f"Warning: failed at T = {T:.2f}: {type(e).__name__}: {e}")
            taus_sokal[i] = np.nan
            taus_blocking[i] = np.nan

    plt.scatter(temperatures, taus_sokal, marker = "x", label = r"$\tau_{int}-sokal$")
    plt.scatter(temperatures, taus_blocking, marker = "o", label = r"$\tau_{int}-blocking$")
    plt.xlabel('Temperature')
    plt.ylabel(r"$\tau_{int}$")
    # plt.yscale('log')
    # plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.title(r"$\tau_{int}$" + f" - N = {N}, dim = {dim}")
    plt.legend()
    plt.savefig(filename)
    plt.close()

    
    # Save temperatures and taus to a text file
    data_to_save = np.column_stack((temperatures, taus_sokal, taus_blocking))




###############################################################################
# Exponential Autocorrelation Time (Tau_exp)
# (For thermalization)

@njit(cache = ALLOW_NUMBA_CACHING)
def autocorrelation(t, observables: np.ndarray):
    '''
    Computes the autocorrelation function's value given:
    - A time (step) t
    - An array of observable values along the Markov chain
    '''

    N = len(observables)
    if t >= N or t < 0:
        return 0.0
    
    var = np.var(observables)

    if var == 0:
        return 0.0

    centered = observables - np.mean(observables)
    autocov = np.sum(centered[:N - t] * centered[t:]) / N

    return autocov / var

def compute_tau_exp(observables, t_min, t_max):
    '''
    Computes the exponential autocorrelation time (tau_exp) by fitting an exponential decay to the ACF calculated in the time domain.
    t_min: Lower limit of the region where the ACF is approximately exponential.
    t_max: Upper limit before noise dominates (e.g., when ACF drops below 0.05).
    '''

    N = len(observables)
    var = np.var(observables)
    
    if var == 0.0:
        return 0.0

    centered = observables - np.mean(observables)
    acf = np.zeros(t_max)
    
    for t in range(t_max):
        acf[t] = np.sum(centered[:N-t] * centered[t:]) / (N * var)
        
    t_data = np.arange(t_min, t_max)
    y_data = acf[t_min:t_max]
    
    p0 = (1.0, t_max / 3.0) 

    def _exp_decay(t, A, tau):
        return A * np.exp(-t / tau)
    
    try:
        popt, pcov = curve_fit(_exp_decay, t_data, y_data, p0=p0)
        A_fit, tau_exp = popt

        plt.plot(t_data, _exp_decay(t_data, *popt), color='black', label=f'$\\tau_{{exp}}$={tau_exp:.2f}')

        return tau_exp
    except RuntimeError:
        return float('nan')


def autocorrelation_graph(N, dim, data_file = "tmp.hdf5", filename = "autocorrelation.png", T_index = 30):
    '''
    Plots the autocorrelation function of an observable O as a function of time (steps) 
    given the raw data stored in an HDF5 file.
    '''

    LEN = 10_000

    # Not using read_data() here to economize memory usage
    with h5py.File(data_file, "r") as file:
        temperatures = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/temperatures"]))
        filtered_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"])[T_index, :LEN])

    print(f"Filtered data shape: {filtered_data.shape} (T = {temperatures[T_index]:.2f})")

    times = np.arange(0, LEN, 1)
    acs = np.zeros_like(times, dtype = float)
    observables = np.array([magnetization(model) for model in filtered_data])

    for i, t in enumerate(times):
        acs[i] = autocorrelation(t, observables)

    plt.figure(figsize=(10, 6))
    plt.plot(times, acs, label='Autocorrelation function', color=(.3, 1, 0))
    plt.plot(0, acs[0], label=f"Initial value: {acs[0]:.2f}", marker='x', markersize=8, color='black')

    plt.xlabel('Time (steps)')
    plt.ylabel('Autocorrelation')
    
    # plt.yscale('symlog', linthresh=1e-3)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    tau_int = int(tau_int_sokal(observables, c = 20.0))
    tmin = 2 * tau_int
    tmax = 8 * tau_int

    compute_tau = compute_tau_exp(observables, t_min = tmin, t_max = tmax)
    # plt.vlines(tmin, ymin = -.2, ymax = 1, colors = (.3, 0, 1), linestyles = '--', alpha = 0.7)
    # plt.vlines(tmax, ymin = -.2, ymax = 1, colors = (.3, 0, 1), linestyles = '--', alpha = 0.7)
    plt.hlines([], 0, 0, alpha = 0, label = f"$\\tau_{{int}}$ = {tmin / 2:.2f}")

    plt.title(f'Autocorrelation Function - N = {N}, dim = {dim}, T = {temperatures[T_index]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Autocorrelation graph saved to {filename}.")


if __name__ == "__main__":
    N = 50
    dim = 1
    data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"
    autocorrelation_graph(N, dim, data_file, filename = "autocorrelation.png", T_index = 30) 

    # with h5py.File(data_file, "r") as file:
    #     temperatures = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/temperatures"]))
    #     filtered_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"])[30, :100_000])
    # observables = np.array([magnetization(model) for model in filtered_data])
    # print(tau_int_sokal(observables, c = 20.0))
    # tau_int_graph(N, dim, data_file, filename = "tau_int.png")