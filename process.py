import numpy as np
import h5py

import time
from datetime import timedelta

from operators import magnetization
from graphics import graph
from matplotlib import pyplot as plt
from iminuit import Minuit
from io_utils import read_data
from numba import njit

ALLOW_NUMBA_CACHING = True

###############################################################################
# Autocorrelation

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


def autocorrelation_graph(N, dim, data_file = "tmp.hdf5", filename = "autocorrelation.png",T_index = 30):
    '''
    Plots the autocorrelation function of an observable O as a function of time (steps) 
    given the raw data stored in an HDF5 file.
    '''

    LEN = 10_000

    # Not using read_data() here to economize memory usage
    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][T_index, :LEN])

    print(f"Filtered data shape: {filtered_data.shape} (T = {temperatures[T_index]:.2f})")

    times = np.arange(0, LEN, 1)
    acs = np.zeros_like(times, dtype = float)
    observables = np.array([magnetization(model) for model in filtered_data])

    for i, t in enumerate(times):
        acs[i] = autocorrelation(t, observables)

    # Fit autocorrelation(t) with a custom function
    mask = np.isfinite(times) & np.isfinite(acs)
    mask[0] = False  # Exclude t = 0 from the fit
    fit_times = times[mask]
    fit_taus = acs[mask]

    tau_fit_function = lambda T, tau_int, K: K * np.exp(- T / tau_int)

    fit_curve = None
    if len(fit_times) >= 3:
        tau0 = float(np.ptp(fit_taus)) if np.ptp(fit_taus) > 0 else float(fit_taus[0])
        K0 = float(np.mean(fit_taus))

        def chi2(tau_int, K):
            return np.sum((fit_taus - tau_fit_function(fit_times, tau_int, K)) ** 2)

        m = Minuit(chi2, tau_int = tau0, K = K0)
        m.errordef = Minuit.LEAST_SQUARES
        m.limits["K"] = (0, None)
        m.migrad()

        fit_curve = tau_fit_function(fit_times, *m.values)
        print(f"Fit parameters: tau_int = {m.values['tau_int']:.2f}, K = {m.values['K']:.2f}")
    

    plt.plot(times, acs, label = f'Autocorrelation function')
    plt.plot(0, acs[0], label = f"Initial value: {acs[0]:.2f}", marker = 'x', markersize = 8, color = 'green')
    plt.plot(fit_times, fit_curve, label = f"Fit - valid: {m.valid}", color = "red")
    plt.xlabel('Time (steps)')
    plt.ylabel('Autocorrelation')
    # plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.title(f'Autocorrelation Function - N = {N}, dim = {dim}, T = {temperatures[T_index]:.2f}, tau_int = {m.values["tau_int"]:.2f}')
    plt.legend()
    plt.savefig(filename)
    plt.close()

    print(f"Autocorrelation graph saved to {filename}.")



###############################################################################
# Integrated Autocorrelation Time (Tau)

@njit(cache = ALLOW_NUMBA_CACHING)
def _tau(observables, max_lag):
    '''
    Computes the integrated autocorrelation time of a given Markov chain
    (with respect to a specific observable).

    Where the integrated autocorrelation time is defined as:
        tau = 0.5 + sum_{t=1}^{max_lag} autocorrelation(t)

    NOTE: This is meant to be used with the self-consistent windowing method.
    '''


    N = len(observables)
    var = np.var(observables)
    centered = observables - np.mean(observables)

    if var == 0:
        return 0.0

    tau = .5

    if max_lag > N:
        print(f"Warning: max_lag ({max_lag}) is greater than the number of observables ({N}).")

    for t in range(1, min(max_lag + 1, N)):
        # if N > 10_000 and t % 1000 == 0: print(f"Progress: {t / N:.2%}")
        autocov_t = np.sum(centered[:N-t] * centered[t:]) / N

        tau += autocov_t / var

    return tau

@njit(cache = ALLOW_NUMBA_CACHING)
def tau_scw(observables, c = 5.0):
    '''
    Computes the integrated autocorrelation time using the self-consistent windowing method.
    c: The windowing parameter, determining how many times tau is used as a window size.
    '''
    
    tau_int = 100
    W_old = c * tau_int + 1

    print("Computing tau (self-consistent windowing)...")

    counter = 0
    while True:
        W_new = c * tau_int
        if np.isclose(W_new, W_old, 1e-5) or counter > 200:
            break

        tau_int = _tau(observables, max_lag = int(W_new))
        W_old = W_new

        counter += 1
        # print(f"tau = {tau_int:.2f}, W = {W_new:.2f}")

    return tau_int


@njit(cache = ALLOW_NUMBA_CACHING)
def tau_int_sokal(observables, c = 5.0):
    '''
    Computes the integrated autocorrelation time using the self-consistent windowing method.
    Optimized as per Sokal/Madras's method.
    c: The windowing parameter, determining how many times tau is used as a window size.
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

    
def tau_graph(N, dim, data_file, filename = "tau.png"):
    '''
    Plots the integrated autocorrelation time (tau) with respect to magnetization
    as a function of temperature.

    Also fits the tau values with a descending exponential function to extract the correlation time.
    '''
    
    temperatures, raw_data = read_data(data_file, N, dim)
    filtered_data = raw_data[:, :10_000]

    print(f"Filtered data shape: {filtered_data.shape}")

    observables = np.array([[magnetization(model) for model in models_at_T] for models_at_T in filtered_data])
    taus = np.zeros_like(temperatures)

    for i, T in enumerate(temperatures):
        print(f"Temperature: {T:.2f}")
        taus[i] = tau_int_sokal(observables[i])

    plt.scatter(temperatures, taus, marker = "x", label = f"tau")
    plt.xlabel('Temperature')
    plt.ylabel('Tau')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.title(f'Tau - N = {N}, dim = {dim}')
    plt.legend()
    plt.savefig(filename)
    plt.close()

    
    # Save temperatures and taus to a text file
    data_to_save = np.column_stack((temperatures, taus))
    np.savetxt("dati.txt", data_to_save, header="temperature tau", fmt="%g")






###############################################################################
# Magnetization study

def magnetization_graph(N, dim, data_file = "tmp.hdf5", filename = "magnetization.png"):
    '''
    Plots the magnetization of the Ising model as a function of temperature 
    given the raw data stored in an HDF5 file.

    NOTE: This code arbitrarily filters the data for the sake of time.
    TODO: Implement proper thermalization and autocorrelation analysis.
    '''

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][:, 10_000::2_000])

    print(f"Filtered data shape: {filtered_data.shape}")
    # filtered_data = raw_data[:, 30_000::10_000]
    mean_magnetization = np.array([np.mean([magnetization(model) for model in models]) for models in filtered_data])
    errors = np.array([np.std([magnetization(model) for model in models]) for models in filtered_data])


    graph(temperatures,
          mean_magnetization,
          yerr = errors,
          xlabel = 'T (Temperature)',
          ylabel = 'Mean magnetization',
          title = f"(*) N = {N}, dim = {dim}",
          filename = filename
          )


def main():

    N = 100
    dim = 1

    data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"

    Tidx = 30
    
    with h5py.File(data_file, "r") as file:
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][Tidx, 10_000:30_000])
    observables = np.array([magnetization(model) for model in filtered_data])

    start = time.perf_counter()

    #################################################################################
    # EXECUTION

    # magnetization_graph(N, dim, data_file = data_file, filename = "tmp_magnetization.png")
    # tau_int_1 = autocorrelation_graph(N, dim, data_file = data_file, filename = "tmp_autocorrelation.png", T_index = Tidx)
    tau_graph(N, dim, data_file = data_file, filename = "tmp_tau.png")

    #################################################################################

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    main()