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

    tau_fit_function = lambda T, tau_exp, K: K * np.exp(- T / tau_exp)

    fit_curve = None
    if len(fit_times) >= 3:
        tau0 = float(np.ptp(fit_taus)) if np.ptp(fit_taus) > 0 else float(fit_taus[0])
        K0 = float(np.mean(fit_taus))

        def chi2(tau_exp, K):
            return np.sum((fit_taus - tau_fit_function(fit_times, tau_exp, K)) ** 2)

        m = Minuit(chi2, tau_exp = tau0, K = K0)
        m.errordef = Minuit.LEAST_SQUARES
        m.limits["K"] = (0, None)
        m.migrad()

        fit_curve = tau_fit_function(fit_times, *m.values)
        print(f"Fit parameters: tau_exp = {m.values['tau_exp']:.2f}, K = {m.values['K']:.2f}")
    

    plt.plot(times, acs, label = f'Autocorrelation function')
    plt.plot(0, acs[0], label = f"Initial value: {acs[0]:.2f}", marker = 'x', markersize = 8, color = 'green')
    plt.plot(fit_times, fit_curve, label = f"Fit - valid: {m.valid}", color = "red")
    plt.xlabel('Time (steps)')
    plt.ylabel('Autocorrelation')
    # plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.title(f'Autocorrelation Function - N = {N}, dim = {dim}, T = {temperatures[T_index]:.2f}, tau_exp = {m.values["tau_exp"]:.2f}')
    plt.legend()
    plt.savefig(filename)
    plt.close()

    print(f"Autocorrelation graph saved to {filename}.")



###############################################################################
# Integrated Autocorrelation Time (Tau_int)

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

    
def tau_int_graph(N, dim, data_file, filename = "tau_int.png"):
    '''
    Plots the integrated autocorrelation time (tau_int) with respect to magnetization
    as a function of temperature.
    '''
    
    temperatures, filtered_data = read_data(data_file, N, dim)

    print(f"Filtered data shape: {filtered_data.shape}")

    observables = np.array([[magnetization(model) for model in models_at_T] for models_at_T in filtered_data])
    taus = np.zeros_like(temperatures)

    for i, T in enumerate(temperatures):
        print(f"Temperature: {T:.2f}")
        taus[i] = tau_int_sokal(observables[i])

    plt.scatter(temperatures, taus, marker = "x", label = r"$\tau_{int}$")
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
    data_to_save = np.column_stack((temperatures, taus))



###############################################################################
# Exponential Autocorrelation Time (Tau_exp)
# (For thermalization analysis)

def tau_exp_fit(observables):
    '''
    Computes the exponential autocorrelation time by fitting.
    '''

    times = np.arange(0, len(observables) // 2, 1)
    acs = np.zeros_like(times, dtype = float)

    for i, t in enumerate(times):
        acs[i] = autocorrelation(t, observables)

    # Fit autocorrelation(t) with a custom function
    mask = np.isfinite(times) & np.isfinite(acs)
    mask[0] = False  # Exclude t = 0 from the fit
    fit_times = times[mask]
    fit_taus = acs[mask]

    tau_fit_function = lambda T, tau_exp, K: K * np.exp(- T / tau_exp)

    if len(fit_times) >= 3:
        tau0 = float(np.ptp(fit_taus)) if np.ptp(fit_taus) > 0 else float(fit_taus[0])
        K0 = float(np.mean(fit_taus))

        def chi2(tau_exp, K):
            return np.sum((fit_taus - tau_fit_function(fit_times, tau_exp, K)) ** 2)

        m = Minuit(chi2, tau_exp = tau0, K = K0)
        m.errordef = Minuit.LEAST_SQUARES
        m.limits["K"] = (0, None)
        m.migrad()

    return m.values['tau_exp']


###############################################################################
# Magnetization study

def magnetization_graph(N, dim, filename = "magnetization.png"):
    '''
    Plots the magnetization of the Ising model as a function of temperature 
    given the raw data stored in an HDF5 file.
    '''

    data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/filtered_data"])
        if f"dim_{dim}_N_{N}/filtered_lengths" in file:
            filtered_lengths = np.array(file[f"dim_{dim}_N_{N}/filtered_lengths"])
        else:
            filtered_lengths = np.full(len(temperatures), filtered_data.shape[1], dtype = np.int32)

    valid_models = [filtered_data[i, :filtered_lengths[i]] for i in range(len(temperatures))]
    mean_magnetization = np.array([np.mean([magnetization(model) for model in models]) for models in valid_models])
    errors = np.array([np.std([magnetization(model) for model in models]) / np.sqrt(len(models)) for models in valid_models])
    # errors = np.array([np.std([magnetization(model) for model in models]) for models in valid_models])

    graph(temperatures,
          mean_magnetization,
          yerr = errors,
          xlabel = 'T (Temperature)',
          ylabel = 'Mean magnetization',
          title = f"(*) N = {N}, dim = {dim}",
          filename = filename
          )


def magnetization_tfixed_graph(N, dim, Tidx, data_file = "tmp.hdf5", filename = "magnetization.png"):
    '''
    Plots the magnetization of the Ising model as a function of time (steps) at a fixed temperature T 
    given the raw data stored in an HDF5 file.
    '''


    with h5py.File(data_file, "r") as file:
        T = (file[f"dim_{dim}_N_{N}/temperatures"])[Tidx]
        data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][Tidx])

    print(f"Filtered data shape: {data.shape}")
    # filtered_data = raw_data[:, 30_000::10_000]
    magnetizations = np.array([magnetization(model) for model in data])

    plt.plot(range(len(magnetizations)), magnetizations, label = f"T = {T:.2f}")
    plt.xlabel('Time (Steps)')
    plt.ylabel('Magnetization')
    plt.title(f"MC Magnetization - N = {N}, dim = {dim}")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def main():

    N = 15
    dim = 2

    data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"

    Tidx = 6
    
    with h5py.File(data_file, "r") as file:
        filtered_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"][Tidx])
    observables = np.array([magnetization(model) for model in filtered_data])

    start = time.perf_counter()

    #################################################################################
    # EXECUTION

    # for N, dim in [(5, 1), (10, 1), (20, 1), (30, 1), (50, 1), (70, 1), (100, 1), (150, 1), (200, 1), (250, 1), (300, 1), (500, 1)]:
    # for N, dim in [(100, 2)]:

    #     data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"

        # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
        
    magnetization_graph(N, dim, filename = "tmp_magnetization_reduced.png")
    # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
    # tau_int_1 = autocorrelation_graph(N, dim, data_file = data_file, filename = "tmp_autocorrelation.png", T_index = Tidx)
    # tau_int_graph(N, dim, data_file = data_file, filename = "tmp_tau.png")

    #################################################################################

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    main()