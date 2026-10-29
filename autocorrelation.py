import numpy as np

from operators import energy
from mcmc_utils import metropolis_ising
from matplotlib import pyplot as plt
from numba import njit
from global_variables import ALLOW_NUMBA_CACHING


##################################################################################################
# Integrated Autocorrelation Time (Tau_int)
##################################################################################################

@njit(cache = ALLOW_NUMBA_CACHING)
def tau_int_sokal(observables, c = 15.0):
    '''
    Computes the integrated autocorrelation time using a self-consistent windowing method.
        c: The windowing parameter. The bigger c, the longer the window.
        longer window: tau has less bias
        shorter window: tau has less variance
    NOTE: higher c values yield more accurate results but require more computation time.
    Returns:
        tau : float, the integrated autocorrelation time
        M   : int, the windowing parameter calculated self-consistently
    '''

    N = len(observables)
    var = np.var(observables)
    
    if var == 0.0:
        return 0.0, 0

    centered = observables - np.mean(observables)
    tau = 0.5

    for t in range(1, N):
        autocov_t = np.sum(centered[:N-t] * centered[t:]) / N
        tau += autocov_t / var

        if t >= c * tau:
            return tau, t

    return tau, N - 1


@njit(cache = ALLOW_NUMBA_CACHING)
def tau_int_blocking(observables, burn_in = None, c = 10.0):
    '''
    Computes the integrated autocorrelation time using the data blocking method (Flyvbjerg and Petersen).
    observables: ndarray of observable values along the Markov chain.
    burn_in: int, optional
        The number of initial samples to discard as burn-in.
        If None, burn_in is estimated as 20 tau_int computed with the Sokal method.
    '''

    N = len(observables)
    if N < 2:
        return 0.0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    if burn_in is None:
        tau0, _ = tau_int_sokal(observables, c = c)
        if not np.isfinite(tau0) or tau0 <= 0.0:
            burn_in = 0
        else:
            burn_in = int(20 * tau0)

    burn_in = max(0, int(burn_in))
    if burn_in >= N // 2:
        print(f"Warning: thermalization time ({burn_in}) is very large for the length of the chain ({N}).")

    observables = observables[burn_in:]
    N = len(observables)
    if N < 2:
        return 0.0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    # Return 0 if the number of samples is too small to compute tau_int
    var = np.var(observables)
    if not np.isfinite(var) or var <= 0.0:
        print(f"Warning: variance is non-positive ({var}) for the observable values after burn-in. Returning zeros.")
        return 0.0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    current_blocks = observables.copy()
    tau_estimates = []
    block_size = 1 # k
    Nb_array = [] # Number of blocks for each block size
    Nb = N # Number of blocks for the current block size (Nb_array[0])

    while Nb >= 2:
        # Estimation
        mean_blocks = np.mean(current_blocks)
        var_blocks = np.sum(np.power(current_blocks - mean_blocks, 2)) / (Nb - 1)
        tau_estimate = .5 * (block_size * (var_blocks / var) - 1)
        if not np.isfinite(tau_estimate):
            print(f"Warning: tau_estimate is not finite ({tau_estimate}) for the observable values after burn-in.")
            return 0.0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
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
        return 0.0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    for i in range(len(tau_estimates) - 1):
        if Nb_array[i] < 30:
            return float(tau_estimates[i]), np.array(Nb_array, dtype=np.int64), np.array(tau_estimates, dtype=np.float64)

        diff = abs(tau_estimates[i + 1] - tau_estimates[i])
        err = (tau_estimates[i] + 0.5) * np.sqrt(2.0 / (Nb_array[i] - 1)) 
        
        if diff < err:
            return float(tau_estimates[i]), np.array(Nb_array, dtype=np.int64), np.array(tau_estimates, dtype=np.float64)

    return float(tau_estimates[-1]), np.array(Nb_array, dtype=np.int64), np.array(tau_estimates, dtype=np.float64)



##################################################################################################
# Autocorrelation Function
##################################################################################################

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



if __name__ == "__main__":
    pass