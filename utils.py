import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from ising import new_random_ising
from numba import njit

ALLOW_NUMBA_CACHING = True

@njit(cache = ALLOW_NUMBA_CACHING)
def maxwell_boltzmann_statistics(epsilon, beta = 1.):
    return np.exp(- (beta * epsilon))

@njit(cache = ALLOW_NUMBA_CACHING)
def _metropolis_ising_1d(model: np.ndarray, T: float, steps: int, burn_in: int = 0):
    '''1D version of Metropolis-Hastings for Ising model (Numba-compiled).'''
    curr_model = np.copy(model)
    models_samples = np.empty((steps,) + curr_model.shape, dtype=curr_model.dtype)
    N = curr_model.shape[0]
    
    for i in range(steps):
        idx0 = np.random.randint(0, N)
        spin = curr_model[idx0]
        forward = (idx0 + 1) % N
        backward = (idx0 - 1) % N
        deltaE = 2.0 * spin * (curr_model[forward] + curr_model[backward])
        
        accept_probability = np.minimum(1.0, maxwell_boltzmann_statistics(deltaE, beta=1.0 / T))
        if np.random.random() <= accept_probability:
            curr_model[idx0] = -curr_model[idx0]
        
        models_samples[i] = curr_model
    
    return models_samples[burn_in:]


@njit(cache = ALLOW_NUMBA_CACHING)
def _metropolis_ising_2d(model: np.ndarray, T: float, steps: int, burn_in: int = 0):
    '''2D version of Metropolis-Hastings for Ising model (Numba-compiled).'''
    curr_model = np.copy(model)
    models_samples = np.empty((steps,) + curr_model.shape, dtype=curr_model.dtype)
    N1, N2 = curr_model.shape[0], curr_model.shape[1]
    
    for i in range(steps):
        idx0 = np.random.randint(0, N1)
        idx1 = np.random.randint(0, N2)
        spin = curr_model[idx0, idx1]
        
        i_fwd = (idx0 + 1) % N1
        i_bwd = (idx0 - 1) % N1
        j_fwd = (idx1 + 1) % N2
        j_bwd = (idx1 - 1) % N2
        neighbors_sum = curr_model[i_fwd, idx1] + curr_model[i_bwd, idx1] + curr_model[idx0, j_fwd] + curr_model[idx0, j_bwd]
        deltaE = 2.0 * spin * neighbors_sum
        
        accept_probability = np.minimum(1.0, maxwell_boltzmann_statistics(deltaE, beta=1.0 / T))
        if np.random.random() <= accept_probability:
            curr_model[idx0, idx1] = -curr_model[idx0, idx1]
        
        models_samples[i] = curr_model
    
    return models_samples[burn_in:]


@njit(cache = ALLOW_NUMBA_CACHING)
def _metropolis_ising_3d(model: np.ndarray, T: float, steps: int, burn_in: int = 0):
    '''3D version of Metropolis-Hastings for Ising model (Numba-compiled).'''
    curr_model = np.copy(model)
    models_samples = np.empty((steps,) + curr_model.shape, dtype=curr_model.dtype)
    N1, N2, N3 = curr_model.shape[0], curr_model.shape[1], curr_model.shape[2]
    
    for i in range(steps):
        idx0 = np.random.randint(0, N1)
        idx1 = np.random.randint(0, N2)
        idx2 = np.random.randint(0, N3)
        spin = curr_model[idx0, idx1, idx2]
        
        i_fwd = (idx0 + 1) % N1
        i_bwd = (idx0 - 1) % N1
        j_fwd = (idx1 + 1) % N2
        j_bwd = (idx1 - 1) % N2
        k_fwd = (idx2 + 1) % N3
        k_bwd = (idx2 - 1) % N3
        neighbors_sum = (curr_model[i_fwd, idx1, idx2] + curr_model[i_bwd, idx1, idx2] + 
                         curr_model[idx0, j_fwd, idx2] + curr_model[idx0, j_bwd, idx2] + 
                         curr_model[idx0, idx1, k_fwd] + curr_model[idx0, idx1, k_bwd])
        deltaE = 2.0 * spin * neighbors_sum
        
        accept_probability = np.minimum(1.0, maxwell_boltzmann_statistics(deltaE, beta=1.0 / T))
        if np.random.random() <= accept_probability:
            curr_model[idx0, idx1, idx2] = -curr_model[idx0, idx1, idx2]
        
        models_samples[i] = curr_model
    
    return models_samples[burn_in:]


def metropolis_ising(model: np.ndarray, T: float, steps: int, min = 0, max = 100_000, burn_in = 0, seed = 0):
    '''
    Given:
        Target probability distribution function
        Initial state of an ising model
        Number of steps to take (in the markov process)
    
    Returns:
        Array of ising model configurations
        which energies are distributed as the maxwell-boltzmann pdf at temperature T.

    Note:
        This function routes to specialized Numba-compiled versions
        based on the dimensionality of the input model for performance.
    '''
    if seed != 0:
        np.random.seed(seed)
    
    if model.ndim == 1:
        return _metropolis_ising_1d(model, T, steps, burn_in)
    elif model.ndim == 2:
        return _metropolis_ising_2d(model, T, steps, burn_in)
    elif model.ndim == 3:
        return _metropolis_ising_3d(model, T, steps, burn_in)
    else:
        raise ValueError("metropolis_ising supports 1D, 2D, and 3D models only")


    return models_samples[burn_in:]


#####################################################################
# Operators that act on models configurations
#####################################################################

# model = ndarray of spins, J = 1 for simplicity everywhere

@njit(cache = ALLOW_NUMBA_CACHING)
def energy(model: np.ndarray) -> float:
    '''
    Given an Ising model, compute its energy.
    '''

    interaction_energy = 0.0
    for axis in range(model.ndim):
        interaction_energy -= np.sum(model * np.roll(model, 1, axis = axis))
    
    return interaction_energy

@njit(cache = ALLOW_NUMBA_CACHING)
def _delta_energy_1d(model: np.ndarray, index: int) -> float:
    '''Delta energy for 1D lattice (Numba-compiled).'''
    N = model.shape[0]
    spin = model[index]
    forward = (index + 1) % N
    backward = (index - 1) % N
    return 2.0 * spin * (model[forward] + model[backward])


@njit(cache = ALLOW_NUMBA_CACHING)
def _delta_energy_2d(model: np.ndarray, i: int, j: int) -> float:
    '''Delta energy for 2D lattice (Numba-compiled).'''
    N1 = model.shape[0]
    N2 = model.shape[1]
    spin = model[i, j]
    
    i_fwd = (i + 1) % N1
    i_bwd = (i - 1) % N1
    j_fwd = (j + 1) % N2
    j_bwd = (j - 1) % N2
    
    neighbors_sum = model[i_fwd, j] + model[i_bwd, j] + model[i, j_fwd] + model[i, j_bwd]
    return 2.0 * spin * neighbors_sum


@njit(cache = ALLOW_NUMBA_CACHING)
def _delta_energy_3d(model: np.ndarray, i: int, j: int, k: int) -> float:
    '''Delta energy for 3D lattice (Numba-compiled).'''
    N1 = model.shape[0]
    N2 = model.shape[1]
    N3 = model.shape[2]
    spin = model[i, j, k]
    
    i_fwd = (i + 1) % N1
    i_bwd = (i - 1) % N1
    j_fwd = (j + 1) % N2
    j_bwd = (j - 1) % N2
    k_fwd = (k + 1) % N3
    k_bwd = (k - 1) % N3
    
    neighbors_sum = (model[i_fwd, j, k] + model[i_bwd, j, k] + 
                     model[i, j_fwd, k] + model[i, j_bwd, k] + 
                     model[i, j, k_fwd] + model[i, j, k_bwd])
    return 2.0 * spin * neighbors_sum


def delta_energy(model: np.ndarray, index: tuple) -> float:
    '''
    Given an Ising model and a spin flip index, compute the change in energy.
    Routes to specialized Numba-compiled versions based on dimensionality.
    '''
    
    if not isinstance(index, tuple):
        index = (index,)
    
    if model.ndim == 1:
        return _delta_energy_1d(model, index[0])
    elif model.ndim == 2:
        return _delta_energy_2d(model, index[0], index[1])
    elif model.ndim == 3:
        return _delta_energy_3d(model, index[0], index[1], index[2])
    else:
        # Fallback for higher dimensions (Python)
        spin = model[index]
        E_local = 0.0
        for axis in range(model.ndim):
            forward_idx = (*index[:axis], (index[axis] + 1) % model.shape[axis], *index[axis+1:])
            backward_idx = (*index[:axis], (index[axis] - 1) % model.shape[axis], *index[axis+1:])
            E_local += spin * (model[forward_idx] + model[backward_idx])
        return 2.0 * E_local


@njit(cache = ALLOW_NUMBA_CACHING)
def magnetization(model: np.ndarray) -> float:
    '''
    Given an Ising model, compute its magnetization.
    '''

    return abs(np.sum(model)) / model.size



if __name__ == "__main__":

    np.random.seed(0)
    m1 = new_random_ising((1000, 1001))
    m2 = np.copy(m1)
    idx = tuple(np.random.randint(0, m1.shape[dim]) for dim in range(m1.ndim))
    m2[idx] = - m2[idx]
    
    print("Energy of m1:", energy(m1))
    print("Energy of m2:", energy(m2))
    print("Delta energy (m2 - m1):", energy(m2) - energy(m1))
    print("Delta energy:", delta_energy(m1, idx))