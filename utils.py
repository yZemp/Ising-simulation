import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from ising import new_random_ising
from numba import njit

def maxwell_boltzmann_statistics(epsilon, beta = 1.):
    return np.exp(- (beta * epsilon))

def random_index(array):
    '''
    Return a random index of a multi-dimensional NumPy array as a tuple.
    '''
    
    array = np.asarray(array)

    # Check if the array is empty or has zero dimensions
    if array.ndim == 0:
        raise ValueError("array must have at least one dimension")
    if array.size == 0:
        raise ValueError("array must not be empty")

    # Generate a random index for each dimension of the array
    return tuple(np.random.randint(0, dimension) for dimension in array.shape)


def metropolis_ising(target: callable, model: np.ndarray, steps: int, min = 0, max = 100_000, burn_in = 0, seed = 0):
    '''
    Given:
        Target probability distribution function
        Initial state of an ising model
        Number of steps to take (in the markov process)
    
    Returns:
        Array of ising model configurations which energies are distributed as the target pdf
    '''
    
    curr_model = np.array(model, copy = True)
    models_samples = np.empty((steps,) + curr_model.shape, dtype = curr_model.dtype)
    rng = np.random.default_rng(seed)

    # Generating candidate using a random spin flip (no-hasting condition satisfied)
    for i in range(0, steps):

        # Print progress every 10% of the steps if steps is large enough
        if steps >= 1_000 and i % (steps // 10) == 0:
            print(f"Progress: {i}/{steps} steps ({(i/steps)*100:.1f}%)")

        index = random_index(curr_model)
        deltaE = delta_energy(curr_model, index)

        # Rejection process
        accept_probability = np.minimum(1.0, target(deltaE))
        if rng.random() <= accept_probability:
            curr_model[index] = -curr_model[index]

        models_samples[i] = curr_model


    return models_samples[burn_in:]


#####################################################################
# Operators that act on models configurations
#####################################################################

# model = ndarray of spins, J = 1 for simplicity everywhere

def energy(model: np.ndarray) -> float:
    '''
    Given an Ising model, compute its energy.
    '''

    interaction_energy = 0.0
    for axis in range(model.ndim):
        interaction_energy -= np.sum(model * np.roll(model, 1, axis = axis))
    
    return interaction_energy

@njit(cache=True)
def _delta_energy_1d(model: np.ndarray, index: int) -> float:
    '''Delta energy for 1D lattice (Numba-compiled).'''
    N = model.shape[0]
    spin = model[index]
    forward = (index + 1) % N
    backward = (index - 1) % N
    return 2.0 * spin * (model[forward] + model[backward])


@njit(cache=True)
def _delta_energy_2d(model: np.ndarray, i: int, j: int) -> float:
    '''Delta energy for 2D lattice (Numba-compiled).'''
    N1, N2 = model.shape
    spin = model[i, j]
    
    i_fwd = (i + 1) % N1
    i_bwd = (i - 1) % N1
    j_fwd = (j + 1) % N2
    j_bwd = (j - 1) % N2
    
    neighbors_sum = model[i_fwd, j] + model[i_bwd, j] + model[i, j_fwd] + model[i, j_bwd]
    return 2.0 * spin * neighbors_sum


@njit(cache=True)
def _delta_energy_3d(model: np.ndarray, i: int, j: int, k: int) -> float:
    '''Delta energy for 3D lattice (Numba-compiled).'''
    N1, N2, N3 = model.shape
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


@njit(cache = True)
def magnetization(model: np.ndarray) -> float:
    '''
    Given an Ising model, compute its magnetization.
    '''

    return abs(np.sum(model)) / model.size



if __name__ == "__main__":

    np.random.seed(0)
    m1 = new_random_ising((1000, 1001))
    m2 = np.copy(m1)
    idx = random_index(m1)
    m2[idx] = - m2[idx]
    
    print("Energy of m1:", energy(m1))
    print("Energy of m2:", energy(m2))
    print("Delta energy (m2 - m1):", energy(m2) - energy(m1))
    print("Delta energy:", delta_energy(m1, idx))