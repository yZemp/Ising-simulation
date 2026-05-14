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

def delta_energy(model: np.ndarray, index: tuple) -> float:
    '''
    Given an Ising model and a spin flip index, compute the change in energy.
    '''

    if not isinstance(index, tuple):
        index = (index,)

    spin = model[index]

    # Sum the interactions with the forward and backward neighbor along each axis.
    E_local = 0.0
    for axis in range(model.ndim):
        # Use tuple unpacking to avoid list creation overhead
        forward_idx = (*index[:axis], (index[axis] + 1) % model.shape[axis], *index[axis+1:])
        backward_idx = (*index[:axis], (index[axis] - 1) % model.shape[axis], *index[axis+1:])
        E_local += spin * (model[forward_idx] + model[backward_idx])

    return 2.0 * E_local

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