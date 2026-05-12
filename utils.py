import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ising import IsingModel

def maxwell_boltzmann_statistics(epsilon, beta = 1.):
    return np.exp(- (beta * epsilon))

def random_index(array):
    """Return a random index of a multi-dimensional NumPy array."""
    array = np.asarray(array)

    # Check if the array is empty or has zero dimensions
    if array.ndim == 0:
        raise ValueError("array must have at least one dimension")
    if array.size == 0:
        raise ValueError("array must not be empty")

    # Generate a random index for each dimension of the array
    if array.ndim == 1:
        index = np.random.randint(0, array.shape[0])
        return index
    else:
        index = tuple(np.random.randint(0, dimension) for dimension in array.shape)
        return index


def metropolis_ising(target: callable, model: IsingModel, steps: int, min = 0, max = 100_000, burn_in = 0, seed = 0):
    '''
    Given:
        Target probability distribution function
        Initial state of an ising model
        Number of steps to take (in the markov process)
    
    Returns:
        Array of ising model configurations which energies are distributed as the target pdf
    '''
    
    energies_samples = np.zeros(steps)
    models_samples = np.zeros(steps, dtype = object)
    current_model = model.copy()
    current_energy = model.energy()

    rng = np.random.default_rng(seed)
    
    # Generating candidate using a random spin flip (no-hasting condition satisfied)
    for i in range(0, steps):

        # Print progress every 10% of the steps if steps is large enough
        if steps >= 1_000 and i % (steps // 10) == 0:
            print(f"Progress: {i}/{steps} steps ({(i/steps)*100:.1f}%)")

        candidate_model = current_model.copy()
        index = random_index(candidate_model.spins)
        candidate_model.spins[index] = - candidate_model.spins[index]
        candidate_energy = candidate_model.energy()
        delta_energy = candidate_energy - current_energy

        # Reject candidate if out of bounds
        # if delta_energy < min or delta_energy > max:
        #     energies_samples[i] = current_energy
        #     models_samples[i] = current_model
        #     continue

        # Rejection process
        r = rng.random()
        threshold = target(delta_energy)
        if r > np.minimum(1, threshold):
            energies_samples[i] = current_energy
            models_samples[i] = current_model
        else:
            current_model = candidate_model
            current_energy = candidate_energy
            energies_samples[i] = candidate_energy
            models_samples[i] = candidate_model


    return energies_samples[burn_in:], models_samples[burn_in:]


#####################################################################
# Operators that act on models configurations
#####################################################################

# Hope is to, one day, abandon the class implementation of the Ising model.
# ==> Just use array of integers for the spins
#     with operators acting on them 

def energy(model: IsingModel):
    '''
    Given an Ising model, compute its energy.
    '''

    interaction_energy = - model.J * sum(
            np.sum(model.spins * np.roll(model.spins, 1, axis = axis))
            for axis in range(model.spins.ndim)
        )
    
    return interaction_energy


def magnetization(model: IsingModel):
    '''
    Given an Ising model, compute its magnetization.
    '''

    return abs(np.sum(model.spins)) / model.spins.size



if __name__ == "__main__":
    arrx = np.linspace(0, 1000)
    for temp in range(1, 1000, 100):
        plt.plot(arrx, maxwell_boltzmann_statistics(arrx, beta = 1 / temp), label = f"T = {temp} K")
    plt.legend()
    plt.show()
