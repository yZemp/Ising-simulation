import numpy as np
from numba import njit
import h5py
from typing import cast

from global_variables import ALLOW_NUMBA_CACHING


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

    if model.ndim == 1:
        n = model.shape[0]
        for i in range(n):
            interaction_energy -= model[i] * model[(i - 1) % n]
        return interaction_energy

    elif model.ndim == 2:
        n1, n2 = model.shape[0], model.shape[1]
        for i in range(n1):
            im1 = (i - 1) % n1
            for j in range(n2):
                jm1 = (j - 1) % n2
                s = model[i, j]
                interaction_energy -= s * model[im1, j]
                interaction_energy -= s * model[i, jm1]
        return interaction_energy

    elif model.ndim == 3:
        n1, n2, n3 = model.shape[0], model.shape[1], model.shape[2]
        for i in range(n1):
            im1 = (i - 1) % n1
            for j in range(n2):
                jm1 = (j - 1) % n2
                for k in range(n3):
                    km1 = (k - 1) % n3
                    s = model[i, j, k]
                    interaction_energy -= s * model[im1, j, k]
                    interaction_energy -= s * model[i, jm1, k]
                    interaction_energy -= s * model[i, j, km1]
        return interaction_energy

    else:
        raise ValueError("energy supports 1D, 2D, and 3D models only")


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
        spin: float = float(model[index])
        E_local: float = 0.0
        for axis in range(model.ndim):
            forward_idx = (*index[:axis], (index[axis] + 1) % model.shape[axis], *index[axis+1:])
            backward_idx = (*index[:axis], (index[axis] - 1) % model.shape[axis], *index[axis+1:])
            forward_value: float = float(model[forward_idx])
            backward_value: float = float(model[backward_idx])
            E_local += spin * (forward_value + backward_value)
        return float(2.0 * E_local)


@njit(cache = ALLOW_NUMBA_CACHING)
def magnetization(model: np.ndarray) -> float:
    '''
    Given an Ising model, compute its magnetization.
    '''

    return abs(np.sum(model)) / model.size


def chi(N, dim, data_file, temperatures):
    '''
    Compute the magnetic susceptibility for a given dimension and system size.
    '''

    chi_values = []
    
    with h5py.File(data_file, "r") as file:
        for T_index, T in enumerate(temperatures):
            filtered_data = np.array(file[f"dim_{dim}_N_{N}/filtered_data"][T_index])
            
            # Filtra le matrici di zero-padding (mantiene solo le matrici con almeno un elemento non nullo)
            valid_idx = np.any(filtered_data, axis=(1, 2))
            valid_data = filtered_data[valid_idx]
            
            if len(valid_data) == 0:
                chi_values.append(0.0)
                continue
                
            magns = np.array([magnetization(model) for model in valid_data])
            chi = ((N ** dim) / T) * (np.mean(magns**2) - np.mean(np.abs(magns))**2)
            chi_values.append(chi)
            
    return np.array(chi_values)

    # for T_index, T in enumerate(temperatures):
    #     with h5py.File(data_file, "r") as file:
    #         filtered_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/filtered_data"])[T_index])
    #         # print(f"Filtered data shape: {filtered_data.shape}")

    #         magns = np.array([magnetization(model) for model in filtered_data])
    #         chi = ((N ** dim) / T) * (np.mean(magns**2) - np.mean(np.abs(magns))**2)
    #         chi_values.append(chi)

    # return np.array(chi_values)
