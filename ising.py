from encodings import undefined

import numpy as np
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, N: np.shape, J = 1., h = 0.):
        self.N = N
        self.J = J
        self.h = h
        self.spins = np.random.randint(0, 2, size = N).astype(np.int8) * 2 - 1

    def energy(self):
        interaction_energy = 0
        for axis in range(self.spins.ndim):
            interaction_energy += np.sum(self.spins * np.roll(self.spins, shift = 1, axis = axis))

        # temporary: ignore external field contribution to energy
        # field_energy = np.sum(self.spins)
        field_energy = 0
        return - self.J * interaction_energy - self.h * field_energy

    def copy(self):
        # Bypass __init__ to avoid generating temporary random spins
        # (Without skip: ~100x slower than copying an array)
        # (With skip   : ~1.5x slower than copying an array)
        new_model = self.__class__.__new__(self.__class__)
        new_model.N = self.N
        new_model.J = self.J
        new_model.h = self.h
        new_model.spins = np.copy(self.spins)
        return new_model

    def __repr__(self):
        return f"IsingModel_{self.N}D:\n{self.spins}"


def new_random_ising(N: np.shape, J = 1., h = 0.):
    # return np.random.choice([-1, 1], size = N)
    return np.ones(N, dtype = np.int8)

if __name__ == "__main__":
    np.random.seed(0)
    m = IsingModel((2, 2))
    m.spins = np.array([[1, -1], [-1, 1]])
    print(m)
    print(m.energy())