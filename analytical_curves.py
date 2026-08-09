import matplotlib.pyplot as plt
import numpy as np

def analytical_curve_2D(t: np.ndarray) -> np.ndarray:

    t = np.asarray(t)

    # Critical temperature for 2D Ising model
    Tc = 2 / np.log(1 + np.sqrt(2))

    result = np.zeros_like(t, dtype=float)
    mask = t < Tc
    result[mask] = np.power(1 - np.power(np.sinh(2 / t[mask]), -4), (1 / 8))
    
    return result



if __name__ == "__main__":

    temperatures = np.linspace(0.1, 5, 100)
    magnetizations = [analytical_curve_2D(T) for T in temperatures]

    plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='r', linestyle='--', label="Critical Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.title("Analytical Magnetization Curve for 2D Ising Model")
    plt.legend()
    plt.grid()
    plt.show()