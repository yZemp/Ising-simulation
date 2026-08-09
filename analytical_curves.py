import matplotlib.pyplot as plt
import numpy as np

def _analytical_curve_1D(t: np.ndarray) -> np.ndarray:
    # No analytical phase transition in 1D Ising model, magnetization is zero for all T > 0
    result = np.zeros_like(t, dtype=float)
    return result


def _analytical_curve_2D(t: np.ndarray) -> np.ndarray:
    # Critical temperature for 2D Ising model
    Tc = 2 / np.log(1 + np.sqrt(2))

    result = np.zeros_like(t, dtype=float)
    mask = t < Tc
    result[mask] = np.power(1 - np.power(np.sinh(2 / t[mask]), -4), (1 / 8))
    
    return result


def _analytical_curve_3D(t: np.ndarray) -> np.ndarray:
    # Aproximate data for 3D Ising model from literature
    Tc = 4.5115
    beta = 0.326419
    B = 1.69 

    result = np.zeros_like(t, dtype=float)
    mask = t < Tc
    result[mask] = B * np.power(1 - t[mask] / Tc, beta)

    return result


def plot_analytical_curve(dim: int, t: np.ndarray) -> None:
    t = np.asarray(t)

    if dim == 1:
        f = _analytical_curve_1D
    elif dim == 2:
        f = _analytical_curve_2D
    elif dim == 3:
        f = _analytical_curve_3D
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    plt.plot(t, f(t), color='black', linestyle='--', label='Analytical Curve')




if __name__ == "__main__":

    temperatures = np.linspace(0.1, 5, 100)
    magnetizations = [_analytical_curve_2D(T) for T in temperatures]

    plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='r', linestyle='--', label="Critical Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.title("Analytical Magnetization Curve for 2D Ising Model")
    plt.legend()
    plt.grid()
    plt.show()