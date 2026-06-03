import numpy as np
import matplotlib.pyplot as plt
from ising import IsingModel
from iminuit import Minuit
from scipy.stats import norm

def fit_gaussian_to_histogram(data):
    """Fit a Gaussian distribution to histogram data using Minuit."""
    counts, bin_edges = np.histogram(data, density = True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    def chi2(mu, sigma, amplitude):
        expected = amplitude * norm.pdf(bin_centers, mu, sigma)
        return np.sum(((counts - expected) ** 2) / (counts + 1e-10))
    
    initial_amplitude = np.max(counts)
    
    m = Minuit(chi2, mu = np.mean(data), sigma = np.std(data), amplitude = initial_amplitude)
    m.limits['sigma'] = (0.01, None) 
    m.limits['amplitude'] = (0.01, None)
    m.migrad()
    return m

def main():
    np.random.seed(0)
    energies = []
    for N in range(5000):
        m = IsingModel(100)
        energies.append(m.energy())
    plt.hist(energies, density = True)

    fit_result = fit_gaussian_to_histogram(energies)
    x = np.linspace(min(energies), max(energies), 1000)
    
    mu = fit_result.values['mu']
    sigma = fit_result.values['sigma']
    amplitude = fit_result.values['amplitude']
    
    gaussian = amplitude * norm.pdf(x, mu, sigma)
    plt.plot(x, gaussian, label=f"Fitted Gaussian (valid = {fit_result.valid})")
    plt.legend()

    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Energy distribution of 1D Ising model")
    plt.show()

if __name__ == "__main__":
    main()