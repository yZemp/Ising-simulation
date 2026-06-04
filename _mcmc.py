import numpy as np
import matplotlib.pyplot as plt
from mcmc_utils import maxwell_boltzmann_statistics

MIN = 0
MAX = 2000
STEP = 0.2

def test_distribution(x):
    return np.exp(-0.5 * np.power(x, 2)) + .5 * np.abs(np.sin(.2 * x + .4))
    # return np.abs(np.cos(.2 * x) + .2 * x)

def metropolis(target: callable, steps: int, min = MIN, max = MAX, burn_in = 10_000):
    samples = np.zeros(steps)
    np.random.seed(1)

    # Generate first sample
    samples[0] = np.random.uniform(min, max)
    current = samples[0]

    # Generating candidate using a gaussian kernel (no hastings)
    for i in range(1, steps - 1):
        candidate = np.random.normal(current, scale = 10.)
        # Reject candidate if out of bounds
        if candidate < min or candidate > max:
            continue

        # Rejection process
        r = np.random.random()
        threshold = target(candidate) / target(current)
        # print(f"{candidate}, {current}, Threshold: {threshold}")
        if r > np.minimum(1, threshold):
            samples[i] = candidate
        else:
            current = candidate
            samples[i] = candidate

    return samples[10_000:]


if __name__ == "__main__":
    target = maxwell_boltzmann_statistics
    x = arr = np.arange(MIN, MAX + STEP, STEP)
    y = target(x) * .003
    plt.plot(x, y)
    plt.title("Test Distribution")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    samples = metropolis(target, 100_000)
    print(samples)
    plt.hist(samples, density = True, alpha = 0.5, label = "Samples")
    # plt.plot(trimmed_samples, np.arange(len(trimmed_samples)) / 2_000, marker = "o", alpha = 0.2, label = "Samples")

    plt.show()