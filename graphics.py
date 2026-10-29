from typing import cast
import h5py
import numpy as np
import matplotlib.pyplot as plt

from operators import magnetization

##################################################################################################
# Simple graph (for standard data visualization)
##################################################################################################

def graph(x, y, yerr = None, xlabel = '', ylabel = '', title = '', filename = 'tmp.png', color = 'blue'):
    '''
    Create a line graph from x and y data and save as PNG.

    - `x` and `y` are iterables of the same length containing the data points.
    - `yerr` is an iterable of the same length as `x` and `y` containing the error bars.
    - `xlabel`, `ylabel`, and `title` are strings for labeling the graph.
    - `filename` is the output PNG file path.

    Returns the saved file path.
    '''

    if len(x) != len(y) or (yerr is not None and len(x) != len(yerr)):
        raise ValueError('`x` and `y` must be of the same length')

    plt.figure()
    if yerr is not None:
        plt.errorbar(x, y, yerr = yerr, fmt = 'None', ecolor = 'red', elinewidth = 1)
    plt.plot(x,
            y,
            color = color,
            marker = '.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.close()

    return filename


def magnetization_tfixed_graph(N, dim, Tidx, data_file = "tmp.hdf5", filename = "magnetization.png"):
    '''
    Plots the magnetization of the Ising model as a function of time (steps) at a fixed temperature T 
    given the raw data stored in an HDF5 file.

    NOTE: This function is not really useful
    it was somewhat useful to check the correct functioning of the mcmc process
    '''


    with h5py.File(data_file, "r") as file:
        group = cast(h5py.Group, file[f"dim_{dim}_N_{N}"])
        temperatures = cast(h5py.Dataset, group["temperatures"])
        raw_data = cast(h5py.Dataset, group["raw_data"])
        T = temperatures[Tidx]
        data = np.array(raw_data[Tidx])

    print(f"Filtered data shape: {data.shape}")
    # filtered_data = raw_data[:, 30_000::10_000]
    magnetizations = np.array([magnetization(model) for model in data])

    plt.plot(range(len(magnetizations)), magnetizations, label = f"T = {T:.2f}")
    plt.xlabel('Time (Steps)')
    plt.ylabel('Magnetization')
    plt.title(f"MC Magnetization - N = {N}, dim = {dim}")
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':

    pass