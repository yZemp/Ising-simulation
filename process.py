import numpy as np
import h5py
import time

from datetime import timedelta
from typing import cast
from operators import magnetization
from graphics import graph
from matplotlib import pyplot as plt

###############################################################################
# Magnetization study

def magnetization_graph(N, dim, data_file = None, filename = "tmp.png"):
    '''
    Plots the magnetization of the Ising model as a function of temperature 
    given the raw data stored in an HDF5 file.
    '''

    if data_file is None:
        data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"

    with h5py.File(data_file, "r") as file:
        group = cast(h5py.Group, file[f"dim_{dim}_N_{N}"])
        temperatures = np.array(cast(h5py.Dataset, group["temperatures"]))
        filtered_data = np.array(cast(h5py.Dataset, group["filtered_data"]))
        if f"dim_{dim}_N_{N}/filtered_lengths" in file:
            filtered_lengths = np.array(cast(h5py.Dataset, group["filtered_lengths"]))
        else:
            filtered_lengths = np.full(len(temperatures), filtered_data.shape[1], dtype = np.int32)

    valid_models = [filtered_data[i, :filtered_lengths[i]] for i in range(len(temperatures))]
    mean_magnetization = np.array([np.mean([magnetization(model) for model in models]) for models in valid_models])
    errors = np.array([np.std([magnetization(model) for model in models]) / np.sqrt(len(models)) for models in valid_models])
    # errors = np.array([np.std([magnetization(model) for model in models]) for models in valid_models])

    graph(temperatures,
          mean_magnetization,
          yerr = errors,
          xlabel = 'T (Temperature)',
          ylabel = 'Mean magnetization',
          title = f"(*) N = {N}, dim = {dim}",
          filename = filename
          )


def magnetization_tfixed_graph(N, dim, Tidx, data_file = "tmp.hdf5", filename = "magnetization.png"):
    '''
    Plots the magnetization of the Ising model as a function of time (steps) at a fixed temperature T 
    given the raw data stored in an HDF5 file.
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


def main():

    N = 15
    dim = 2

    data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"

    Tidx = 6
    
    with h5py.File(data_file, "r") as file:
        group = cast(h5py.Group, file[f"dim_{dim}_N_{N}"])
        raw_data = cast(h5py.Dataset, group["raw_data"])
        filtered_data = np.array(raw_data[Tidx])
    observables = np.array([magnetization(model) for model in filtered_data])

    start = time.perf_counter()

    #################################################################################
    # EXECUTION

    # for N, dim in [(5, 1), (10, 1), (20, 1), (30, 1), (50, 1), (70, 1), (100, 1), (150, 1), (200, 1), (250, 1), (300, 1), (500, 1)]:
    # for N, dim in [(100, 2)]:

    #     data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"

        # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
        
    magnetization_graph(N, dim, filename = "tmp_magnetization_reduced.png")
    # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
    # tau_int_1 = autocorrelation_graph(N, dim, data_file = data_file, filename = "tmp_autocorrelation.png", T_index = Tidx)
    # tau_int_graph(N, dim, data_file = data_file, filename = "tmp_tau.png")

    #################################################################################

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    main()