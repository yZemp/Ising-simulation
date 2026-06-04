import h5py
import numpy as np
from main import mcmc_sampling
from mcmc_utils import magnetization
from graphics import graph
from matplotlib import pyplot as plt


def tmp_write_shit(N, sample_length, dim):
    temps = np.arange(0.05, 7.0, .2)
    magns = np.zeros_like(temps)
    errors = np.zeros_like(temps)
    current_model = None

    model_shape = tuple([N] * dim)
    raw_data_shape = (len(temps), sample_length) + model_shape

    with h5py.File("tmp.hdf5", "w") as f:
        tmp_group = f.create_group(f"dim_{dim}_N_{N}_sample_length_{sample_length}")
        tmp_group.create_dataset('temperatures', data = temps)
        tmp_group.create_dataset(
        "raw_data", 
        shape = raw_data_shape, 
        dtype = np.int8,
        compression = "lzf", 
        chunks = True
        )
        
        for i, t in enumerate(temps):
            models = mcmc_sampling(
                N = N,
                dim = dim,
                T = t,
                sample_length = sample_length,
                initial_model = current_model,
            )
            current_model = models[-1]
            # magn_i = [magnetization(model) for model in models]
            # magns[i] = np.mean(magn_i)
            # errors[i] = np.std(magn_i)

            f.get(f"dim_{dim}_N_{N}_sample_length_{sample_length}/raw_data")[i] = models

            print(f"Computed {(i + 1) / len(temps) * 100:.1f}%")


if __name__ == "__main__":

    N = 20
    dim = 2
    sample_length = 200

    tmp_write_shit(N=N, sample_length=sample_length, dim=dim)

    with h5py.File("tmp.hdf5", "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}_sample_length_{sample_length}/temperatures"])
        raw_data = np.array(file[f"dim_{dim}_N_{N}_sample_length_{sample_length}/raw_data"])

    mean_magnetization = np.array([np.mean([magnetization(model) for model in models]) for models in raw_data])
    errors = np.array([np.std([magnetization(model) for model in models]) for models in raw_data])


    graph(temperatures,
          mean_magnetization,
          yerr = errors,
          xlabel = 'T (Temperature)',
          ylabel = 'Mean magnetization',
          title = f"N = {N}, dim = {dim}, sample_length = {sample_length}",
          filename = 'tmp.png'
          )