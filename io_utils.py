import h5py
import numpy as np

def read_data(data_file, N, dim):
        
    with h5py.File(data_file, "r") as file:
            temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
            raw_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"])

    return temperatures, raw_data