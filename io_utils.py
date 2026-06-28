import h5py
import numpy as np

def read_data(data_file, N, dim):
    
    print(f"Reading data from {data_file}...")

    with h5py.File(data_file, "r") as file:
            temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
            raw_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"])

    print("Done.")

    return temperatures, raw_data