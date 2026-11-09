import numpy as np
import h5py
import time

from datetime import timedelta
from typing import cast
from autocorrelation import tau_int_sokal, tau_int_blocking
from operators import magnetization

###############################################################################
# Magnetization study

def magnetization_bake(N, dim, data_file = ""):
    '''
    Bakes the magnetization of the Ising model as a function of temperature
    and stores the results in an HDF5 file.
    The magnetization is the mean of the magnetization of the filtered models at each temperature.

    (Prepares the data for plotting)
    '''

    print(f"Baking magnetization for N = {N}, dim = {dim}...")

    if data_file is None or data_file == "":
        data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"

    with h5py.File(data_file, "a") as file:
        group_key = f"dim_{dim}_N_{N}"
        if group_key not in file:
            print(f"Data for N = {N}, dim = {dim} not found in {data_file}. Please run simulate() first.")
            return

        group = cast(h5py.Group, file[group_key])

        if "filtered_lengths" not in group or "filtered_data" not in group:
            print(f"Filtered data not found in {data_file}. Please run filter_data() first.")
            return

        temperatures = cast(h5py.Dataset, group["temperatures"])[:]
        n_temps = len(temperatures)
        
        ds_lengths = cast(h5py.Dataset, group["filtered_lengths"])
        filtered_lengths = ds_lengths[:n_temps]

        # Load filtered data into memory (only the relevant part)
        ds_data = cast(h5py.Dataset, group["filtered_data"])
        
        mean_magnetization = np.empty(n_temps)
        errors = np.empty(n_temps)

        # Lazy loading of the data in memory
        for i in range(n_temps):
            length = int(filtered_lengths[i])
            
            models = ds_data[i, :length] 

            # Compute magnetization
            mags = np.array([magnetization(model) for model in models])
            mean_magnetization[i] = np.mean(mags)
            
            # Compute error
            tau, _ = tau_int_sokal(mags)
            var_mu = np.maximum(1.0, 2 * tau) * np.var(mags) / len(mags)
            errors[i] = np.sqrt(var_mu)

        # Store results in the HDF5 file
        if "magnetizations" in group:
            del group["magnetizations"]
        if "magnetization_errors" in group:
            del group["magnetization_errors"]
            
        group.create_dataset("magnetizations", data=mean_magnetization)
        group.create_dataset("magnetization_errors", data=errors)

    print(f"Baked magnetization values for N = {N}, dim = {dim}.")

###################################################################################
# Data filtering

def filter_data(N, dim, data_file = "", max_chunk_size = 100_000):
    '''
    Filters the raw data stored in an HDF5 producing an actual sample of Ising states.
    Saves the filtered data in the same HDF5 file.

    The raw Markov chain is processed in chunks to keep memory usage bounded.
    A short pilot chunk is used only to estimate tau_int for each temperature.
    NOTE: The bigger the chunk size, the more accurate the tau_int estimation will be.
    Probably should be somewhere close to 100_000 or higher.
    
    burn_in:
        TODO: implement using tau_exp or graphical method
    thinning:
        1 element every 2 * tau_int
        where tau_int can be estimated using Sokal's method or the data blocking method.
    '''
    
    if data_file is None or data_file == "":
        data_file = f"dim_{dim}_N_{N}_data.hdf5"
    group_name = f"dim_{dim}_N_{N}"
    filtered_data_path = f"{group_name}/filtered_data"
    filtered_lengths_path = f"{group_name}/filtered_lengths"

    print(f"Filtering data for N = {N}, dim = {dim}...")

    with h5py.File(data_file, "r+") as file:
        raw_dataset = cast(h5py.Dataset, file[f"{group_name}/raw_data"])
        temperatures = np.array(cast(h5py.Dataset, file[f"{group_name}/temperatures"]))
        steps = raw_dataset.shape[1]
        model_shape = raw_dataset.shape[2:]
        raw_dtype = raw_dataset.dtype

        if steps > max_chunk_size:
            print(
                f"The simulation length exceeds the maximum chunk size ({steps} > {max_chunk_size}).\n"
                f"Filtering will be done in chunks to avoid memory issues."
            )

        filtered_lengths = np.zeros(len(temperatures), dtype = np.int32)
        burn_ins = np.zeros(len(temperatures), dtype = np.int32)
        thinnings = np.zeros(len(temperatures), dtype = np.int32)

        if filtered_data_path in file:
            del file[filtered_data_path]

        if filtered_lengths_path in file:
            del file[filtered_lengths_path]

        def count_filtered_samples(burn_in, thinning):
            # Counts the number of samples that will be kept after filtering, given burn-in and thinning parameters
            count = 0

            for chunk_start in range(burn_in, steps, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, steps)
                first_sample = chunk_start + ((thinning - ((chunk_start - burn_in) % thinning)) % thinning)

                if first_sample < chunk_end:
                    count += ((chunk_end - first_sample - 1) // thinning) + 1

            return count

        def iter_filtered_chunks(temperature_index, burn_in, thinning):
            # Iterates over the filtered samples for a given temperature index, yielding chunks of filtered samples
            for chunk_start in range(burn_in, steps, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, steps)
                
                data_chunk = raw_dataset[temperature_index, chunk_start:chunk_end]

                offset = (thinning - ((chunk_start - burn_in) % thinning)) % thinning
                if offset < data_chunk.shape[0]:
                    yield data_chunk[offset::thinning]

        pilot_length = min(max_chunk_size, steps)

        # First pass: estimate tau_int from a short prefix and determine output lengths.
        for i, T in enumerate(temperatures):
            print("----------------------------------------------------------------------")
            print(f"Estimating filtering window for T = {T:.2f} ({i / len(temperatures) * 100:.1f}%)")

            pilot_data = raw_dataset[i, :pilot_length]
            axes_to_reduce = tuple(range(1, pilot_data.ndim))
            observables = np.abs(np.mean(pilot_data, axis=axes_to_reduce))
            tau_int, _ = tau_int_sokal(observables, c = 10.0)

            if tau_int > 0:
                burn_in = min(int(20 * tau_int), max(steps - 1, 0))
                thinning = max(int(2 * tau_int), 1)
            else:
                # Default values for non properly physical systems (e.g. T --> 0)
                burn_in = 0
                thinning = 10_000

            burn_ins[i] = burn_in
            thinnings[i] = thinning
            filtered_lengths[i] = count_filtered_samples(burn_in, thinning)

        max_length = max(1, int(np.max(filtered_lengths)))
        bytes_per_model = int(np.prod(model_shape) * raw_dtype.itemsize)
        target_chunk_bytes = 1024 * 1024
        chunk_time_size = max(1, target_chunk_bytes // bytes_per_model)
        chunk_time_size = int(max(1, min(max_length, chunk_time_size)))
        filtered_dataset = file.create_dataset(
            filtered_data_path,
            shape=(len(temperatures), max_length) + model_shape,
            dtype=raw_dtype,
            compression="lzf",
            chunks=(1, chunk_time_size) + model_shape
        )
        file.create_dataset(
            filtered_lengths_path,
            data = filtered_lengths,
        )

        # Second pass: stream the filtered samples directly into the output dataset.
        for i, T in enumerate(temperatures):
            print("----------------------------------------------------------------------")
            print(f"Filtering data {i} ({i / len(temperatures) * 100:.1f}%)")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

            write_pos = 0
            for filtered_chunk in iter_filtered_chunks(i, burn_ins[i], thinnings[i]):
                chunk_length = filtered_chunk.shape[0]
                filtered_dataset[i, write_pos:write_pos + chunk_length] = filtered_chunk
                write_pos += chunk_length

            if write_pos == 0:
                filtered_dataset[i, 0] = raw_dataset[i, steps - 1]

        filtered_dataset.attrs["lengths_dataset"] = "filtered_lengths"

        print("Filtering completed for N = {N}, dim = {dim}.")


def main():

    N = 100
    dim = 1

    # data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"
    data_file = f"dataset_copy" + ".hdf5"

    Tidx = 6
    
    # with h5py.File(data_file, "r") as file:
    #     group = cast(h5py.Group, file[f"dim_{dim}_N_{N}"])
    #     raw_data = cast(h5py.Dataset, group["raw_data"])
    #     filtered_data = np.array(raw_data[Tidx])
    # observables = np.array([magnetization(model) for model in filtered_data])

    start = time.perf_counter()

    #################################################################################
    # EXECUTION

    # for N, dim in [(5, 1), (10, 1), (20, 1), (30, 1), (50, 1), (70, 1), (100, 1), (150, 1), (200, 1), (250, 1), (300, 1), (500, 1)]:
    # for N, dim in [(100, 2)]:

    #     data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"

        # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
        
    # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
    # tau_int_1 = autocorrelation_graph(N, dim, data_file = data_file, filename = "tmp_autocorrelation.png", T_index = Tidx)
    # tau_int_graph(N, dim, data_file = data_file, filename = "tmp_tau.png")
    # filter_data(N, dim, data_file = data_file)
    magnetization_bake(N, dim, data_file = data_file)

    #################################################################################

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":

    Ns = [30]
    dim = 3


    for N in Ns:
        data_file = r"E:\simulations_data\sweep_based_chains\dim_{dim}_N_{N}_data.hdf5".format(N = N, dim = dim)
        # filter_data(N, dim, data_file = data_file)
        magnetization_bake(N, dim, data_file = data_file)

    # N = 4
    # dim = 2
    # data_file = f"E:\\simulations_data\\sweep_based_chains\\dim_{dim}_N_{N}" + "_data.hdf5".format(N = N, dim = dim)
    # with h5py.File(data_file, "r") as file:
    #     sweep_sampling = bool(file[f"dim_{dim}_N_{N}"].attrs.get('sweep_sampling', False))
    #     print(f"sweep_sampling = {sweep_sampling}")

    # main()