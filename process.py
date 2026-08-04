import numpy as np
import h5py
import time

from datetime import timedelta
from typing import cast
from autocorrelation import tau_int_sokal
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

    if data_file is None or data_file == "":
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

    with h5py.File(data_file, "r+") as file:
        group = cast(h5py.Group, file[f"dim_{dim}_N_{N}"])
        if "magnetizations" in group:
            del group["magnetizations"]
        if "magnetization_errors" in group:
            del group["magnetization_errors"]
        group.create_dataset("magnetizations", data = mean_magnetization)
        group.create_dataset("magnetization_errors", data = errors)



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
        where tau_int is calculated with tau_int_sokal()
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
            count = 0

            for chunk_start in range(burn_in, steps, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, steps)
                first_sample = chunk_start + ((thinning - ((chunk_start - burn_in) % thinning)) % thinning)

                if first_sample < chunk_end:
                    count += ((chunk_end - first_sample - 1) // thinning) + 1

            return count

        def iter_filtered_chunks(temperature_index, burn_in, thinning):
            for chunk_start in range(burn_in, steps, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, steps)
                first_sample = chunk_start + ((thinning - ((chunk_start - burn_in) % thinning)) % thinning)

                if first_sample < chunk_end:
                    yield np.array(raw_dataset[temperature_index, first_sample:chunk_end:thinning])

        pilot_length = min(max_chunk_size, steps)

        # First pass: estimate tau_int from a short prefix and determine output lengths.
        for i, T in enumerate(temperatures):
            print("----------------------------------------------------------------------")
            print(f"Estimating filtering window for T = {T:.2f} ({i / len(temperatures) * 100:.1f}%)")

            pilot_data = np.array(raw_dataset[i, :pilot_length])
            observables = np.array([magnetization(model) for model in pilot_data])
            tau_int = tau_int_sokal(observables, c = 20.0)

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

        max_length = int(np.max(filtered_lengths))
        filtered_dataset = file.create_dataset(
            filtered_data_path,
            shape = (len(temperatures), max_length) + model_shape,
            dtype = raw_dtype,
            compression = "lzf",
            chunks = True,
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
                filtered_dataset[i, 0] = np.array(raw_dataset[i, steps - 1])

        filtered_dataset.attrs["lengths_dataset"] = "filtered_lengths"


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
        
    # magnetization_tfixed_graph(N, dim, Tidx, data_file = data_file, filename = "tmp_magnetization_tfixed.png")
    # tau_int_1 = autocorrelation_graph(N, dim, data_file = data_file, filename = "tmp_autocorrelation.png", T_index = Tidx)
    # tau_int_graph(N, dim, data_file = data_file, filename = "tmp_tau.png")

    #################################################################################

    end = time.perf_counter()
    print(f"Elapsed = {timedelta(seconds = end - start)}")


if __name__ == "__main__":
    main()