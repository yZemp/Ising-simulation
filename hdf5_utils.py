import h5py
import numpy as np
import os

# This is barely (if ever) used for performance reasons
# Stored as reference more than anything else

def read_data(data_file, N, dim):
    print(f"Reading data from {data_file}...")

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        raw_data = np.array(file[f"dim_{dim}_N_{N}/raw_data"])

    print("Done.")

    return temperatures, raw_data


def verify_hdf5_integrity(file_path):
    file_dir = os.path.dirname(os.path.abspath(file_path))

    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"Access successful. Root groups found: {keys}")

            failing_node = None

            def _check_node(name, obj):
                nonlocal failing_node
                failing_node = f"/{name}" if name else "/"

                # Force metadata access on groups and datasets.
                if isinstance(obj, h5py.Group):
                    list(obj.keys())
                elif isinstance(obj, h5py.Dataset):
                    _ = obj.shape
                    _ = obj.dtype

                    # Lightweight read to catch dataset-level corruption.
                    if obj.shape == ():
                        _ = obj[()]
                    elif (obj.size or 0) > 0:
                        first_index = tuple(0 for _ in obj.shape)
                        _ = obj[first_index]

            # Deep iteration to verify each node and report exact failing path.
            f.visititems(_check_node)
            
            return True

    except OSError as e:
        print(
            f"Critical I/O Error in directory '{file_dir}' "
            f"(Truncated file or missing header): {e}"
        )
        return False
    
    except Exception as e:
        print(
            f"Structure Error in directory '{file_dir}' at HDF5 path "
            f"'{locals().get('failing_node', 'unknown')}': {e}"
        )
        return False



def print_hdf5_tree(file_path):

    def func(name, node):
        if isinstance(node, h5py.Dataset):
            print(f"{name} [Dataset] - Shape: {node.shape}, Dtype: {node.dtype}")
        else:
            print(f"{name}/ [Group]")

    try:
        with h5py.File(file_path, "r") as f:
                print(f"--- Tree of {file_path} ---")
                f.visititems(func)
    except OSError as e:
        print(f"Error reading file (corrupted or non-existent): {e}")



if __name__ == "__main__":

    N = 15
    dim = 3

    data_file = r"E:\simulations_data\dim_{dim}_N_{N}_data.hdf5".format(dim = dim, N = N)

    verify_hdf5_integrity(data_file)
    print_hdf5_tree(data_file)