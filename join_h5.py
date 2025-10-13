
import os
import h5py
import argparse
import glob
import numpy as np

def join_h5_files(folder, output_name):
    """
    Merges multiple HDF5 files from a given folder into a single HDF5 file.
    It assumes that all HDF5 files have the same datasets.
    Args:
        folder (str): The path to the folder containing the .h5 files.
        output_name (str): The name of the output file to be created.
    """
    h5_files = glob.glob(os.path.join(folder, '*.h5'))
    if not h5_files:
        print(f"No .h5 files found in {folder}")
        return

    # Remove the output file from the list if it already exists
    output_path = os.path.join(folder, output_name)
    if output_path in h5_files:
        h5_files.remove(output_path)

    if not h5_files:
        print(f"No source .h5 files to merge into {output_name}. If {output_name} was the only .h5 file, the operation is skipped.")
        return

    # If the output file exists, remove it to start fresh
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists and will be overwritten.")
        os.remove(output_path)

    print(f"Found {len(h5_files)} files to merge.")
    
    with h5py.File(output_path, 'w') as f_out:
        # Use the first file to initialize the datasets
        first_file = h5_files[0]
        print(f"Initializing with file: {first_file}")
        with h5py.File(first_file, 'r') as f_in:
            for dset_name in f_in.keys():
                data = f_in[dset_name]
                maxshape = (None,) + data.shape[1:]
                f_out.create_dataset(
                    dset_name, 
                    data=data, 
                    maxshape=maxshape, 
                    chunks=True
                )

        # Append data from the rest of the files
        for h5_file in h5_files[1:]:
            print(f"Appending file: {h5_file}")
            with h5py.File(h5_file, 'r') as f_in:
                for dset_name in f_in.keys():
                    data = f_in[dset_name][:]
                    current_size = f_out[dset_name].shape[0]
                    new_size = current_size + data.shape[0]
                    f_out[dset_name].resize(new_size, axis=0)
                    f_out[dset_name][current_size:] = data

    print("\nMerging complete.")
    with h5py.File(output_path, 'r') as f:
        print(f"Merged file created at: {output_path}")
        print("Datasets in merged file:")
        for name in f:
            print(f"  - {name}: shape {f[name].shape}, dtype {f[name].dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join multiple HDF5 files from a directory into a single file."
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        default="data", 
        help="Path to the folder containing the .h5 files (default: 'data')."
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="all.h5", 
        help="Name for the output merged .h5 file (default: 'all.h5')."
    )
    args = parser.parse_args()
    join_h5_files(args.folder, args.output_name)
