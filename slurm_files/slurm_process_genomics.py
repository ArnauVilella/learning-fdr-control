import os
import numpy as np
import json
from tqdm import tqdm
from trexselector import random_experiments
from sklearn.preprocessing import StandardScaler
import time
import argparse
from mpi4py import MPI
import h5py


def create_subdir(subdir):
    """Create a subdirectory if it doesn't exist."""
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

def process_system(args):
    """
    Process a single system, sample a random v and T, and compute FDP.
    """
    # Unpack arguments
    system_id = args['system_id']
    X_files = args['X_files']
    X_dir = args['X_dir']
    beta_files = args['beta_files']
    beta_dir = args['beta_dir']
    y = args['y']
    K = args['K']
    T_stop_max = args['T_stop_max']
    num_dummies = args['num_dummies']
    num_dummies_factor = args['num_dummies_factor']
    trex_parallel_process = args['trex_parallel_process']
    storage_format = args['storage_format']
    output_path = args['output_path']
    verbose = args['verbose']
    multilevel = args['multilevel']
    all_pairs = args['all_pairs']
    test = args['test']

    # Get X matrix (cycling through available matrices)
    X_file = X_files[system_id % len(X_files)]
    X = np.loadtxt(os.path.join(X_dir, X_file), dtype=np.int16)
    p = X.shape[1]

    beta_file = beta_files[system_id % len(beta_files)]
    beta = np.loadtxt(os.path.join(beta_dir, beta_file), dtype=np.int16)

    # Create random permutation
    n = X.shape[0]
    perm = np.arange(n)
    X_perm = X[perm, :]
    y_perm = y[perm]
    beta_perm = beta

    if num_dummies_factor == 1:
        L = num_dummies
    else:
        lower_bound = num_dummies
        upper_bound = num_dummies_factor * num_dummies
        possible_Ls = np.arange(lower_bound, upper_bound + 1, num_dummies)
        if len(possible_Ls) == 0:
            L = num_dummies
        else:
            L = int(np.random.choice(possible_Ls))
    
    if verbose:
        start_time = time.time()
    scaler = StandardScaler()
    res_exp = random_experiments(scaler.fit_transform(X_perm), y_perm - np.mean(y_perm), K=K, T_stop=T_stop_max, num_dummies=L, 
                              verbose=False, parallel_process=trex_parallel_process)
    if verbose:
        print(f"System {system_id}: Time taken for random_experiments: {time.time() - start_time:.2f} seconds")
    Phi_mat = res_exp["phi_T_mat"]

    # Pad matrices to a fixed size
    p_padded = 545
    if Phi_mat.shape[0] < p_padded:
        Phi_mat = np.pad(Phi_mat, ((0, p_padded - Phi_mat.shape[0]), (0, 0)), mode='constant', constant_values=0)
    if beta_perm.shape[0] < p_padded:
        beta_perm = np.pad(beta_perm, (0, p_padded - beta_perm.shape[0]), mode='constant', constant_values=0)
    
    X_perm_padded = X_perm
    if test and X_perm.shape[1] < p_padded:
        X_perm_padded = np.pad(X_perm, ((0, 0), (0, p_padded - X_perm.shape[1])), mode='constant', constant_values=0)

    system_results = {
        'betas': [], 'phi_mats': [], 'v_values': [],
        'T_stop_values': [], 'fdp_values': [], 'experiment_ids': [],
        'L_values': []
    }
    if test:
        system_results['X_mats'] = []
        system_results['y_vecs'] = []

    num_to_sample = min(multilevel, len(all_pairs))
    indices = np.random.choice(len(all_pairs), num_to_sample, replace=False)
    pairs_to_process = [all_pairs[i] for i in indices]

    for i, (v, T) in enumerate(pairs_to_process):
        global_exp_id = (system_id * multilevel) + i
        FDP = np.sum((1 - beta_perm) * (Phi_mat[:, T-1] > v), axis=0) / np.maximum(1, np.sum(Phi_mat[:, T-1] > v, axis=0))
        
        if storage_format == 'json' and output_path is not None:
            experiment_data = {
                "beta": beta_perm.tolist(), "Phi_mat": Phi_mat.tolist(), "v": float(v),
                "T_stop": int(T), "FDP": float(FDP), "system_id": int(system_id),
                "experiment_id": int(global_exp_id),
                "L": int(L)
            }
            if test:
                experiment_data['X_mat'] = X_perm_padded.tolist()
                experiment_data['y_vec'] = y_perm.tolist()
            json_path = os.path.join(output_path, f"experiment_{global_exp_id}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
        
        system_results['betas'].append(beta_perm.copy())
        system_results['phi_mats'].append(Phi_mat.copy())
        if test:
            system_results['X_mats'].append(X_perm_padded.copy())
            system_results['y_vecs'].append(y_perm.copy())
        system_results['v_values'].append(v)
        system_results['T_stop_values'].append(T)
        system_results['fdp_values'].append(FDP)
        system_results['experiment_ids'].append(global_exp_id)
        system_results['L_values'].append(L)
    return system_results

def generate_genomics_FDP_labeling(data_dir="data", specific_folder=None, T_stop_max=1, 
                                   K=100, num_dummies=150, num_dummies_factor=1, SNR=1.0, 
                                   show_progress=True, storage_format='json', trex_parallel_process=False,
                                   N_systems_multiple=1, verbose=False, multilevel=1, test=False):
    """
    Generates FDP data using fixed genomics data with random permutations.

    Parameters
    ----------
    storage_format : str, default='json'
        Storage format for the dataset. Options: 'json', 'npz', 'hdf5'.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"MPI initialized with {size} processes. Storage format: {storage_format.upper()}")

    X_dir = os.path.join("genomics_data", "preprocessed_matrices")
    if not os.path.exists(X_dir): raise FileNotFoundError(f"X matrices directory not found at {X_dir}")
    X_files = sorted([f for f in os.listdir(X_dir) if f.startswith("X_") and f.endswith(".txt")])
    if not X_files: raise ValueError(f"No X matrix files found in {X_dir}")
    
    beta_dir = os.path.join("genomics_data", "preprocessed_betas")
    if not os.path.exists(beta_dir): raise FileNotFoundError(f"betas directory not found at {beta_dir}")
    beta_files = sorted([f for f in os.listdir(beta_dir) if f.startswith("beta_") and f.endswith(".txt")])
    if not beta_files: raise ValueError(f"No beta files found in {beta_dir}")
    
    # y = np.zeros(1000); y[:300] = 1
    y = np.array([0]*100 + [1]*200)
    N_systems = len(X_files) * N_systems_multiple
    p_padded = 545

    output_path = None
    if rank == 0:
        if specific_folder is None:
            num_dummies_str = f"num_dummies={num_dummies}" if num_dummies_factor == 1 else f"num_dummies=[{num_dummies}, {num_dummies_factor * num_dummies}]"
            param_order = [f"N_systems={N_systems}", f"SNR={SNR}", f"T_stop_max={T_stop_max}", num_dummies_str, f"K={K}"]
            folder_prefix = f"(genomics_multilevel_{multilevel})_" if multilevel > 1 else "(genomics)_"
            if test:
                folder_prefix = folder_prefix.replace(")_", "_test)_")
            specific_folder = folder_prefix + ','.join(param_order)
        
        create_subdir(data_dir)
        if storage_format == 'json':
            output_path = os.path.join(data_dir, specific_folder)
            create_subdir(output_path)
        elif storage_format == 'npz':
            output_path = os.path.join(data_dir, f"{specific_folder}.npz")
        elif storage_format == 'hdf5':
            output_path = os.path.join(data_dir, f"{specific_folder}.h5")
    
    output_path = comm.bcast(output_path, root=0)

    if storage_format == 'hdf5':
        if h5py is None:
            if rank == 0: raise ImportError("h5py is not installed. Please run 'pip install h5py'.")
            comm.Barrier(); return None
        
        total_experiments = N_systems * multilevel
        if rank == 0: print(f"Creating HDF5 file at {output_path} for {total_experiments} experiments.")
        
        with h5py.File(output_path, 'w', driver='mpio', comm=comm) as f:
            f.create_dataset('betas', (total_experiments, p_padded), dtype='i2')
            f.create_dataset('phi_mats', (total_experiments, p_padded, K), dtype='f4')
            if test:
                f.create_dataset('X_mats', (total_experiments, 300, p_padded), dtype='i2')
                f.create_dataset('y_vecs', (total_experiments, 300), dtype='f4')
            f.create_dataset('v_values', (total_experiments,), dtype='f4')
            f.create_dataset('T_stop_values', (total_experiments,), dtype='i4')
            f.create_dataset('fdp_values', (total_experiments,), dtype='f4')
            f.create_dataset('L_values', (total_experiments,), dtype='i4')
        comm.Barrier()

    systems_per_process = N_systems // size
    remainder = N_systems % size
    start_idx = rank * systems_per_process + min(rank, remainder)
    end_idx = start_idx + systems_per_process + (1 if rank < remainder else 0)
    local_systems = list(range(start_idx, end_idx))
    
    if rank == 0: print(f"Processing {N_systems} systems across {size} MPI processes.")
    print(f"MPI Process {rank}: handling {len(local_systems)} systems ({start_idx}-{end_idx-1})")
    
    results = []
    if local_systems:
        pbar = tqdm(total=len(local_systems), desc=f"MPI Process {rank}", disable=not (show_progress and rank == 0))
        
        all_v = np.arange(0.5, 1, 1/K)
        all_v = np.append(all_v, 1 - np.finfo(float).eps)
        all_T = range(1, T_stop_max + 1)
        all_pairs = [(v, T) for v in all_v for T in all_T]

        for system_id in local_systems:
            result = process_system(({
                'system_id': system_id, 'X_files': X_files, 'X_dir': X_dir, 'beta_files': beta_files,
                'beta_dir': beta_dir, 'y': y, 'K': K, 'T_stop_max': T_stop_max,
                'num_dummies': num_dummies, 'num_dummies_factor': num_dummies_factor,
                'trex_parallel_process': trex_parallel_process, 'storage_format': storage_format,
                'output_path': output_path, 'verbose': verbose, 'multilevel': multilevel,
                'all_pairs': all_pairs, 'test': test
            }))
            results.append(result)
            pbar.update(1)
        pbar.close()

    print(f"MPI Process {rank}: Completed processing {len(results)} systems.")
        
    all_local_results = {k: [] for k in ['betas', 'phi_mats', 'v_values', 'T_stop_values', 'fdp_values', 'experiment_ids', 'L_values']}
    if test:
        all_local_results['X_mats'] = []
        all_local_results['y_vecs'] = []
    for res in results: 
        if res: 
            for key in all_local_results: all_local_results[key].extend(res[key])
    
    if storage_format == 'hdf5':
        # Option 1: Sequential writing (simpler and more reliable)
        comm.Barrier()  # Ensure file is created
        
        # Each rank writes sequentially to avoid collective I/O issues
        for write_rank in range(size):
            if rank == write_rank:
                write_offset = start_idx * multilevel
                num_local_experiments = len(all_local_results['fdp_values'])
                
                if num_local_experiments > 0:
                    print(f"MPI Process {rank}: Writing {num_local_experiments} experiments to HDF5 at offset {write_offset}")
                    
                    # Open file in sequential mode for this rank
                    with h5py.File(output_path, 'r+') as f:
                        keys_to_write = ['betas', 'phi_mats', 'v_values', 'T_stop_values', 'fdp_values', 'L_values']
                        if test:
                            keys_to_write.append('X_mats')
                            keys_to_write.append('y_vecs')
                        for key in keys_to_write:
                            dset = f[key]
                            local_data = np.array(all_local_results[key])
                            dset[write_offset:write_offset + num_local_experiments] = local_data
            
            comm.Barrier()  # Synchronize after each rank writes
        
        if rank == 0: 
            print(f"All results saved to: {output_path}")
        return output_path
    
    if rank == 0:
        print("Gathering results from all MPI processes...")
    all_process_results = comm.gather(all_local_results, root=0)

    if rank == 0:
        if storage_format == 'npz':
            combined = {k: [] for k in ['betas', 'phi_mats', 'v_values', 'T_stop_values', 'fdp_values', 'L_values']}
            if test:
                combined['X_mats'] = []
                combined['y_vecs'] = []
            for res in all_process_results: 
                if res: 
                    for key in combined:
                        if key in res:
                            combined[key].extend(res[key])
            
            save_dict = {
                "betas": np.array(combined['betas']),
                "phi_mats": np.array(combined['phi_mats']),
                "v_values": np.array(combined['v_values']),
                "T_stop_values": np.array(combined['T_stop_values']),
                "fdp_values": np.array(combined['fdp_values']),
                "L_values": np.array(combined['L_values'])
            }
            if test:
                save_dict['X_mats'] = np.array(combined['X_mats'])
                save_dict['y_vecs'] = np.array(combined['y_vecs'])
            np.savez(
                output_path,
                **save_dict
            )
        elif storage_format == 'json':
            total_experiments = sum(len(res['experiment_ids']) for res in all_process_results if res)
            print(f"Total results collected and saved as JSON: {total_experiments}")
        
        print(f"Final results saved to: {output_path}")
        return output_path
            
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate genomics data for FDP prediction using MPI.')
    
    parser.add_argument('--storage_format', type=str, default='json', choices=['json', 'npz', 'hdf5'], help="Storage format for the dataset (default: json)")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save data')
    parser.add_argument('--specific_folder', type=str, default=None, help='Specific folder name under data_dir')
    parser.add_argument('--T_stop_max', type=int, default=1, help='Maximum T_stop used')
    parser.add_argument('--K', type=int, default=20, help='Number of random experiments')
    parser.add_argument('--num_dummies', type=int, default=150,
                        help='Number of dummies. Default is 150.')
    parser.add_argument('--num_dummies_factor', type=int, default=1,
                        help='Factor for num_dummies range. Default is 1.')
    parser.add_argument('--SNR', type=float, default=1.0, help='Signal-to-noise ratio (used in folder name)')
    parser.add_argument('--show_progress', dest='show_progress', action='store_true', help='Show progress bar on rank 0 (default).')
    parser.add_argument('--no-show-progress', dest='show_progress', action='store_false', help='Do not show progress bar.')
    parser.set_defaults(show_progress=True)
    parser.add_argument('--trex_parallel_process', action='store_true', help='Use parallel processing in random_experiments. Default is False.')
    parser.add_argument('--N_systems_multiple', type=int, default=1, help='Multiplier for number of systems')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--multilevel', type=int, default=1, help='Number of v, T pairs to sample per system. Default is 1.')
    parser.add_argument('--test', action='store_true', help='Store X matrices for testing.')

    args = parser.parse_args()
    
    data_dir = generate_genomics_FDP_labeling(
        data_dir=args.data_dir,
        specific_folder=args.specific_folder,
        T_stop_max=args.T_stop_max,
        K=args.K,
        num_dummies=args.num_dummies,
        num_dummies_factor=args.num_dummies_factor,
        SNR=args.SNR,
        show_progress=args.show_progress,
        storage_format=args.storage_format,
        trex_parallel_process=args.trex_parallel_process,
        N_systems_multiple=args.N_systems_multiple,
        verbose=args.verbose,
        multilevel=args.multilevel,
        test=args.test
    )
