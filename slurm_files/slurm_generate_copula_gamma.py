import os
import numpy as np
import json
from tqdm import tqdm
from trexselector import random_experiments
from sklearn.preprocessing import StandardScaler
import time
import argparse
from mpi4py import MPI
from multiprocessing import Pool
import h5py
from scipy.stats import norm, gamma as gamma_dist


def create_subdir(subdir):
    """Create a subdirectory if it doesn't exist."""
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

def process_system_copula_gamma(args):
    """
    Process a single system using a Gaussian Copula with Gamma marginals.
    Args is a tuple/dict containing all necessary parameters.
    """
    # Unpack arguments
    system_id = args['system_id']
    n = args['n']
    p = args['p']
    K = args['K']
    T_stop_max = args['T_stop_max']
    num_act_range = args['num_act_range']
    num_dummies = args['num_dummies']
    num_dummies_factor = args['num_dummies_factor']
    SNR_range = args['SNR_range']
    rho_range = args['rho_range']
    shape_range = args['shape_range']
    storage_format = args['storage_format']
    output_path = args['output_path']
    seed_offset = args['seed_offset']
    verbose = args['verbose']
    multilevel = args.get('multilevel', 1)
    all_pairs = args.get('all_pairs')
    test = args.get('test', False)
    
    # Set seed for reproducibility
    np.random.seed(seed_offset + system_id)

    # Determine SNR for this system
    if SNR_range[0] == SNR_range[1]:
        SNR = SNR_range[0]
    else:
        # Biased uniform: 2/3 probability from first third of range, 1/3 from last two-thirds
        SNR = np.random.uniform(SNR_range[0], SNR_range[0] + (SNR_range[1]-SNR_range[0])/3) if np.random.rand() < 2/3 else np.random.uniform(SNR_range[0] + (SNR_range[1]-SNR_range[0])/3, SNR_range[1])

    
    # Determine num_act for this system
    if num_act_range[0] == num_act_range[1]:
        num_act = num_act_range[0]
    else:
        num_act = np.random.randint(num_act_range[0], num_act_range[1] + 1)

    # Create beta vector with random active indices
    beta_coeffs = np.zeros(p)
    active_indices = np.random.choice(p, num_act, replace=False)
    beta_coeffs[active_indices] = 1
    
    # 1. Sample parameters for copula and marginals
    rho = np.random.uniform(rho_range[0], rho_range[1])
    shape = np.random.uniform(shape_range[0], shape_range[1])
    if shape <= 0:
        raise ValueError("Shape must be positive.")

    # 2. Generate correlated standard normal data
    exponents = np.abs(np.arange(p) - np.arange(p)[:, np.newaxis])
    Sigma = rho ** exponents
    Z = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

    # 3. Apply normal CDF to get uniform marginals
    U = norm.cdf(Z)
    # Clip values to avoid inf issues with ppf at 1.0
    U = np.clip(U, 1e-9, 1 - 1e-9)

    # 4. Apply inverse CDF of Gamma to get target marginals
    # (scale is fixed to 1)
    X_raw = gamma_dist.ppf(U, a=shape)

    # 5. Standardize the result
    mean_gamma = shape # Mean of Gamma(shape, scale=1)
    var_gamma = shape  # Variance of Gamma(shape, scale=1)
    std_dev_gamma = np.sqrt(var_gamma)

    if std_dev_gamma > 1e-6:
        X = (X_raw - mean_gamma) / std_dev_gamma
    else:
        X = X_raw - mean_gamma
    
    # Calculate noise standard deviation for the given SNR
    sd = np.sqrt(np.var(X @ beta_coeffs) / SNR)
    y = X @ beta_coeffs + np.random.normal(size=n, scale=sd)
    
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
    
    parallel_process_trex = args.get('parallel_process_trex', False)

    if verbose:
        start_time = time.time()
    scaler = StandardScaler()
    res_exp = random_experiments(scaler.fit_transform(X), y - np.mean(y), K=K, T_stop=T_stop_max, num_dummies=int(L), 
                              verbose=False, parallel_process=parallel_process_trex)
    if verbose:
        print(f"System {system_id}: Time taken for random_experiments: {time.time() - start_time:.2f} seconds")
    Phi_mat = res_exp["phi_T_mat"]
    # Storage for this system's results
    system_results = {
        'betas': [], 'phi_mats': [], 'v_values': [],
        'T_stop_values': [], 'fdp_values': [], 'experiment_ids': [],
        'L_values': []
    }
    if test:
        system_results['X_mats'] = []
        system_results['y_vecs'] = []

    # Sample `multilevel` pairs from the pre-generated list
    num_to_sample = min(multilevel, len(all_pairs))
    indices = np.random.choice(len(all_pairs), num_to_sample, replace=False)
    pairs_to_process = [all_pairs[i] for i in indices]

    for i, (v, T) in enumerate(pairs_to_process):
        FDP = np.sum((1 - beta_coeffs) * (Phi_mat[:, T-1] > v), axis=0) / np.maximum(1, np.sum(Phi_mat[:, T-1] > v, axis=0))
        
        # Create unique experiment ID
        global_exp_id = (system_id * multilevel) + i
        
        # Save as JSON if requested
        if storage_format == 'json' and output_path is not None:
            experiment_data = {
                "beta": beta_coeffs.tolist(), "Phi_mat": Phi_mat.tolist(), "v": float(v),
                "T_stop": int(T), "FDP": float(FDP), "system_id": int(system_id),
                "experiment_id": int(global_exp_id),
                "L": int(L)
            }
            if test:
                experiment_data['X_mat'] = X.tolist()
                experiment_data["y_vec"] = y.tolist()
            json_path = os.path.join(output_path, f"experiment_{global_exp_id}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
        
        # Append results for this v and T
        system_results['betas'].append(beta_coeffs.copy())
        system_results['phi_mats'].append(Phi_mat.copy())
        if test:
            system_results['X_mats'].append(X.copy())
            system_results['y_vecs'].append(y.copy())
        system_results['v_values'].append(v)
        system_results['T_stop_values'].append(T)
        system_results['fdp_values'].append(FDP)
        system_results['experiment_ids'].append(global_exp_id)
        system_results['L_values'].append(L)
        
    return system_results

def generate_copula_gamma_FDP_labeling_hybrid(data_dir="data", specific_folder=None, T_stop_max=1, 
                                           N_systems=100, n=75, p=150, K=100, num_act_range=[1, 10], 
                                           num_dummies=150, num_dummies_factor=1, SNR_range=[0.5, 3.0], 
                                           rho_range=(0.1, 0.9), shape_range=(0.5, 10.0),
                                           show_progress=True, storage_format='json', n_cores_per_node=None, 
                                           verbose=False, multilevel=1, node_level_multiprocessing=True,
                                           trex_parallel_process=False, test=False):
    """
    Generates data with correlated Gamma predictors using a Gaussian Copula.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"MPI initialized with {size} processes. Storage format: {storage_format.upper()}")
        if storage_format == 'npz' and N_systems * multilevel > 50000:
            print("Warning: Using 'npz' format with a large number of systems/experiments can lead to high memory usage on the root node.")

    if specific_folder is None:
        num_dummies_str = f"num_dummies={num_dummies}" if num_dummies_factor == 1 else f"num_dummies=[{num_dummies}, {num_dummies_factor * num_dummies}]"
        SNR_str = f"SNR={SNR_range[0]}" if SNR_range[0] == SNR_range[1] else f"SNR=[{SNR_range[0]}, {SNR_range[1]}]"
        num_act_str = f"num_act={num_act_range[0]}" if num_act_range[0] == num_act_range[1] else f"num_act=[{num_act_range[0]}, {num_act_range[1]}]"
        param_order = [
            f"N_systems={N_systems}", SNR_str, f"T_stop_max={T_stop_max}",
            f"n={n}", num_act_str, num_dummies_str,
            f"p={p}", f"K={K}", 
            f"rho_min={rho_range[0]}", f"rho_max={rho_range[1]}",
            f"shape_min={shape_range[0]}", f"shape_max={shape_range[1]}",
        ]
        folder_prefix = f"(copula_gamma_multilevel_{multilevel})_" if multilevel > 1 else "(copula_gamma)_"
        if test:
            folder_prefix = folder_prefix.replace(")_", "_test)_")
        specific_folder = folder_prefix + ','.join(param_order)
    
    # Determine output path on rank 0
    output_path = None
    if rank == 0:
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
            comm.Barrier()
            return None
        
        total_experiments = N_systems * multilevel
        if rank == 0: print(f"Creating HDF5 file at {output_path} for {total_experiments} experiments.")
        
        with h5py.File(output_path, 'w', driver='mpio', comm=comm) as f:
            f.create_dataset('betas', (total_experiments, p), dtype='f4')
            f.create_dataset('phi_mats', (total_experiments, p, T_stop_max), dtype='f4')
            if test:
                f.create_dataset('X_mats', (total_experiments, n, p), dtype='f4')
                f.create_dataset('y_vecs', (total_experiments, n), dtype='f4')
            f.create_dataset('v_values', (total_experiments,), dtype='f4')
            f.create_dataset('T_stop_values', (total_experiments,), dtype='i4')
            f.create_dataset('fdp_values', (total_experiments,), dtype='f4')
            f.create_dataset('L_values', (total_experiments,), dtype='i4')
        comm.Barrier()

    # Distribute work
    systems_per_process = N_systems // size
    remainder = N_systems % size
    start_idx = rank * systems_per_process + min(rank, remainder)
    end_idx = start_idx + systems_per_process + (1 if rank < remainder else 0)
    local_systems = list(range(start_idx, end_idx))
    
    if rank == 0: print(f"Processing {N_systems} systems across {size} MPI processes.")
    print(f"MPI Process {rank}: handling {len(local_systems)} systems ({start_idx}-{end_idx-1})")
    
    results = []
    if local_systems:
        if n_cores_per_node is None: n_cores_per_node = os.cpu_count()
        print(f"MPI Process {rank}: Using {n_cores_per_node} cores for multiprocessing")
        
        all_v = np.arange(0.5, 1, 1/K)
        all_v = np.append(all_v, 1 - np.finfo(float).eps)
        all_T = range(1, T_stop_max + 1)
        all_pairs = [(v, T) for v in all_v for T in all_T]

        system_args = [{'system_id': sid, 'n': n, 'p': p, 'K': K, 'T_stop_max': T_stop_max,
                        'num_act_range': num_act_range, 'num_dummies': num_dummies, 'num_dummies_factor': num_dummies_factor, 'SNR_range': SNR_range, 
                        'rho_range': rho_range, 'shape_range': shape_range,
                        'storage_format': storage_format, 'output_path': output_path,
                        'seed_offset': rank * 100000, 'verbose': verbose,
                        'multilevel': multilevel, 'all_pairs': all_pairs,
                        'parallel_process_trex': trex_parallel_process, 'test': test} for sid in local_systems]
        
        use_multiprocessing = node_level_multiprocessing and n_cores_per_node > 1 and len(local_systems) > 1
        if use_multiprocessing:
            with Pool(processes=min(n_cores_per_node, len(local_systems))) as pool:
                map_func = pool.imap if show_progress and rank == 0 else pool.map
                pbar = tqdm(map_func(process_system_copula_gamma, system_args), total=len(local_systems), 
                            desc=f"MPI Process {rank} (Multiprocessing)", disable=not (show_progress and rank == 0))
                results = list(pbar)
        else:
            pbar = tqdm(system_args, desc=f"MPI Process {rank} (Single-threaded)", disable=not (show_progress and rank == 0))
            results = [process_system_copula_gamma(args) for args in pbar]

    print(f"MPI Process {rank}: Completed processing {len(results)} systems.")
    
    all_local_results = {k: [] for k in ['betas', 'phi_mats', 'v_values', 'T_stop_values', 'fdp_values', 'experiment_ids', 'L_values']}
    if test:
        all_local_results['X_mats'] = []
        all_local_results['y_vecs'] = []
    for res in results:
        if res:
            for key in all_local_results: all_local_results[key].extend(res[key])


    
    if storage_format == 'hdf5':
        comm.Barrier()
        for write_rank in range(size):
            if rank == write_rank:
                write_offset = start_idx * multilevel
                num_local_experiments = len(all_local_results['fdp_values'])
                
                if num_local_experiments > 0:
                    print(f"MPI Process {rank}: Writing {num_local_experiments} experiments to HDF5 at offset {write_offset}")
                    
                    with h5py.File(output_path, 'r+') as f:
                        keys_to_write = ['betas', 'phi_mats', 'v_values', 'T_stop_values', 'fdp_values', 'L_values']
                        if test:
                            keys_to_write.append('X_mats')
                            keys_to_write.append('y_vecs')
                        for key in keys_to_write:
                            dset = f[key]
                            local_data = np.array(all_local_results[key])
                            dset[write_offset:write_offset + num_local_experiments] = local_data
            
            comm.Barrier()
        
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
            
            print(f"Total results collected: {len(combined['fdp_values'])}")
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
    parser = argparse.ArgumentParser(description='Generate correlated Gamma data using a Gaussian Copula.')
    
    parser.add_argument('--storage_format', type=str, default='json', choices=['json', 'npz', 'hdf5'], help="Storage format for the dataset (default: json)")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save data')
    parser.add_argument('--specific_folder', type=str, default=None, help='Specific folder name under data_dir')
    parser.add_argument('--T_stop_max', type=int, default=1, help='Maximum T_stop used')
    parser.add_argument('--N_systems', type=int, default=100, help='Number of different systems')
    parser.add_argument('--n', type=int, default=75, help='Number of observations')
    parser.add_argument('--p', type=int, default=150, help='Number of variables')
    parser.add_argument('--K', type=int, default=100, help='Number of random experiments')
    parser.add_argument('--num_act_range', type=str, default="1,10",
                        help='Range of number of active variables as comma-separated values, e.g. 1,10')
    parser.add_argument('--num_dummies', type=int, default=150,
                        help='Number of dummies. Default is 150.')
    parser.add_argument('--num_dummies_factor', type=int, default=1,
                        help='Factor for num_dummies range. Default is 1.')
    parser.add_argument('--SNR_range', type=str, default="0.5,3.0",
                        help='Range for Signal-to-noise ratio as comma-separated values, e.g. 0.5,3.0')
    
    parser.add_argument('--rho_min', type=float, default=0.1, help='Minimum rho for AR(1) covariance')
    parser.add_argument('--rho_max', type=float, default=0.9, help='Maximum rho for AR(1) covariance')
    parser.add_argument('--shape_min', type=float, default=0.5, help='Minimum shape for Gamma marginals')
    parser.add_argument('--shape_max', type=float, default=10.0, help='Maximum shape for Gamma marginals')

    parser.add_argument('--show_progress', dest='show_progress', action='store_true', help='Show progress bar (only on rank 0)')
    parser.add_argument('--no-show-progress', dest='show_progress', action='store_false')
    parser.set_defaults(show_progress=True)
    parser.add_argument('--n_cores_per_node', type=int, default=None, help='Number of cores to use per MPI node')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--multilevel', type=int, default=1, help='Number of v, T pairs to sample per system')
    parser.add_argument('--node_level_multiprocessing', dest='node_level_multiprocessing', action='store_true', help='Enable multiprocessing at the node level (default).')
    parser.add_argument('--no-node-level-multiprocessing', dest='node_level_multiprocessing', action='store_false', help='Disable multiprocessing at the node level.')
    parser.set_defaults(node_level_multiprocessing=True)
    parser.add_argument('--trex_parallel_process', action='store_true', help='Enable parallel processing inside trexselector.random_experiments. Default is False.')
    parser.add_argument('--test', action='store_true', help='Store X matrices for testing. Default is False.')

    args = parser.parse_args()

    args.num_act_range = [int(x) for x in args.num_act_range.split(',')]
    args.SNR_range = [float(x) for x in args.SNR_range.split(',')]

    if abs(args.rho_min) >= 1 or abs(args.rho_max) >= 1:
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            parser.error("rho values must be between -1 and 1 (exclusive).")
    if args.shape_min <= 0:
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            parser.error("shape_min must be greater than 0.")
    
    generate_copula_gamma_FDP_labeling_hybrid(
        data_dir=args.data_dir,
        specific_folder=args.specific_folder,
        T_stop_max=args.T_stop_max,
        N_systems=args.N_systems,
        n=args.n,
        p=args.p,
        K=args.K,
        num_act_range=args.num_act_range,
        num_dummies=args.num_dummies,
        num_dummies_factor=args.num_dummies_factor,
        SNR_range=args.SNR_range,
        rho_range=(args.rho_min, args.rho_max),
        shape_range=(args.shape_min, args.shape_max),
        show_progress=args.show_progress,
        storage_format=args.storage_format,
        n_cores_per_node=args.n_cores_per_node,
        verbose=args.verbose,
        multilevel=args.multilevel,
        node_level_multiprocessing=args.node_level_multiprocessing,
        trex_parallel_process=args.trex_parallel_process,
        test=args.test
    )
