import os
import numpy as np
import json
import time
from tqdm import tqdm
from trexselector import random_experiments
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool


def create_subdir(subdir):
    """Create a subdirectory if it doesn't exist."""
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)


def setup_directory(data_dir, specific_folder):
    """
    Set up the directory structure for data generation.
    Creates data_dir if it doesn't exist.
    If specific_folder exists, removes it completely and creates a fresh directory.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
    specific_folder : str
        Name of the specific generation folder
    
    Returns
    -------
    str
        Path to the base directory for this generation
    """
    create_subdir(data_dir)
    return os.path.join(data_dir, specific_folder)


def FDP_inv(target_FDP, phi_T, true_actives, K, T_stop=1):
    """
    Find the value of v such that FDP <= target_FDP.
    
    Parameters
    ----------
    target_FDP : float
        Target FDP
    phi_T : ndarray
        Phi matrix
    true_actives : ndarray
        Indices of active variables
    K : int
        Number of random experiments
    T_stop : int, default=1
        T_stop
        
    Returns
    -------
    float
        Threshold v
    """
    for v in np.arange(0.5, 1 + 1/K, 1/K):
        not_true_actives = np.ones(phi_T.shape[0], dtype=bool)
        not_true_actives[true_actives] = False
        
        FDP = np.sum(phi_T[not_true_actives, T_stop-1] > v) / max(np.sum(phi_T[:, T_stop-1] > v), 1)
        
        if FDP <= target_FDP:
            return v
    
    return 1.0


def npz_to_json(npz_file_path, show_progress=True):
    """
    Convert an existing .npz data file to a folder of JSON files.
    
    Parameters
    ----------
    npz_file_path : str
        Path to the .npz file to convert
    show_progress : bool, default=True
        Whether to show a progress bar during conversion
        
    Returns
    -------
    str
        Path to the directory containing the JSON files
    """
    # Check if the file exists
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"The file {npz_file_path} does not exist")
    
    # Extract the base name without extension to use as folder name
    folder_name = os.path.splitext(os.path.basename(npz_file_path))[0]
    parent_dir = os.path.dirname(npz_file_path)
    
    # Create the output folder
    output_folder = os.path.join(parent_dir, folder_name)
    create_subdir(output_folder)
    
    # Load the .npz file
    data = np.load(npz_file_path, allow_pickle=True)
    
    # Determine the type of data based on keys
    if 'v_thresholds' in data:
        # This is data from generate_data_pred_v
        data_type = 'v_prediction'
        betas = data['betas']
        true_actives = data['true_actives']
        phi_mats = data['phi_mats']
        v_thresholds = data['v_thresholds']
        L_values = data['L_values']
        
        total = len(betas)
        
        # Setup progress bar
        if show_progress:
            pbar = tqdm(total=total, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%)')
            pbar.set_description("Converting to JSON")
        
        for i in range(total):
            experiment_data = {
                "beta": betas[i].tolist(),
                "true_actives": true_actives[i].tolist(),
                "Phi_mat": phi_mats[i].tolist(),
                "v_thresh": float(v_thresholds[i]),
                "L": int(L_values[i]),
                "experiment_id": i + 1
            }
            
            # Save to a JSON file
            json_path = os.path.join(output_folder, f"experiment_{i+1}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    elif 'v_values' in data and 'fdp_values' in data:
        # This is data from generate_data_pred_FDP_all_v or generate_data_pred_FDP_single_v
        betas = data['betas']
        phi_mats = data['phi_mats']
        v_values = data['v_values']
        fdp_values = data['fdp_values']
        L_values = data['L_values']
        
        # Determine format based on data shape
        single_v_type = len(betas) == len(phi_mats) and len(phi_mats) == len(v_values)
        
        total = len(betas)
        
        # Setup progress bar
        if show_progress:
            pbar = tqdm(total=total, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%)')
            pbar.set_description("Converting to JSON")
        
        # Check if format has multiple v values per system
        if not single_v_type:
            # Determine the number of systems and v values per system
            unique_phis = {}
            system_id = 1
            v_index = 1
            
            for i in range(total):
                # Create hash of the Phi matrix to identify unique systems
                phi_hash = hash(phi_mats[i].tobytes())
                
                if phi_hash not in unique_phis:
                    unique_phis[phi_hash] = system_id
                    system_id += 1
                    v_index = 1
                else:
                    v_index += 1
                
                experiment_data = {
                    "beta": betas[i].tolist(),
                    "Phi_mat": phi_mats[i].tolist(),
                    "v": float(v_values[i]),
                    "FDP": float(fdp_values[i]),
                    "L": int(L_values[i]),
                    "system_id": unique_phis[phi_hash],
                    "v_index": v_index,
                    "experiment_id": i + 1
                }
                
                # Save to a JSON file
                json_path = os.path.join(output_folder, f"experiment_{i+1}.json")
                with open(json_path, 'w') as f:
                    json.dump(experiment_data, f)
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
        else:
            # Single v value per system format
            for i in range(total):
                experiment_data = {
                    "beta": betas[i].tolist(),
                    "Phi_mat": phi_mats[i].tolist(),
                    "v": float(v_values[i]),
                    "FDP": float(fdp_values[i]),
                    "L": int(L_values[i]),
                    "experiment_id": i + 1
                }
                
                # Save to a JSON file
                json_path = os.path.join(output_folder, f"experiment_{i+1}.json")
                with open(json_path, 'w') as f:
                    json.dump(experiment_data, f)
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    else:
        raise ValueError("Unknown .npz file format. The file should contain data generated by one of the generate_data_* functions.")
    
    return output_folder


def json_to_npz(json_folder_path, output_file=None, show_progress=True):
    """
    Convert a folder of JSON files back to a single compressed .npz file.
    This function is the inverse of npz_to_json.
    
    Parameters
    ----------
    json_folder_path : str
        Path to the folder containing JSON files
    output_file : str, optional
        Path where the .npz file will be saved. If None, will use the folder name with .npz extension
    show_progress : bool, default=True
        Whether to show progress during conversion
        
    Returns
    -------
    str
        Path to the created .npz file
    """
    # Check if the folder exists
    if not os.path.exists(json_folder_path) or not os.path.isdir(json_folder_path):
        raise FileNotFoundError(f"The folder {json_folder_path} does not exist or is not a directory")
    
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    if not json_files:
        raise ValueError(f"No JSON files found in {json_folder_path}")
    
    # Sort files by experiment ID to ensure consistent order
    json_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Determine output file path
    if output_file is None:
        output_file = os.path.join(os.path.dirname(json_folder_path), os.path.basename(json_folder_path) + '.npz')
    
    # Load the first JSON file to determine the data type
    with open(os.path.join(json_folder_path, json_files[0]), 'r') as f:
        first_file = json.load(f)
    
    # Initialize arrays to store data
    betas = []
    phi_mats = []
    
    total = len(json_files)
    
    # Setup progress bar
    if show_progress:
        pbar = tqdm(total=total, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%)')
        pbar.set_description("Converting to NPZ")
    
    # Check the data type based on keys in the first file
    if 'v_thresh' in first_file:
        # This is data from generate_data_pred_v
        data_type = 'v_prediction'
        true_actives_list = []
        v_thresholds = []
        L_values = []
        
        for i, file_name in enumerate(json_files):
            with open(os.path.join(json_folder_path, file_name), 'r') as f:
                data = json.load(f)
            
            betas.append(np.array(data['beta']))
            true_actives_list.append(np.array(data['true_actives']))
            phi_mats.append(np.array(data['Phi_mat']))
            v_thresholds.append(float(data['v_thresh']))
            L_values.append(int(data['L']))
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
        
        # Save data to npz file
        np.savez(
            output_file,
            betas=np.array(betas),
            true_actives=np.array(true_actives_list, dtype=object),  # Use object dtype for variable length arrays
            phi_mats=np.array(phi_mats),
            v_thresholds=np.array(v_thresholds),
            L_values=np.array(L_values)
        )
    
    elif 'v' in first_file and 'FDP' in first_file:
        # This is data from generate_data_pred_FDP_all_v or generate_data_pred_FDP_single_v
        v_values = []
        fdp_values = []
        L_values = []
        
        for i, file_name in enumerate(json_files):
            with open(os.path.join(json_folder_path, file_name), 'r') as f:
                data = json.load(f)
            
            betas.append(np.array(data['beta']))
            phi_mats.append(np.array(data['Phi_mat']))
            v_values.append(float(data['v']))
            fdp_values.append(float(data['FDP']))
            L_values.append(int(data['L']))
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
        
        # Save data to npz file
        np.savez(
            output_file,
            betas=np.array(betas),
            phi_mats=np.array(phi_mats),
            v_values=np.array(v_values),
            fdp_values=np.array(fdp_values),
            L_values=np.array(L_values)
        )
    
    else:
        raise ValueError("Unknown JSON file format. The files should contain data generated by one of the generate_data_* functions.")
    
    if show_progress:
        pbar.close()
    
    return output_file


def generate_data_pred_v(data_dir="data", specific_folder=None, target_FDP=0.1, T_stop=1, N_data=100, 
                         n=75, p=150, K=100, num_act=3, num_dummies=150, SNR=1.0, 
                         show_progress=True, save_as_json=False, use_multiprocessing=False):
    """
    Generates beta, true_actives, Phi_mat, v_thresh (objective, train v predicting net directly).
    Data is saved as a single .npz file or as individual JSON files.
    
    Parameters
    ----------
    data_dir : str, default="data"
        Directory to save data
    specific_folder : str, optional
        Specific folder name under data_dir. If None, will be generated from parameters
    target_FDP : float, default=0.1
        Target FDP
    T_stop : int, default=1
        T_stop
    N_data : int, default=100
        Dataset size to create
    n : int, default=75
        Number of observations
    p : int, default=150
        Number of variables
    K : int, default=100
        Number of random experiments
    num_act : int, default=3
        Number of active variables
    num_dummies : int, default=150
        Number of dummies
    SNR : float, default=1.0
        SNR
    show_progress : bool, default=True
        Whether to show the progress bar
    save_as_json : bool, default=False
        If True, save each experiment as a separate JSON file in a folder
    use_multiprocessing : bool, default=False
        If True, use parallel processing in random_experiments
        
    Returns
    -------
    str
        Path to the npz file or the directory containing JSON files
    """
    if specific_folder is None:
        # Use a specific ordering for parameters in the filename
        param_order = [
            (f"N_data={N_data}"),
            (f"target_FDP={target_FDP}"),
            (f"SNR={SNR}"),
            (f"T_stop={T_stop}"),
            (f"n={n}"),
            (f"num_act={num_act}"),
            (f"num_dummies={num_dummies}"),
            (f"p={p}"),
            (f"K={K}"),
        ]
        specific_folder = "(v)_" + ','.join(param_order)
    
    # Create data directory if it doesn't exist
    create_subdir(data_dir)
    
    # Create a folder for JSON files if needed
    json_folder = None
    if save_as_json:
        json_folder = os.path.join(data_dir, specific_folder)
        create_subdir(json_folder)
    
    # Lists to store results
    betas = []
    true_actives_list = []
    phi_mats = []
    v_thresholds = []
    L_values = []
    
    # Setup progress bar
    iterator = range(1, N_data + 1)
    if show_progress:
        pbar = tqdm(total=N_data, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%)')
        pbar.set_description("Generating data")
    
    for i in iterator:
        beta = np.zeros(p)
        active_indices = np.random.choice(p, num_act, replace=False)
        beta[active_indices] = 1
        true_actives = np.where(beta > 0)[0]
        
        X = np.random.normal(size=(n, p))
        sd = np.sqrt(np.var(X @ beta) / SNR)
        y = X @ beta + np.random.normal(size=n, scale=sd)
        
        # Use parallel processing in random_experiments if requested
        res_exp = random_experiments(X, y, K=K, T_stop=T_stop, num_dummies=num_dummies, 
                                  verbose=False, parallel_process=use_multiprocessing)
        
        Phi_mat = res_exp["phi_T_mat"]
        v_thresh = FDP_inv(target_FDP, Phi_mat, true_actives, K, T_stop)
        
        # Save as JSON if requested
        if save_as_json:
            experiment_data = {
                "beta": beta.tolist(),
                "true_actives": true_actives.tolist(),
                "Phi_mat": Phi_mat.tolist(),
                "v_thresh": float(v_thresh),
                "L": p,
                "experiment_id": i
            }
            
            json_path = os.path.join(json_folder, f"experiment_{i}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
        
        # Store data for .npz file
        betas.append(beta)
        true_actives_list.append(true_actives)
        phi_mats.append(Phi_mat)
        v_thresholds.append(v_thresh)
        L_values.append(p)
        
        # Update progress bar
        if show_progress:
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    if not save_as_json:
        # Save all data to a single npz file
        output_file = os.path.join(data_dir, f"{specific_folder}.npz")
        np.savez(
            output_file,
            betas=np.array(betas),
            true_actives=np.array(true_actives_list, dtype=object),  # Use object dtype for variable length arrays
            phi_mats=np.array(phi_mats),
            v_thresholds=np.array(v_thresholds),
            L_values=np.array(L_values)
        )
        return output_file
    else:
        return json_folder


def generate_data_pred_FDP_all_v(data_dir="data", specific_folder=None, T_stop_max=1, N_systems=100, 
                                 n=75, p=150, K=100, num_act=3, num_dummies=150, SNR=1.0, 
                                 show_progress=True, save_as_json=False, use_multiprocessing=False):
    """
    Generates beta, Phi, v, T_stop, FDP (objective, train FDP predicting net and then do minimization on it)
    for a single Phi matrix, saves the FDP for each v and T_stop in a single .npz file or as individual JSON files.
    
    Parameters
    ----------
    data_dir : str, default="data"
        Directory to save data
    specific_folder : str, optional
        Specific folder name under data_dir. If None, will be generated from parameters
    T_stop_max : int, default=1
        Maximum T_stop used
    N_systems : int, default=100
        Number of different systems that create the dataset
    n : int, default=75
        Number of observations
    p : int, default=150
        Number of variables
    K : int, default=100
        Number of random experiments
    num_act : int, default=3
        Number of active variables
    num_dummies : int, default=150
        Number of dummies
    SNR : float, default=1.0
        SNR
    show_progress : bool, default=True
        Whether to show the progress bar
    save_as_json : bool, default=False
        If True, save each experiment as a separate JSON file in a folder
    use_multiprocessing : bool, default=False
        If True, use parallel processing in random_experiments
        
    Returns
    -------
    str
        Path to the npz file or the directory containing JSON files
    """
    if specific_folder is None:
        # Use a specific ordering for parameters in the filename
        param_order = [
            (f"N_systems={N_systems}"),
            (f"SNR={SNR}"),
            (f"T_stop_max={T_stop_max}"),
            (f"n={n}"),
            (f"num_act={num_act}"),
            (f"num_dummies={num_dummies}"),
            (f"p={p}"),
            (f"K={K}"),
        ]
        specific_folder = "(all FDP)_" + ','.join(param_order)
    
    # Create data directory if it doesn't exist
    create_subdir(data_dir)
    
    # Create a folder for JSON files if needed
    json_folder = None
    if save_as_json:
        json_folder = os.path.join(data_dir, specific_folder)
        create_subdir(json_folder)
    
    # Lists to store results
    betas = []
    phi_mats = []
    v_values = []
    T_stop_values = []
    fdp_values = []
    L_values = []
    
    total_iterations = N_systems * (int(0.5*K) + 1) * T_stop_max
    
    # Setup progress bar
    if show_progress:
        pbar = tqdm(total=total_iterations, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%)')
        pbar.set_description("Generating data")
    
    experiment_id = 0

    for i in range(1, N_systems + 1):
        beta = np.zeros(p)
        active_indices = np.random.choice(p, num_act, replace=False)
        beta[active_indices] = 1
        
        X = np.random.normal(size=(n, p))
        sd = np.sqrt(np.var(X @ beta) / SNR)
        y = X @ beta + np.random.normal(size=n, scale=sd)
        
        # Use parallel processing in random_experiments if requested
        res_exp = random_experiments(X, y, K=K, T_stop=T_stop_max, num_dummies=num_dummies, 
                                  verbose=False, parallel_process=use_multiprocessing)
        Phi_mat = res_exp["phi_T_mat"]
        
        for j in range(1, int(0.5*K) + 2):
            v = 0.5 + (j - 1) / K
            for k in range(1, T_stop_max + 1):
                experiment_id += 1
                FDP = np.sum((1 - beta) * (Phi_mat[:, k-1] > v), axis=0) / np.maximum(1, np.sum(Phi_mat[:, k-1] > v, axis=0))
            
                # Save as JSON if requested
                if save_as_json:
                    experiment_data = {
                        "beta": beta.tolist(),
                        "Phi_mat": Phi_mat.tolist(),
                        "v": float(v),
                        "T_stop": k,
                        "FDP": float(FDP),
                        "L": p,
                        "system_id": i,
                        "v_index": j,
                        "experiment_id": experiment_id
                    }
                    
                    json_path = os.path.join(json_folder, f"experiment_{experiment_id}.json")
                    with open(json_path, 'w') as f:
                        json.dump(experiment_data, f)
            
                # Store data for .npz file
                # MAYBE ADD IF NOT SAVE_AS_JASON (or else)
                betas.append(beta)
                phi_mats.append(Phi_mat)
                v_values.append(v)
                T_stop_values.append(k)
                fdp_values.append(FDP)
                L_values.append(p)
            
                # Update progress bar
                if show_progress:
                    pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    if not save_as_json:
        # Save all data to a single npz file
        output_file = os.path.join(data_dir, f"{specific_folder}.npz")
        np.savez(
            output_file,
            betas=np.array(betas),
            phi_mats=np.array(phi_mats),
            v_values=np.array(v_values),
            T_stop_values=np.array(T_stop_values),
            fdp_values=np.array(fdp_values),
            L_values=np.array(L_values)
        )
        return output_file
    else:
        return json_folder


def generate_data_pred_FDP_single_v(data_dir="data", specific_folder=None, T_stop=1, N_systems=100, 
                                    n=75, p=150, K=100, num_act=3, num_dummies=150, SNR=1.0, 
                                    show_progress=True, save_as_json=False, use_multiprocessing=False):
    """
    Generates beta, Phi_mat, v, FDP (objective, train v predicting net directly)
    for a single Phi matrix, saves the FDP for a single v (randomly chosen within [0.5, 1])
    in a single .npz file or as individual JSON files.
    
    Parameters
    ----------
    data_dir : str, default="data"
        Directory to save data
    specific_folder : str, optional
        Specific folder name under data_dir. If None, will be generated from parameters
    T_stop : int, default=1
        T_stop
    N_systems : int, default=100
        Number of different systems that create the dataset
    n : int, default=75
        Number of observations
    p : int, default=150
        Number of variables
    K : int, default=100
        Number of random experiments
    num_act : int, default=3
        Number of active variables
    num_dummies : int, default=150
        Number of dummies
    SNR : float, default=1.0
        SNR
    show_progress : bool, default=True
        Whether to show the progress bar
    save_as_json : bool, default=False
        If True, save each experiment as a separate JSON file in a folder
    use_multiprocessing : bool, default=False
        If True, use parallel processing in random_experiments
        
    Returns
    -------
    str
        Path to the npz file or the directory containing JSON files
    """
    if specific_folder is None:
        # Use a specific ordering for parameters in the filename
        param_order = [
            (f"N_systems={N_systems}"),
            (f"SNR={SNR}"),
            (f"T_stop={T_stop}"),
            (f"n={n}"),
            (f"num_act={num_act}"),
            (f"num_dummies={num_dummies}"),
            (f"p={p}"),
            (f"K={K}"),
        ]
        specific_folder = "(single FDP)_" + ','.join(param_order)
    
    # Create data directory if it doesn't exist
    create_subdir(data_dir)
    
    # Create a folder for JSON files if needed
    json_folder = None
    if save_as_json:
        json_folder = os.path.join(data_dir, specific_folder)
        create_subdir(json_folder)
    
    # Lists to store results
    betas = []
    phi_mats = []
    v_values = []
    fdp_values = []
    L_values = []
    
    # Setup progress bar
    if show_progress:
        pbar = tqdm(total=N_systems, bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%)')
        pbar.set_description("Generating data")
    
    for i in range(1, N_systems + 1):
        beta = np.zeros(p)
        active_indices = np.random.choice(p, num_act, replace=False)
        beta[active_indices] = 1
        
        X = np.random.normal(size=(n, p))
        sd = np.sqrt(np.var(X @ beta) / SNR)
        y = X @ beta + np.random.normal(size=n, scale=sd)
        
        # Use parallel processing in random_experiments if requested
        res_exp = random_experiments(X, y, K=K, T_stop=T_stop, num_dummies=num_dummies, 
                                  verbose=False, parallel_process=use_multiprocessing)
        
        Phi_mat = res_exp["phi_T_mat"]
        Phi_mat = Phi_mat[:, -1]
        
        v = np.random.uniform(0.5, 1)
        FDP = np.sum((1 - beta) * (Phi_mat > v)) / max(1, np.sum(Phi_mat > v))
        
        # Save as JSON if requested
        if save_as_json:
            experiment_data = {
                "beta": beta.tolist(),
                "Phi_mat": Phi_mat.tolist(),
                "v": float(v),
                "FDP": float(FDP),
                "L": p,
                "experiment_id": i
            }
            
            json_path = os.path.join(json_folder, f"experiment_{i}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
        
        # Store data for .npz file
        betas.append(beta)
        phi_mats.append(Phi_mat)
        v_values.append(v)
        fdp_values.append(FDP)
        L_values.append(p)
        
        # Update progress bar
        if show_progress:
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    if not save_as_json:
        # Save all data to a single npz file
        output_file = os.path.join(data_dir, f"{specific_folder}.npz")
        np.savez(
            output_file,
            betas=np.array(betas),
            phi_mats=np.array(phi_mats),
            v_values=np.array(v_values),
            fdp_values=np.array(fdp_values),
            L_values=np.array(L_values)
        )
        return output_file
    else:
        return json_folder


def process_genomics_FDP_labeling(data_dir="data", specific_folder=None, T_stop_max=1,
                                   K=100, num_dummies=150, num_dummies_factor=1, SNR=1.0,
                                   show_progress=True, storage_format='npz', trex_parallel_process=False,
                                   N_systems_multiple=1, verbose=False, multilevel=1, test=False):
    """
    Generates FDP data using fixed genomics data with random permutations.
    This is a local version of the slurm_process_genomics.py script.

    Parameters
    ----------
    data_dir : str, default="data"
        Directory to save data
    specific_folder : str, optional
        Specific folder name under data_dir. If None, will be generated from parameters
    T_stop_max : int, default=1
        Maximum T_stop used
    K : int, default=100
        Number of random experiments
    num_dummies : int, default=150
        Number of dummies
    num_dummies_factor : int, default=1
        Factor for num_dummies range. Default is 1.
    SNR : float, default=1.0
        SNR
    show_progress : bool, default=True
        Whether to show the progress bar
    storage_format : str, default='npz'
        Storage format for the dataset. Options: 'json', 'npz'.
    trex_parallel_process : bool, default=False
        If True, use parallel processing in random_experiments
    N_systems_multiple : int, default=1
        Number of systems will be N_systems_multiple * number_of_X_matrices
    verbose : bool, default=False
        Enable verbose output.
    multilevel : int, default=1
        Number of v, T pairs to sample per system. Default is 1.
    test : bool, default=False
        Store X matrices for testing.

    Returns
    -------
    str
        Path to the npz file or the directory containing JSON files
    """
    X_dir = os.path.join("genomics_data", "preprocessed_matrices")
    if not os.path.exists(X_dir): raise FileNotFoundError(f"X matrices directory not found at {X_dir}")
    X_files = sorted([f for f in os.listdir(X_dir) if f.startswith("X_") and f.endswith(".txt")])
    if not X_files: raise ValueError(f"No X matrix files found in {X_dir}")
    
    beta_dir = os.path.join("genomics_data", "preprocessed_betas")
    if not os.path.exists(beta_dir): raise FileNotFoundError(f"betas directory not found at {beta_dir}")
    beta_files = sorted([f for f in os.listdir(beta_dir) if f.startswith("beta_") and f.endswith(".txt")])
    if not beta_files: raise ValueError(f"No beta files found in {beta_dir}")
    
    y = np.array([0]*100 + [1]*200)
    N_systems = len(X_files) * N_systems_multiple
    p_padded = 545

    output_path = None
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
    
    all_results = {
        'betas': [], 'phi_mats': [], 'v_values': [],
        'T_stop_values': [], 'fdp_values': [], 'experiment_ids': [],
        'L_values': []
    }
    if test:
        all_results['X_mats'] = []
        all_results['y_vecs'] = []

    pbar = tqdm(total=N_systems, desc="Generating data", disable=not show_progress)
    
    all_v = np.arange(0.5, 1, 1/K)
    all_v = np.append(all_v, 1 - np.finfo(float).eps)
    all_T = range(1, T_stop_max + 1)
    all_pairs = [(v, T) for v in all_v for T in all_T]

    for system_id in range(N_systems):
        X_file = X_files[system_id % len(X_files)]
        X = np.loadtxt(os.path.join(X_dir, X_file), dtype=np.int16)  # uint8

        beta_file = beta_files[system_id % len(beta_files)]
        beta = np.loadtxt(os.path.join(beta_dir, beta_file), dtype=np.int16)  #uint8

        n = X.shape[0]
        perm = np.random.permutation(n)
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

        X_scaled = scaler.fit_transform(X_perm)
        y_cent = y_perm - np.mean(y_perm)
        res_exp = random_experiments(X_scaled, y_cent, K=K, T_stop=T_stop_max, num_dummies=5*L, 
                                  verbose=False, parallel_process=trex_parallel_process)
        if verbose:
            print(f"System {system_id}: Time taken for random_experiments: {time.time() - start_time:.2f} seconds")
        Phi_mat = res_exp["phi_T_mat"]

        if Phi_mat.shape[0] < p_padded:
            Phi_mat = np.pad(Phi_mat, ((0, p_padded - Phi_mat.shape[0]), (0, 0)), mode='constant', constant_values=0)
        if beta_perm.shape[0] < p_padded:
            beta_perm = np.pad(beta_perm, (0, p_padded - beta_perm.shape[0]), mode='constant', constant_values=0)

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
                    experiment_data['X_mat'] = X_perm.tolist()
                    experiment_data['y_vec'] = y_perm.tolist()
                json_path = os.path.join(output_path, f"experiment_{global_exp_id}.json")
                with open(json_path, 'w') as f:
                    json.dump(experiment_data, f)
            
            all_results['betas'].append(beta_perm.copy())
            all_results['phi_mats'].append(Phi_mat.copy())
            if test:
                all_results['X_mats'].append(X_perm.copy())
                all_results['y_vecs'].append(y_perm.copy())
            all_results['v_values'].append(v)
            all_results['T_stop_values'].append(T)
            all_results['fdp_values'].append(FDP)
            all_results['experiment_ids'].append(global_exp_id)
            all_results['L_values'].append(L)
        
        pbar.update(1)
    
    pbar.close()

    if storage_format == 'npz':
        save_dict = {
            "betas": np.array(all_results['betas']),
            "phi_mats": np.array(all_results['phi_mats']),
            "v_values": np.array(all_results['v_values']),
            "T_stop_values": np.array(all_results['T_stop_values']),
            "fdp_values": np.array(all_results['fdp_values']),
            "L_values": np.array(all_results['L_values'])
        }
        if test:
            save_dict['X_mats'] = np.array(all_results['X_mats'])
            save_dict['y_vecs'] = np.array(all_results['y_vecs'])
        np.savez(
            output_path,
            **save_dict
        )
    
    if output_path:
        print(f"Final results saved to: {output_path}")
    return output_path


def _process_system_gaussian(args):
    # Unpack arguments
    system_id = args['system_id']
    n = args['n']
    p = args['p']
    K = args['K']
    T_stop_max = args['T_stop_max']
    num_act = args['num_act']
    num_dummies = args['num_dummies']
    num_dummies_factor = args['num_dummies_factor']
    SNR = args['SNR']
    seed_offset = args['seed_offset']
    verbose = args['verbose']
    multilevel = args.get('multilevel', 1)
    all_pairs = args.get('all_pairs')
    trex_parallel_process = args.get('trex_parallel_process', False)
    test = args.get('test', False)

    # Set seed for reproducibility
    np.random.seed(seed_offset + system_id)
    
    # Create beta vector with random active indices
    beta = np.zeros(p)
    active_indices = np.random.choice(p, num_act, replace=False)
    beta[active_indices] = 1
    
    # Generate X and y
    X = np.random.normal(size=(n, p))
    sd = np.sqrt(np.var(X @ beta) / SNR)
    y = X @ beta + np.random.normal(size=n, scale=sd)
    
    # Calculate L based on num_dummies and num_dummies_factor
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
    res_exp = random_experiments(X, y, K=K, T_stop=T_stop_max, num_dummies=int(L), 
                              verbose=False, parallel_process=trex_parallel_process)
    if L < 1:
        raise ValueError(f"Calculated L is invalid: {L}. num_dummies={num_dummies}, num_dummies_factor={num_dummies_factor}")
    if verbose:
        print(f"System {system_id}: Time taken for random_experiments: {time.time() - start_time:.2f} seconds")
    if verbose:
        print(f"System {system_id}: L = {L}")
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
        FDP = np.sum((1 - beta) * (Phi_mat[:, T-1] > v), axis=0) / np.maximum(1, np.sum(Phi_mat[:, T-1] > v, axis=0))
        
        # Create unique experiment ID
        global_exp_id = (system_id * multilevel) + i
        
        # Append results for this v and T
        system_results['betas'].append(beta.copy())
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


def generate_gaussian_FDP_labeling(data_dir="data", specific_folder=None, T_stop_max=1, 
                                 N_systems=100, n=75, p=150, K=100, num_act=3, 
                                 num_dummies=150, num_dummies_factor=1, SNR=1.0, show_progress=True, 
                                 save_as_json=False, node_level_multiprocessing=False, trex_parallel_process=False,
                                 multilevel=1, verbose=False, test=False):
    """
    Generates beta, Phi, v, T_stop, FDP entries dataset, following gaussian dist.
    (objective, train FDP predicting net and then do minimization on it) saves in a single .npz file or as individual JSON files.
    
    Parameters
    ----------
    data_dir : str, default="data"
        Directory to save data
    specific_folder : str, optional
        Specific folder name under data_dir. If None, will be generated from parameters
    T_stop_max : int, default=1
        Maximum T_stop used
    N_systems : int, default=100
        Number of different systems that create the dataset
    n : int, default=75
        Number of observations
    p : int, default=150
        Number of variables
    K : int, default=100
        Number of random experiments
    num_act : int, default=3
        Number of active variables
    num_dummies : int, default=150
        Number of dummies
    SNR : float, default=1.0
        SNR
    show_progress : bool, default=True
        Whether to show the progress bar
    save_as_json : bool, default=False
        If True, save each experiment as a separate JSON file in a folder
    use_multiprocessing : bool, default=False
        If True, use parallel processing in random_experiments
    multilevel : int, default=1
        Number of v, T pairs to sample per system. If 1, samples one random pair.
        If > 1, samples that many pairs without replacement.
    verbose : bool, default=False
        Enable verbose output.
        
    Returns
    -------
    str
        Path to the npz file or the directory containing JSON files
    """
    if specific_folder is None:
        # Use a specific ordering for parameters in the filename
        num_dummies_str = f"num_dummies={num_dummies}" if num_dummies_factor == 1 else f"num_dummies=[{num_dummies}, {num_dummies_factor * num_dummies}]"
        param_order = [
            (f"N_systems={N_systems}"),
            (f"SNR={SNR}"),
            (f"T_stop_max={T_stop_max}"),
            (f"n={n}"),
            (f"num_act={num_act}"),
            num_dummies_str,
            (f"p={p}"),
            (f"K={K}"),
        ]
        if multilevel > 1:
            folder_prefix = f"(gaussian_multilevel_{multilevel})_"
        else:
            folder_prefix = "(gaussian)_"
        specific_folder = folder_prefix + ','.join(param_order)
    
    # Create data directory if it doesn't exist
    create_subdir(data_dir)
    
    # Create a folder for JSON files if needed
    json_folder = None
    if save_as_json:
        json_folder = os.path.join(data_dir, specific_folder)
        create_subdir(json_folder)
    
    # Lists to store results
    betas = []
    phi_mats = []
    v_values = []
    T_stop_values = []
    fdp_values = []
    experiment_ids = []

    # Generate all (v, T) pairs once
    all_v = np.arange(0.5, 1, 1/K)
    all_v = np.append(all_v, 1 - np.finfo(float).eps)
    all_T = range(1, T_stop_max + 1)
    all_pairs = [(v, T) for v in all_v for T in all_T]

    system_args = [{
        'system_id': sid, 'n': n, 'p': p, 'K': K, 'T_stop_max': T_stop_max,
        'num_act': num_act, 'num_dummies': num_dummies, 'num_dummies_factor': num_dummies_factor, 'SNR': SNR,
        'seed_offset': 0, 'verbose': verbose, # seed_offset is 0 as we are not using MPI ranks
        'multilevel': multilevel, 'all_pairs': all_pairs,
        'trex_parallel_process': trex_parallel_process,
        'test': test
    } for sid in range(N_systems)]
    
    results = []
    if N_systems > 0: # Only process if there are systems
        if node_level_multiprocessing:
            with Pool(processes=os.cpu_count()) as pool:
                map_func = pool.imap if show_progress else pool.map
                pbar = tqdm(map_func(_process_system_gaussian, system_args), total=N_systems, desc="Generating data", disable=not show_progress)
                results = list(pbar)
        else:
            pbar = tqdm(system_args, desc="Generating data", disable=not show_progress)
            results = [_process_system_gaussian(args) for args in pbar]

    # Aggregate results
    all_betas = []
    all_phi_mats = []
    all_v_values = []
    all_T_stop_values = []
    all_fdp_values = []
    all_experiment_ids = []
    all_Ls = []
    all_X_mats = []
    all_y_vecs = []

    for res in results:
        if res:
            all_betas.extend(res['betas'])
            all_phi_mats.extend(res['phi_mats'])
            all_v_values.extend(res['v_values'])
            all_T_stop_values.extend(res['T_stop_values'])
            all_fdp_values.extend(res['fdp_values'])
            all_experiment_ids.extend(res['experiment_ids'])
            all_Ls.extend(res['L_values'])
            if test:
                all_X_mats.extend(res['X_mats'])
                all_y_vecs.extend(res['y_vecs'])

    if save_as_json:
        # Save as JSON if requested
        for i, exp_id in enumerate(all_experiment_ids):
            experiment_data = {
                "beta": all_betas[i].tolist(),
                "Phi_mat": all_phi_mats[i].tolist(),
                "v": float(all_v_values[i]),
                "T_stop": int(all_T_stop_values[i]),
                "FDP": float(all_fdp_values[i]),
                "system_id": int(all_experiment_ids[i] // multilevel),
                "experiment_id": int(exp_id),
                "L": int(all_Ls[i])
            }
            if test:
                experiment_data["X_mat"] = all_X_mats[i].tolist()
                experiment_data["y_vec"] = all_y_vecs[i].tolist()
            
            json_path = os.path.join(json_folder, f"experiment_{exp_id}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
        return json_folder
    else:
        # Save all data to a single npz file
        output_file = os.path.join(data_dir, f"{specific_folder}.npz")
        save_dict = {
            "betas": np.array(all_betas),
            "phi_mats": np.array(all_phi_mats),
            "v_values": np.array(all_v_values),
            "T_stop_values": np.array(all_T_stop_values),
            "fdp_values": np.array(all_fdp_values),
            "L_values": np.array(all_Ls)
        }
        if test:
            save_dict['X_mats'] = np.array(all_X_mats)
            save_dict['y_vecs'] = np.array(all_y_vecs)
        np.savez(
            output_file,
            **save_dict
        )
        return output_file


def _process_system_t_student(args):
    # Unpack arguments
    system_id = args['system_id']
    n = args['n']
    p = args['p']
    K = args['K']
    T_stop_max = args['T_stop_max']
    num_act = args['num_act']
    num_dummies = args['num_dummies']
    num_dummies_factor = args['num_dummies_factor']
    SNR = args['SNR']
    df_range = args['df_range']
    seed_offset = args['seed_offset']
    verbose = args['verbose']
    multilevel = args.get('multilevel', 1)
    all_pairs = args.get('all_pairs')
    trex_parallel_process = args.get('trex_parallel_process', False)
    test = args.get('test', False)
    
    # Set seed for reproducibility
    np.random.seed(seed_offset + system_id)
    
    # Create beta vector with random active indices
    beta = np.zeros(p)
    active_indices = np.random.choice(p, num_act, replace=False)
    beta[active_indices] = 1
    
    # Sample degrees of freedom
    df = np.random.randint(df_range[0], df_range[1] + 1)
    if df <= 2:
        raise ValueError("Degrees of freedom must be greater than 2.")

    # Generate X from t-student distribution and y with Gaussian noise
    X = np.random.standard_t(df, size=(n, p)) * np.sqrt((df - 2) / df)
    sd = np.sqrt(np.var(X @ beta) / SNR)
    y = X @ beta + np.random.normal(size=n, scale=sd)
    
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
    res_exp = random_experiments(X, y, K=K, T_stop=T_stop_max, num_dummies=int(L), 
                              verbose=False, parallel_process=trex_parallel_process)
    if L < 1:
        raise ValueError(f"Calculated L is invalid: {L}. num_dummies={num_dummies}, num_dummies_factor={num_dummies_factor}")
    if verbose:
        print(f"System {system_id}: Time taken for random_experiments: {time.time() - start_time:.2f} seconds")
    if verbose:
        print(f"System {system_id}: L = {L}")
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
        FDP = np.sum((1 - beta) * (Phi_mat[:, T-1] > v), axis=0) / np.maximum(1, np.sum(Phi_mat[:, T-1] > v, axis=0))
        
        # Create unique experiment ID
        global_exp_id = (system_id * multilevel) + i
        
        # Append results for this v and T
        system_results['betas'].append(beta.copy())
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


def generate_tstudent_FDP_labeling(data_dir="data", specific_folder=None, T_stop_max=1, 
                                   N_systems=100, n=75, p=150, K=100, num_act=3, 
                                   num_dummies=150, num_dummies_factor=1, SNR=1.0, df_range=(5, 10), show_progress=True, 
                                   save_as_json=False, node_level_multiprocessing=False, trex_parallel_process=False,
                                   multilevel=1, verbose=False, test=False):
    """
    Generates beta, Phi, v, T_stop, FDP entries dataset, following tstudent dist.
    (objective, train FDP predicting net and then do minimization on it) saves in a single .npz file or as individual JSON files.
    
    Parameters
    ----------
    data_dir : str, default="data"
        Directory to save data
    specific_folder : str, optional
        Specific folder name under data_dir. If None, will be generated from parameters
    T_stop_max : int, default=1
        Maximum T_stop used
    N_systems : int, default=100
        Number of different systems that create the dataset
    n : int, default=75
        Number of observations
    p : int, default=150
        Number of variables
    K : int, default=100
        Number of random experiments
    num_act : int, default=3
        Number of active variables
    num_dummies : int, default=150
        Number of dummies
    SNR : float, default=1.0
        SNR
    df_range : tuple, default=(5, 10)
        Range of degrees of freedom for the t-student distribution.
    show_progress : bool, default=True
        Whether to show the progress bar
    save_as_json : bool, default=False
        If True, save each experiment as a separate JSON file in a folder
    use_multiprocessing : bool, default=False
        If True, use parallel processing in random_experiments
    multilevel : int, default=1
        Number of v, T pairs to sample per system. If 1, samples one random pair.
        If > 1, samples that many pairs without replacement.
    verbose : bool, default=False
        Enable verbose output.
        
    Returns
    -------
    str
        Path to the npz file or the directory containing JSON files
    """
    if df_range[0] <= 2:
        raise ValueError("Degrees of freedom must be greater than 2.")

    if specific_folder is None:
        # Use a specific ordering for parameters in the filename
        num_dummies_str = f"num_dummies={num_dummies}" if num_dummies_factor == 1 else f"num_dummies=[{num_dummies}, {num_dummies_factor * num_dummies}]"
        param_order = [
            (f"N_systems={N_systems}"),
            (f"SNR={SNR}"),
            (f"T_stop_max={T_stop_max}"),
            (f"n={n}"),
            (f"num_act={num_act}"),
            num_dummies_str,
            (f"p={p}"),
            (f"K={K}"),
            (f"df_min={df_range[0]}"),
            (f"df_max={df_range[1]}"),
        ]
        if multilevel > 1:
            folder_prefix = f"(tstudent_multilevel_{multilevel})_"
        else:
            folder_prefix = "(tstudent)_"
        specific_folder = folder_prefix + ','.join(param_order)
    
    # Create data directory if it doesn't exist
    create_subdir(data_dir)
    
    # Create a folder for JSON files if needed
    json_folder = None
    if save_as_json:
        json_folder = os.path.join(data_dir, specific_folder)
        create_subdir(json_folder)
    
    # Lists to store results
    betas = []
    phi_mats = []
    v_values = []
    T_stop_values = []
    fdp_values = []
    experiment_ids = []

    # Generate all (v, T) pairs once
    all_v = np.arange(0.5, 1, 1/K)
    all_v = np.append(all_v, 1 - np.finfo(float).eps)
    all_T = range(1, T_stop_max + 1)
    all_pairs = [(v, T) for v in all_v for T in all_T]

    system_args = [{
        'system_id': sid, 'n': n, 'p': p, 'K': K, 'T_stop_max': T_stop_max,
        'num_act': num_act, 'num_dummies': num_dummies, 'num_dummies_factor': num_dummies_factor, 'SNR': SNR,
        'df_range': df_range,
        'seed_offset': 0, 'verbose': verbose, # seed_offset is 0 as we are not using MPI ranks
        'multilevel': multilevel, 'all_pairs': all_pairs,
        'trex_parallel_process': trex_parallel_process,
        'test': test
    } for sid in range(N_systems)]
    
    results = []
    if N_systems > 0: # Only process if there are systems
        if node_level_multiprocessing:
            with Pool(processes=os.cpu_count()) as pool:
                map_func = pool.imap if show_progress else pool.map
                pbar = tqdm(map_func(_process_system_t_student, system_args), total=N_systems, desc="Generating data", disable=not show_progress)
                results = list(pbar)
        else:
            pbar = tqdm(system_args, desc="Generating data", disable=not show_progress)
            results = [_process_system_t_student(args) for args in pbar]

    # Aggregate results
    all_betas = []
    all_phi_mats = []
    all_v_values = []
    all_T_stop_values = []
    all_fdp_values = []
    all_experiment_ids = []
    all_Ls = []
    all_X_mats = []
    all_y_vecs = []

    for res in results:
        if res:
            all_betas.extend(res['betas'])
            all_phi_mats.extend(res['phi_mats'])
            all_v_values.extend(res['v_values'])
            all_T_stop_values.extend(res['T_stop_values'])
            all_fdp_values.extend(res['fdp_values'])
            all_experiment_ids.extend(res['experiment_ids'])
            all_Ls.extend(res['L_values'])
            if test:
                all_X_mats.extend(res['X_mats'])
                all_y_vecs.extend(res['y_vecs'])

    if save_as_json:
        # Save as JSON if requested
        for i, exp_id in enumerate(all_experiment_ids):
            experiment_data = {
                "beta": all_betas[i].tolist(),
                "Phi_mat": all_phi_mats[i].tolist(),
                "v": float(all_v_values[i]),
                "T_stop": int(all_T_stop_values[i]),
                "FDP": float(all_fdp_values[i]),
                "system_id": int(all_experiment_ids[i] // multilevel),
                "experiment_id": int(exp_id),
                "L": int(all_Ls[i])
            }
            if test:
                experiment_data["X_mat"] = all_X_mats[i].tolist()
                experiment_data["y_vec"] = all_y_vecs[i].tolist()
            
            json_path = os.path.join(json_folder, f"experiment_{exp_id}.json")
            with open(json_path, 'w') as f:
                json.dump(experiment_data, f)
        return json_folder
    else:
        # Save all data to a single npz file
        output_file = os.path.join(data_dir, f"{specific_folder}.npz")
        save_dict = {
            "betas": np.array(all_betas),
            "phi_mats": np.array(all_phi_mats),
            "v_values": np.array(all_v_values),
            "T_stop_values": np.array(all_T_stop_values),
            "fdp_values": np.array(all_fdp_values),
            "L_values": np.array(all_Ls)
        }
        if test:
            save_dict['X_mats'] = np.array(all_X_mats)
            save_dict['y_vecs'] = np.array(all_y_vecs)
        np.savez(
            output_file,
            **save_dict
        )
        return output_file
