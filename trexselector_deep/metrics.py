import numpy as np
import torch
from tqdm import tqdm
from trexselector import Phi_prime_fun, fdp_hat, select_var_fun, trex

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def grid_search(model, Phis, alpha=0.1, T_stop_max=10, L_max_factor=10, K=100, eps=np.finfo(float).eps, device="cpu"):
    """
    Grid search (vectorized) over L, v and T_stop values to find the optimal values that maximize the number of selected variables
    (and, in case of tie, minimizing v and maximizing T) while keeping the estimated FDP under alpha.
    
    Parameters
    ----------
    model : nn.Module
        The trained model to use for FDP prediction
    Phis : torch.Tensor
        Batch of Phi matrices
    alpha : float, default=0.1
        Target FDR threshold
    T_stop_max : int, default=10
        Maxium T allowed
    K : int, default=100
        Number of threshold values to test
    eps : float, default=1e-6
        Small value to avoid edge cases
    device : torch.device, default=device
        The device to use for computation
        
    Returns
    -------
    list
        List of tuples (L, v, T_stop) containing optimal threshold values for each Phi matrix
    """
    Phis = Phis.to(device)
    
    # Handle batch dimension - check if input is batched or not
    original_unbatched = False
    if Phis.dim() == 2:  # Unbatched case: (n_vars, T_stop_max)
        Phis = Phis.unsqueeze(0)  # Add batch dimension: (1, n_vars, T_stop_max)
        original_unbatched = True
    elif Phis.dim() != 3:  # Should be (batch_size, n_vars, T_stop_max)
        raise ValueError(f"Expected Phis to have 2 or 3 dimensions, got {Phis.dim()}")
    
    batch_size = Phis.shape[0]
    p = Phis.shape[1]

    # Vectorized L-search
    with torch.no_grad():
        L_max = L_max_factor * p
        L_grid = torch.arange(p, L_max + 1, p, device=device, dtype=torch.float32)
        num_l = L_grid.numel()

        Phis_expanded_L = Phis.repeat_interleave(num_l, dim=0)
        L_expanded = L_grid.repeat(batch_size)
        
        v_l_search = torch.full((batch_size * num_l,), 0.75, device=device)
        T_l_search = torch.ones(batch_size * num_l, device=device)

        FDPs_L = model(Phis_expanded_L, v_l_search, T_l_search, L_expanded).view(batch_size, num_l)

        valid_mask_L = FDPs_L <= alpha
        
        first_valid_idx = torch.argmax(valid_mask_L.to(torch.int), dim=1)
        
        final_indices = torch.where(
            FDPs_L[torch.arange(batch_size), first_valid_idx] <= alpha,
            first_valid_idx,
            num_l - 1
        )
        optimal_Ls = L_grid[final_indices]

    # Initialize tracking variables for each sample in the batch
    best_v = torch.ones(batch_size, device=device)
    best_T_stop = torch.ones(batch_size, dtype=torch.long, device=device)
    best_num_selected = torch.zeros(batch_size, device=device)

    # Create grid of threshold values to test (ascending order: 0.5 -> 1.0)
    v_grid = torch.arange(0.5, 1, 1/K, device=device)
    v_grid = torch.cat([v_grid, torch.tensor([1 - eps], device=device)])
    num_v = v_grid.numel()

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for T_stop in range(1, T_stop_max + 1):
            if not active_mask.any():
                break

            idx = active_mask.nonzero(as_tuple=True)[0]
            cur_phis = Phis[active_mask]
            cur_ls = optimal_Ls[active_mask]

            v_strict = v_grid[-1]
            T_stop_tensor = torch.tensor([T_stop], device=device)

            fdp_strict = model(
                cur_phis,
                v_strict.expand(idx.numel()),
                T_stop_tensor.expand(idx.numel()),
                cur_ls
            ).view(-1)

            failed = fdp_strict > alpha
            active_mask[idx[failed]] = False
            
            if not active_mask.any():
                continue

            idx = active_mask.nonzero(as_tuple=True)[0]
            cur_phis = Phis[active_mask]
            cur_ls = optimal_Ls[active_mask]
            cur_batch = cur_phis.shape[0]

            Phis_expanded = cur_phis.repeat_interleave(num_v, dim=0)
            V_expanded = v_grid.repeat(cur_batch)
            T_expanded = T_stop_tensor.expand(cur_batch * num_v)
            Ls_expanded = cur_ls.repeat_interleave(num_v, dim=0)

            FDP = model(Phis_expanded, V_expanded, T_expanded, Ls_expanded).view(cur_batch, num_v)

            phi_t = cur_phis[:, :, T_stop - 1]
            num_selected = (phi_t.unsqueeze(2) > v_grid).sum(dim=1).float()

            valid_mask = (FDP <= alpha)
            counts_masked = torch.where(valid_mask, num_selected, -1.0)

            max_counts, chosen_idx = counts_masked.max(dim=1)
            chosen_v = v_grid[chosen_idx]

            should_update = max_counts >= best_num_selected[active_mask]
            to_update = idx[should_update]
            
            best_num_selected[to_update] = max_counts[should_update]
            best_T_stop[to_update] = T_stop
            best_v[to_update] = chosen_v[should_update]

    results = list(zip(optimal_Ls.cpu().tolist(), best_v.cpu().tolist(), best_T_stop.cpu().tolist()))
    
    if original_unbatched:
        return results[0]
    else:
        return results


def calibration_trex_fixed_L(Phis, Ls, alpha=0.1, K=100, eps=1e-6, T_stop_max=10, max_dummies=10):
    """
    Calibrate the trexselector method for comparison with the deep learning approach. (fixed L=p)
    
    Parameters
    ----------
    Phis : torch.Tensor
        Batch of Phi matrices
    Ls : torch.Tensor
        Batch of L values
    alpha : float, default=0.1
        Target FDR threshold
    K : int, default=100
        Number of threshold values to test
    eps : float, default=1e-6
        Small value to avoid edge cases
    T_stop_max : int, default=10
        Maximum T allowed
    max_dummies : int, default=10
        Maximum number of dummy variables
        
    Returns
    -------
    list
        List of tuples (v, T_stop) containing optimal threshold values for each Phi matrix
    """
    Phis = Phis.cpu().detach().numpy()
    Ls = Ls.cpu().detach().numpy()
    batch_optimal_pairs = []
    
    for full_phi, L in zip(Phis, Ls):
        # full_phi: matrix of relative ocurrences for all variables (rows) and for T = 1, ..., T_stop_max (columns)
        p = full_phi.shape[0]
        
        V = np.arange(0.5, 1, 1/K)
        V = np.append(V, 1.0 - eps)
        T_stop = 1

        # T_stop_max def with num_dummies
        phi_T_mat = full_phi[:, :T_stop]
        Phi = full_phi[:, T_stop - 1]
        
        Phi_prime = Phi_prime_fun(p, T_stop, int(L), phi_T_mat, Phi)
        FDP_hat = fdp_hat(V, Phi, Phi_prime)

        Phi_mat = np.expand_dims(Phi, axis=0)
        FDP_hat_mat = np.expand_dims(FDP_hat, axis=0)

        fdp_lower_tFDR = FDP_hat[-1] <= alpha

        while fdp_lower_tFDR and T_stop < T_stop_max:
            T_stop += 1

            phi_T_mat = full_phi[:, :T_stop]
            Phi = full_phi[:, T_stop - 1]

            Phi_prime = Phi_prime_fun(p, T_stop, int(L), phi_T_mat, Phi)
            FDP_hat = fdp_hat(V, Phi, Phi_prime)

            Phi_mat = np.vstack((Phi_mat, Phi))
            FDP_hat_mat = np.vstack((FDP_hat_mat, FDP_hat))

            fdp_lower_tFDR = FDP_hat[-1] <= alpha
        
        best_v = select_var_fun(p, alpha, T_stop, FDP_hat_mat, Phi_mat, V)['v_thresh']
        batch_optimal_pairs.append((best_v, T_stop))
    return batch_optimal_pairs


def calibration_trex(Xs, ys, alpha=0.1, T_stop_max=5):
    """
    Calibrate the trexselector method for comparison with the deep learning approach.
    Algorithm 2 of T-Rex Selector paper.

    Parameters
    ----------
    Xs : torch.Tensor or list of torch.Tensor
        A batch of data matrices (features), with shape (batch_size, n_samples, n_features), or a list of such tensors.
    ys : torch.Tensor or list of torch.Tensor
        A batch of target vectors, with shape (batch_size, n_samples), or a list of such tensors.
    alpha : float, default=0.1
        The target theoretical False Discovery Rate (tFDR).

    Returns
    -------
    list
        A list of tuples, where each tuple contains the optimal (v_thresh, T_stop)
        pair for a corresponding dataset in the batch.
    list
        A list of numpy.ndarray, where each array contains the selected variables
        for a corresponding dataset in the batch.
    """
    if isinstance(Xs, torch.Tensor):
        Xs = Xs.cpu().detach().numpy()
    else:  # list of tensors
        Xs = [X.cpu().detach().numpy() for X in Xs]
    
    if isinstance(ys, torch.Tensor):
        ys = ys.cpu().detach().numpy()
    else:  # list of tensors
        ys = [y.cpu().detach().numpy() for y in ys]

    batch_optimal_pairs = []
    batch_selected_vars = []

    for X, y in zip(Xs, ys):
        res = trex(X, y, tFDR=alpha, verbose=False, max_T_stop=T_stop_max)
        batch_optimal_pairs.append((res['v_thresh'], res['T_stop']))
        batch_selected_vars.append(res['selected_var'])
    
    return batch_optimal_pairs, batch_selected_vars


def get_comparison_results(model, loader_1, loader_2=None, alpha=0.1, K=100, eps=1e-6, T_stop_max=10, L_max_factor=10, num_batches_to_group=1, device=device):
    """
    Compares Deep-TRexSelector with T-Rex Selector.
    Finds optimal threshold values (L, v, T_stop) and calculates FDP/TPP metrics.
    Can aggregate multiple small batches into a larger one for efficiency.
    
    Parameters
    ----------
    model : nn.Module
        The trained model for FDP prediction.
    loader_1 : DataLoader
        DataLoader for the first dataset.
    loader_2 : DataLoader, optional
        DataLoader for the second dataset.
    alpha : float, default=0.1
        Target FDR threshold.
    K : int, default=100
        Number of threshold values to test.
    eps : float, default=1e-6
        Small value to avoid edge cases.
    T_stop_max : int, default=10
        Maximum T allowed.
    num_batches_to_group : int, default=1
        The number of batches from the loader to group together for processing.
        A value > 1 can improve performance on GPUs but will use more memory.
    device : torch.device, default=device
        The device for computation.
        
    Returns
    -------
    dict
        A dictionary containing optimal L,v,T triplets and metrics for the model, and v,T pairs for trexselector.
    """
    results = {}

    # Helper function to process a loader
    def _process_loader(loader, loader_name):
        # Buffers to hold data from multiple batches
        phis_buffer, betas_buffer, ls_buffer = [], [], []
        x_mat_buffer, y_vec_buffer = [], []  # Optional buffers
        
        # Lists to store final results for the entire loader
        all_model_triplets, all_trex_pairs = [], []
        all_trex_selected = []
        all_model_fdps, all_model_tpps = [], []
        all_trex_fdps, all_trex_tpps = [], []

        use_calibration_trex = None  # Decide based on first batch

        # Helper function to process an aggregated batch of data
        def _process_aggregated_batch(aggregated_phis, aggregated_betas, aggregated_ls, aggregated_x_mat=None, aggregated_y_vec=None):
            # Get optimal pairs for the entire aggregated batch
            batch_model_triplets = grid_search(model, aggregated_phis, K=K, eps=eps, T_stop_max=T_stop_max, L_max_factor=L_max_factor, alpha=alpha, device=device)

            
            if use_calibration_trex:
                batch_trex_pairs, batch_selected_trex = calibration_trex(aggregated_x_mat, aggregated_y_vec, alpha=alpha, T_stop_max=T_stop_max)
                all_trex_selected.extend(batch_selected_trex)
            else:
                batch_trex_pairs = calibration_trex_fixed_L(aggregated_phis, aggregated_ls, alpha=alpha, K=K, eps=eps, T_stop_max=T_stop_max)

            # Store the pairs
            all_model_triplets.extend(batch_model_triplets)
            all_trex_pairs.extend(batch_trex_pairs)

            # Calculate metrics for each sample in the aggregated batch
            for i in range(len(aggregated_phis)):
                phi, beta = aggregated_phis[i].cpu().numpy(), aggregated_betas[i].cpu().numpy()
                true_actives = np.nonzero(beta)[0]
                
                # Model metrics
                _, v_opt_model, T_opt_model = batch_model_triplets[i]
                selected_model = np.where(phi[:, T_opt_model - 1] > v_opt_model)[0]
                selected_actives_model = np.intersect1d(selected_model, true_actives, assume_unique=True)
                all_model_fdps.append(len(np.setdiff1d(selected_model, selected_actives_model, assume_unique=True)) / max(len(selected_model), 1))
                all_model_tpps.append(len(selected_actives_model) / max(len(true_actives), 1))
                
                # Trex metrics
                if use_calibration_trex:
                    selected_trex = batch_selected_trex[i]
                else:
                    v_opt_trex, T_opt_trex = batch_trex_pairs[i]
                    selected_trex = np.where(phi[:, T_opt_trex - 1] > v_opt_trex)[0]
                selected_actives_trex = np.intersect1d(selected_trex, true_actives, assume_unique=True)
                all_trex_fdps.append(len(np.setdiff1d(selected_trex, selected_actives_trex, assume_unique=True)) / max(len(selected_trex), 1))
                all_trex_tpps.append(len(selected_actives_trex) / max(len(true_actives), 1))

        # Main loop over the data loader
        for i, data in enumerate(tqdm(loader, desc=f"Processing {loader_name}")):
            if i == 0:
                use_calibration_trex = 'X_mat' in data and 'y_vec' in data

            phis_buffer.append(data['phi'])
            betas_buffer.append(data['beta'])
            ls_buffer.append(data['L'])
            if use_calibration_trex:
                x_mat_buffer.extend(data['X_mat'])
                y_vec_buffer.extend(data['y_vec'])
            
            # If we've collected enough batches, process them
            if len(phis_buffer) == num_batches_to_group:
                agg_phis = torch.cat(phis_buffer, dim=0)
                agg_betas = torch.cat(betas_buffer, dim=0)
                agg_ls = torch.cat(ls_buffer, dim=0)
                agg_x_mat = x_mat_buffer if use_calibration_trex else None
                agg_y_vec = y_vec_buffer if use_calibration_trex else None
                _process_aggregated_batch(agg_phis, agg_betas, agg_ls, agg_x_mat, agg_y_vec)
                phis_buffer.clear()
                betas_buffer.clear()
                ls_buffer.clear()
                x_mat_buffer.clear()
                y_vec_buffer.clear()
                
        # Process any remaining batches that didn't form a full group
        if phis_buffer:
            agg_phis = torch.cat(phis_buffer, dim=0)
            agg_betas = torch.cat(betas_buffer, dim=0)
            agg_ls = torch.cat(ls_buffer, dim=0)
            agg_x_mat = x_mat_buffer if use_calibration_trex else None
            agg_y_vec = y_vec_buffer if use_calibration_trex else None
            _process_aggregated_batch(agg_phis, agg_betas, agg_ls, agg_x_mat, agg_y_vec)

        # Return the collected results in a structured dictionary
        return {
            'model': {
                'optimal_L_v_T_stop_triplets': all_model_triplets,
                'metrics': (all_model_fdps, all_model_tpps)
            },
            'trex': {
                'optimal_v_T_stop_pairs': all_trex_pairs,
                'metrics': (all_trex_fdps, all_trex_tpps),
                **({'selected_vars': all_trex_selected} if use_calibration_trex else {})
            }
        }

    # Process each loader
    results['loader_1'] = _process_loader(loader_1, "loader_1")
    if loader_2:
        results['loader_2'] = _process_loader(loader_2, "loader_2")

    return results
