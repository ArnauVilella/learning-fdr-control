import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
from tqdm import tqdm
import plotly.graph_objects as go
plt.rcParams['text.usetex'] = True

# Define device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_combined_FDP(train_loader_FDP, train_infer_FDP, test_loader_FDP, test_infer_FDP):
    """
    Plot histograms of actual and predicted FDP values for both training and test data in one figure.
    
    Parameters
    ----------
    train_loader_FDP : list
        List of actual FDP values for training data
    train_infer_FDP : list
        List of predicted FDP values for training data
    test_loader_FDP : list
        List of actual FDP values for test data
    test_infer_FDP : list
        List of predicted FDP values for test data
    """
    bin_edges = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.001]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training data
    ax1.hist(train_loader_FDP, bins=bin_edges, edgecolor='black', alpha=0.7, color='blue')
    ax1.set_title('Labeled FDP (Training Data)')
    ax1.set_xlabel('FDP Values')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(train_infer_FDP, bins=bin_edges, edgecolor='black', alpha=0.7, color='blue')
    ax2.set_title('Estimated FDP (Training Data)')
    ax2.set_xlabel('FDP Values')
    ax2.set_ylabel('Frequency')
    
    # Test data
    ax3.hist(test_loader_FDP, bins=bin_edges, edgecolor='black', alpha=0.7, color='orange')
    ax3.set_title('Labeled FDP (Test Data)')
    ax3.set_xlabel('FDP Values')
    ax3.set_ylabel('Frequency')
    
    ax4.hist(test_infer_FDP, bins=bin_edges, edgecolor='black', alpha=0.7, color='orange')
    ax4.set_title('Estimated FDP (Test Data)')
    ax4.set_xlabel('FDP Values')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def plot_hist_FDP_overestimation(loader_FDP, infer_FDP, stage, ax=None, show=True):
    """
    Plot histogram of FDP overestimation (where predicted > actual).
    
    Parameters
    ----------
    loader_FDP : list
        List of actual FDP values
    infer_FDP : list
        List of predicted FDP values
    stage : str
        Stage identifier (e.g., 'train' or 'test')
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis. If None, create new figure
    show : bool, default=True
        Whether to call plt.show() at the end
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if ax is None, otherwise None
    """
    FDP_diff = np.array(infer_FDP) - np.array(loader_FDP)
    FDP_overestimation = FDP_diff[FDP_diff > 0]
    FDP_overestimation_rate = len(FDP_overestimation) / len(FDP_diff)

    bin_edges = [0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.125, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    
    ax.hist(FDP_overestimation, bins=bin_edges, alpha=0.7)
    ax.set_title(f"Histogram of FDP Overestimation ({stage} data)")
    ax.set_xlabel("Overestimation Amount")
    ax.set_ylabel("Frequency")

    ax.text(0.5, 0.8*ax.get_ylim()[1], f"FDP Overestimation Rate: {FDP_overestimation_rate:.2f}", 
            fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5))

    if show and fig is not None:
        plt.tight_layout()
        plt.show()
    
    return fig


def plot_hist_FDP_diff(loader_FDP, infer_FDP, stage, ax=None, show=True):
    """
    Plot histogram of the difference between predicted and actual FDP values.
    
    Parameters
    ----------
    loader_FDP : list
        List of actual FDP values
    infer_FDP : list
        List of predicted FDP values
    stage : str
        Stage identifier (e.g., 'train' or 'test')
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis. If None, create new figure
    show : bool, default=True
        Whether to call plt.show() at the end
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if ax is None, otherwise None
    """
    FDP_diff = np.array(infer_FDP) - np.array(loader_FDP)
    FDP_overestimation = FDP_diff[FDP_diff > 0]
    FDP_overestimation_rate = len(FDP_overestimation) / len(FDP_diff)

    bin_edges = [-1.001, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.15, -0.125, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, -0.0001, 
             0, 0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.125, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    
    N, bins, patches = ax.hist(FDP_diff, bins=bin_edges, alpha=0.7)
    
    # Highlight overestimations and underestimations with different colors
    neg_patches = [p for i, p in enumerate(patches) if bins[i] < 0]
    pos_patches = [p for i, p in enumerate(patches) if bins[i] >= 0]
    
    for p in neg_patches:
        p.set_facecolor('blue')
    for p in pos_patches:
        p.set_facecolor('red')
    
    ax.set_title(f"Histogram of FDP difference ({stage} data)")
    ax.set_xlabel("Difference (Predicted - Actual)")
    ax.set_ylabel("Frequency")

    ax.text(0.3, 0.8*ax.get_ylim()[1], f"FDP Overestimation Rate: {FDP_overestimation_rate:.2f}", 
            fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5))

    if show and fig is not None:
        plt.tight_layout()
        plt.show()
    
    return fig


def plot_combined_overestimation_and_diff(train_loader_FDP, train_infer_FDP, test_loader_FDP, test_infer_FDP):
    """
    Plot histograms of both FDP overestimation and differences for training and test data in one figure.
    
    Parameters
    ----------
    train_loader_FDP : list
        List of actual FDP values for training data
    train_infer_FDP : list
        List of predicted FDP values for training data
    test_loader_FDP : list
        List of actual FDP values for test data
    test_infer_FDP : list
        List of predicted FDP values for test data
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: Overestimation
    plot_hist_FDP_overestimation(train_loader_FDP, train_infer_FDP, "training", ax=ax1, show=False)
    plot_hist_FDP_overestimation(test_loader_FDP, test_infer_FDP, "test", ax=ax2, show=False)
    
    # Bottom row: FDP differences
    plot_hist_FDP_diff(train_loader_FDP, train_infer_FDP, "training", ax=ax3, show=False)
    plot_hist_FDP_diff(test_loader_FDP, test_infer_FDP, "test", ax=ax4, show=False)
    
    # Set consistent y-axis limits for comparability
    max_overest_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, max_overest_y)
    ax2.set_ylim(0, max_overest_y)
    
    max_diff_y = max(ax3.get_ylim()[1], ax4.get_ylim()[1])
    ax3.set_ylim(0, max_diff_y)
    ax4.set_ylim(0, max_diff_y)
    
    # Add row titles
    fig.text(0.5, 0.98, 'FDP Overestimation', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.49, 'FDP Difference (Predicted - Actual)', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add column titles
    fig.text(0.25, 0.02, 'Training Data', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.02, 'Test Data', ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def plot_FDR_TPR(model_metrics, trex_results, title_prefix="Train", alpha=0.1):
    """
    Plot comparison of FDR and TPR between model and trexselector.
    
    Parameters
    ----------
    model_metrics : tuple
        (FDPs, TPPs) for the model
    trex_results : dict
        Dictionary with metrics and other info for trexselector.
        Expected to contain 'metrics' and optionally 'selected_vars'.
    title_prefix : str, default="Training Data"
        Prefix for plot titles
    alpha : float, default=0.1
        Target FDR threshold
    """
    model_FDPs, model_TPPs = model_metrics
    trex_FDPs, trex_TPPs = trex_results['metrics']
    
    # Determine T-Rex title
    is_extended_calib = 'selected_vars' in trex_results
    trex_title = "T-Rex Selector (extended calibration)" if is_extended_calib else "T-Rex Selector (fixed L)"
    
    bins = np.arange(0, 1.05, 0.05)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model metrics
    ax1.hist(np.array(model_FDPs), bins=bins)
    ax1.axvline(x=alpha, color='red', linestyle='--', linewidth=2, label=f'Target FDR ({alpha})')
    ax1.axvline(x=np.mean(model_FDPs), color='black', linestyle='-', linewidth=2, label=f'FDR ({np.mean(model_FDPs):.4f})')
    ax1.set_title(f"[{title_prefix}]    Model: FDPs")
    ax1.set_xlabel('FDP Values')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    ax2.hist(np.array(model_TPPs), bins=bins)
    ax2.axvline(x=np.mean(model_TPPs), color='black', linestyle='-', linewidth=2, label=f'TPR ({np.mean(model_TPPs):.4f})')
    ax2.set_title(f"[{title_prefix}]    Model: TPPs")
    ax2.set_xlabel('TPP Values')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Trexselector metrics
    ax3.hist(np.array(trex_FDPs), bins=bins)
    ax3.axvline(x=alpha, color='red', linestyle='--', linewidth=2, label=f'Target FDR ({alpha})')
    ax3.axvline(x=np.mean(trex_FDPs), color='black', linestyle='-', linewidth=2, label=f'FDR ({np.mean(trex_FDPs):.4f})')
    ax3.set_title(f"[{title_prefix}]    {trex_title}: FDPs")
    ax3.set_xlabel('FDP Values')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    ax4.hist(np.array(trex_TPPs), bins=bins)
    ax4.axvline(x=np.mean(trex_TPPs), color='black', linestyle='-', linewidth=2, label=f'TPR ({np.mean(trex_TPPs):.4f})')
    ax4.set_title(f"[{title_prefix}]    {trex_title}: TPPs")
    ax4.set_xlabel('TPP Values')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, test_losses):
    """
    Plot training and testing loss history.
    
    Parameters
    ----------
    train_losses : list
        List of training losses for each epoch
    test_losses : list
        List of testing losses for each epoch
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='orange')
    plt.title('Training and Testing Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print final losses
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")


def plot_optimal_v_hist(results, loader_2=False, loader_1_name="Train", loader_2_name="Test"):
    """
    Plot histograms comparing optimal v values between model and trexselector.
    
    Parameters
    ----------
    results : dict
        Dictionary containing optimal v,T pairs and metrics for both model and trexselector
        Expected structure:
        {
            'loader_1': {
                'model': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)},
                'trex': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)}
            },
            'loader_2': {  # Optional, only if loader_2=True
                'model': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)},
                'trex': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)}
            }
        }
    loader_2 : bool, default=False
        Whether to plot loader 2 data
    loader_1_name : str, default="Training"
        Name for the first loader's data
    loader_2_name : str, default="Test"
        Name for the second loader's data
    """
    bins = np.linspace(0.5, 1.0, 11)  # 10 bins from 0.5 to 1.0 inclusive
    
    if loader_2:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Determine T-Rex title for loader 1
    is_extended_calib_l1 = 'selected_vars' in results['loader_1']['trex']
    trex_title_l1 = "T-Rex Selector (extended calibration)" if is_extended_calib_l1 else "T-Rex Selector (fixed L)"
    
    # Extract v values from the pairs
    loader1_model_vs = [triplet[1] for triplet in results['loader_1']['model']['optimal_L_v_T_stop_triplets']]
    loader1_trex_vs = [pair[0] for pair in results['loader_1']['trex']['optimal_v_T_stop_pairs']]
    
    if loader_2:
        # Determine T-Rex title for loader 2
        is_extended_calib_l2 = 'selected_vars' in results['loader_2']['trex']
        trex_title_l2 = "T-Rex Selector (extended calibration)" if is_extended_calib_l2 else "T-Rex Selector (fixed L)"
        loader2_model_vs = [triplet[1] for triplet in results['loader_2']['model']['optimal_L_v_T_stop_triplets']]
        loader2_trex_vs = [pair[0] for pair in results['loader_2']['trex']['optimal_v_T_stop_pairs']]
    
    # Loader 1 data - Model
    ax1.hist(loader1_model_vs, bins=bins, alpha=0.7, color='blue')
    ax1.set_title(f'[{loader_1_name}]    Model: Optimal v values')
    ax1.set_xlabel('v values')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=np.mean(loader1_model_vs), 
                color='red', linestyle='--', 
                label=f'Mean: {np.mean(loader1_model_vs):.3f}')
    ax1.legend()
    
    # Loader 1 data - Trexselector
    ax_to_use = ax2
    ax_to_use.hist(loader1_trex_vs, bins=bins, alpha=0.7, color='green')
    ax_to_use.set_title(f'[{loader_1_name}]    {trex_title_l1}: Optimal v values')
    ax_to_use.set_xlabel('v values')
    ax_to_use.set_ylabel('Frequency')
    ax_to_use.axvline(x=np.mean(loader1_trex_vs), 
                color='red', linestyle='--', 
                label=f'Mean: {np.mean(loader1_trex_vs):.3f}')
    ax_to_use.legend()
    
    if loader_2:
        # Loader 2 data - Model
        ax3.hist(loader2_model_vs, bins=bins, alpha=0.7, color='blue')
        ax3.set_title(f'[{loader_2_name}]    Model: Optimal v values')
        ax3.set_xlabel('v values')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=np.mean(loader2_model_vs), 
                    color='red', linestyle='--', 
                    label=f'Mean: {np.mean(loader2_model_vs):.3f}')
        ax3.legend()
        
        # Loader 2 data - Trexselector
        ax4.hist(loader2_trex_vs, bins=bins, alpha=0.7, color='green')
        ax4.set_title(f'[{loader_2_name}]    {trex_title_l2}: Optimal v values')
        ax4.set_xlabel('v values')
        ax4.set_ylabel('Frequency')
        ax4.axvline(x=np.mean(loader2_trex_vs), 
                    color='red', linestyle='--', 
                    label=f'Mean: {np.mean(loader2_trex_vs):.3f}')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()


def plot_optimal_T_stop_hist(results, loader_2=False, loader_1_name="Train", loader_2_name="Test", T_stop_max=10):
    """
    Plot bar charts comparing optimal T_stop values between model and trexselector.
    
    Parameters
    ----------
    results : dict
        Dictionary containing optimal v,T pairs and metrics for both model and trexselector
        Expected structure:
        {
            'loader_1': {
                'model': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)},
                'trex': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)}
            },
            'loader_2': {  # Optional, only if loader_2=True
                'model': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)},
                'trex': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)}
            }
        }
    loader_2 : bool, default=False
        Whether to plot loader 2 data
    loader_1_name : str, default="Training"
        Name for the first loader's data
    loader_2_name : str, default="Test"
        Name for the second loader's data
    T_stop_max : int, default=10
        Maximum value for T_stop
    """
    if loader_2:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Determine T-Rex title for loader 1
    is_extended_calib_l1 = 'selected_vars' in results['loader_1']['trex']
    trex_title_l1 = "T-Rex Selector (extended calibration)" if is_extended_calib_l1 else "T-Rex Selector (fixed L)"
    
    # Extract T_stop values from the pairs
    loader1_model_Ts = [triplet[2] for triplet in results['loader_1']['model']['optimal_L_v_T_stop_triplets']]
    loader1_trex_Ts = [pair[1] for pair in results['loader_1']['trex']['optimal_v_T_stop_pairs']]
    
    if loader_2:
        # Determine T-Rex title for loader 2
        is_extended_calib_l2 = 'selected_vars' in results['loader_2']['trex']
        trex_title_l2 = "T-Rex Selector (extended calibration)" if is_extended_calib_l2 else "T-Rex Selector (fixed L)"
        loader2_model_Ts = [triplet[2] for triplet in results['loader_2']['model']['optimal_L_v_T_stop_triplets']]
        loader2_trex_Ts = [pair[1] for pair in results['loader_2']['trex']['optimal_v_T_stop_pairs']]
    
    # Function to compute frequencies
    def get_frequencies(T_values):
        frequencies = np.zeros(T_stop_max)
        unique, counts = np.unique(T_values, return_counts=True)
        for T, count in zip(unique, counts):
            if T > 0 and T <= T_stop_max:
                frequencies[T-1] = count
        return frequencies
    
    # Loader 1 data - Model
    frequencies = get_frequencies(loader1_model_Ts)
    ax1.bar(range(1, T_stop_max + 1), frequencies, alpha=0.7, color='blue')
    ax1.set_title(f'[{loader_1_name}]    Model: Optimal T_stop values')
    ax1.set_xlabel('T_stop values')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=np.mean(loader1_model_Ts), 
                color='red', linestyle='--', 
                label=f'Mean: {np.mean(loader1_model_Ts):.1f}')
    ax1.legend()
    
    # Loader 1 data - Trexselector
    frequencies = get_frequencies(loader1_trex_Ts)
    ax_to_use = ax2
    ax_to_use.bar(range(1, T_stop_max + 1), frequencies, alpha=0.7, color='green')
    ax_to_use.set_title(f'[{loader_1_name}]    {trex_title_l1}: Optimal T_stop values')
    ax_to_use.set_xlabel('T_stop values')
    ax_to_use.set_ylabel('Frequency')
    ax_to_use.axvline(x=np.mean(loader1_trex_Ts), 
                color='red', linestyle='--', 
                label=f'Mean: {np.mean(loader1_trex_Ts):.1f}')
    ax_to_use.legend()
    
    if loader_2:
        # Loader 2 data - Model
        frequencies = get_frequencies(loader2_model_Ts)
        ax3.bar(range(1, T_stop_max + 1), frequencies, alpha=0.7, color='blue')
        ax3.set_title(f'[{loader_2_name}]    Model: Optimal T_stop values')
        ax3.set_xlabel('T_stop values')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=np.mean(loader2_model_Ts), 
                    color='red', linestyle='--', 
                    label=f'Mean: {np.mean(loader2_model_Ts):.1f}')
        ax3.legend()
        
        # Loader 2 data - Trexselector
        frequencies = get_frequencies(loader2_trex_Ts)
        ax4.bar(range(1, T_stop_max + 1), frequencies, alpha=0.7, color='green')
        ax4.set_title(f'[{loader_2_name}]    {trex_title_l2}: Optimal T_stop values')
        ax4.set_xlabel('T_stop values')
        ax4.set_ylabel('Frequency')
        ax4.axvline(x=np.mean(loader2_trex_Ts), 
                    color='red', linestyle='--', 
                    label=f'Mean: {np.mean(loader2_trex_Ts):.1f}')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()


def plot_optimal_v_T_stop_heatmap(results, loader_2=False, loader_1_name="Train", loader_2_name="Test", T_stop_max=5, v_bins=20, L=150):
    """
    Plot heatmaps showing the joint distribution of optimal v and T_stop values for a specific L value for the model.
    
    Parameters
    ----------
    results : dict
        Dictionary containing optimal L,v,T triplets for the model and v,T pairs for trexselector.
        Expected structure:
        {
            'loader_1': {
                'model': {'optimal_L_v_T_stop_triplets': [...], 'metrics': (...)},
                'trex': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)}
            },
            'loader_2': {  # Optional, only if loader_2=True
                'model': {'optimal_L_v_T_stop_triplets': [...], 'metrics': (...)},
                'trex': {'optimal_v_T_stop_pairs': [...], 'metrics': (...)}
            }
        }
    loader_2 : bool, default=False
        Whether to plot loader 2 data
    loader_1_name : str, default="Training"
        Name for the first loader's data
    loader_2_name : str, default="Test"
        Name for the second loader's data
    T_stop_max : int, default=5
        Maximum value for T_stop for plotting range. Note: the T_stop_max from grid_search in metrics.py is the one that defines the data range.
    v_bins : int, default=20
        Number of bins for v values
    L : int, default=150
        The L value to plot the model's heatmap for.
    """
    # Extract data
    loader1_model_triplets = np.array(results['loader_1']['model']['optimal_L_v_T_stop_triplets'])
    loader1_trex_pairs = np.array(results['loader_1']['trex']['optimal_v_T_stop_pairs'])
    
    # Create v bins
    v_edges = np.linspace(0.5, 1.0, v_bins + 1)
    
    # Function to compute 2D histogram
    def compute_2d_hist(pairs):
        if pairs.size == 0:
            return np.zeros((T_stop_max, v_bins))
        hist, _, _ = np.histogram2d(
            pairs[:, 0],  # v values
            pairs[:, 1],  # T_stop values
            bins=[v_edges, np.arange(0.5, T_stop_max + 1.5)]
        )
        return hist.T

    num_subplots = 4 if loader_2 else 2
    fig, axes = plt.subplots(num_subplots // 2, 2, figsize=(15, 5 * (num_subplots // 2)), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f'v vs T_stop Distribution for L={L}', fontsize=16)

    # --- LOADER 1 ---
    is_extended_calib_l1 = 'selected_vars' in results['loader_1']['trex']
    trex_title_l1 = "T-Rex Selector (extended calibration)" if is_extended_calib_l1 else "T-Rex Selector (fixed L)"

    # Model
    model_l1_for_L = loader1_model_triplets[loader1_model_triplets[:, 0] == L]
    model_pairs_l1 = model_l1_for_L[:, 1:]
    
    hist = compute_2d_hist(model_pairs_l1)
    im1 = axes[0].imshow(hist, aspect='auto', origin='lower', extent=[0.5, 1.0, 0.5, T_stop_max + 0.5])
    axes[0].set_title(f'[{loader_1_name}] Model')
    axes[0].set_xlabel('v values')
    axes[0].set_ylabel('T_stop values')
    axes[0].set_yticks(range(1, T_stop_max + 1))
    plt.colorbar(im1, ax=axes[0], label='Frequency')
    
    if model_pairs_l1.size > 0:
        mean_v, mean_T = np.mean(model_pairs_l1, axis=0)
        axes[0].plot(mean_v, mean_T, 'r*', markersize=10, label=f'Mean (v={mean_v:.2f}, T={mean_T:.1f})')
        axes[0].legend()

    # Trexselector
    hist = compute_2d_hist(loader1_trex_pairs)
    im2 = axes[1].imshow(hist, aspect='auto', origin='lower', extent=[0.5, 1.0, 0.5, T_stop_max + 0.5])
    axes[1].set_title(f'[{loader_1_name}] {trex_title_l1}')
    axes[1].set_xlabel('v values')
    axes[1].set_ylabel('T_stop values')
    axes[1].set_yticks(range(1, T_stop_max + 1))
    plt.colorbar(im2, ax=axes[1], label='Frequency')
    
    if loader1_trex_pairs.size > 0:
        mean_v, mean_T = np.mean(loader1_trex_pairs, axis=0)
        axes[1].plot(mean_v, mean_T, 'r*', markersize=10, label=f'Mean (v={mean_v:.2f}, T={mean_T:.1f})')
        axes[1].legend()

    # --- LOADER 2 ---
    if loader_2:
        loader2_model_triplets = np.array(results['loader_2']['model']['optimal_L_v_T_stop_triplets'])
        loader2_trex_pairs = np.array(results['loader_2']['trex']['optimal_v_T_stop_pairs'])
        
        is_extended_calib_l2 = 'selected_vars' in results['loader_2']['trex']
        trex_title_l2 = "T-Rex Selector (extended calibration)" if is_extended_calib_l2 else "T-Rex Selector (fixed L)"

        # Model
        model_l2_for_L = loader2_model_triplets[loader2_model_triplets[:, 0] == L]
        model_pairs_l2 = model_l2_for_L[:, 1:]

        hist = compute_2d_hist(model_pairs_l2)
        im3 = axes[2].imshow(hist, aspect='auto', origin='lower', extent=[0.5, 1.0, 0.5, T_stop_max + 0.5])
        axes[2].set_title(f'[{loader_2_name}] Model')
        axes[2].set_xlabel('v values')
        axes[2].set_ylabel('T_stop values')
        axes[2].set_yticks(range(1, T_stop_max + 1))
        plt.colorbar(im3, ax=axes[2], label='Frequency')
        
        if model_pairs_l2.size > 0:
            mean_v, mean_T = np.mean(model_pairs_l2, axis=0)
            axes[2].plot(mean_v, mean_T, 'r*', markersize=10, label=f'Mean (v={mean_v:.2f}, T={mean_T:.1f})')
            axes[2].legend()
        
        # Trexselector
        hist = compute_2d_hist(loader2_trex_pairs)
        im4 = axes[3].imshow(hist, aspect='auto', origin='lower', extent=[0.5, 1.0, 0.5, T_stop_max + 0.5])
        axes[3].set_title(f'[{loader_2_name}] {trex_title_l2}')
        axes[3].set_xlabel('v values')
        axes[3].set_ylabel('T_stop values')
        axes[3].set_yticks(range(1, T_stop_max + 1))
        plt.colorbar(im4, ax=axes[3], label='Frequency')
        
        if loader2_trex_pairs.size > 0:
            mean_v, mean_T = np.mean(loader2_trex_pairs, axis=0)
            axes[3].plot(mean_v, mean_T, 'r*', markersize=10, label=f'Mean (v={mean_v:.2f}, T={mean_T:.1f})')
            axes[3].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_FDP_heatmap(model, dataloader, alpha=0.1, T_stop_max=10, K=100, eps=1e-6, device=device, show_plot=True, L=150):
    """
    Plot a heatmap of average FDP estimates across different v and T_stop values for all Phi matrices in the dataloader.
    
    Parameters
    ----------
    model : nn.Module
        The trained model to use for FDP prediction
    dataloader : DataLoader
        DataLoader containing Phi matrices to evaluate
    alpha : float, default=0.1
        Target FDR threshold (shown as a contour line)
    T_stop_max : int, default=10
        Maximum T_stop value to evaluate
    K : int, default=100
        Number of v values to evaluate
    eps : float, default=1e-6
        Small value to avoid edge cases
    device : torch.device, default=device
        The device to use for computation
    show_plot : bool, default=True
        Whether to display the plot
    L : int, default=150
        The L value to use for the model prediction.
        
    Returns
    -------
    numpy.ndarray
        Matrix of FDP estimates
    """
    # Create v grid (numpy for plotting and parameters)
    V_np = np.arange(0.5, 1, 1/K)
    V_np = np.append(V_np, 1.0 - eps)
    T_stops_np = np.arange(1, T_stop_max + 1)
    
    # Convert grids to tensors on the specified device
    V_tensor = torch.tensor(V_np, device=device, dtype=torch.float32)
    T_stops_tensor = torch.tensor(T_stops_np, device=device, dtype=torch.float32)

    # Create meshgrid for T_stops and V_values to get all combinations
    T_mesh, V_mesh = torch.meshgrid(T_stops_tensor, V_tensor, indexing='ij')
    # T_mesh shape: (len(T_stops_np), K), V_mesh shape: (len(T_stops_np), K)
    
    T_flat = T_mesh.reshape(-1) # Shape: (len(T_stops_np) * K,)
    V_flat = V_mesh.reshape(-1) # Shape: (len(T_stops_np) * K,)
    num_combinations = len(T_flat)

    L_tensor = torch.tensor([L], device=device, dtype=torch.float32).expand(num_combinations)

    # Initialize FDP estimate sum matrix on GPU
    FDP_estimates_sum_gpu = torch.zeros((len(T_stops_np), len(V_np)), device=device, dtype=torch.float32)
    count = 0
    
    # Compute FDP estimates for each combination
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing FDP estimates:"):
            Phis_batch = batch['phi'].to(device)  # First element is Phi matrices
            
            for Phi_single in Phis_batch: # Phi_single has shape (rows, cols)
                # Expand Phi_single to match the number of (T, V) combinations
                # Phi_single shape: (p, d), model expects (batch_size, p, d)
                Phi_expanded = Phi_single.unsqueeze(0).expand(num_combinations, -1, -1) 
                # Phi_expanded shape: (num_combinations, p, d)
                
                # Model call processes all (T,V) combinations for this Phi_single
                # model expects v and T_stop to be (batch_size,)
                FDP_ests_for_Phi = model(Phi_expanded, V_flat, T_flat, L_tensor) # Output: (num_combinations, 1)
                
                # Reshape and add to sum
                FDP_ests_reshaped_gpu = FDP_ests_for_Phi.squeeze().reshape(len(T_stops_np), len(V_np))
                FDP_estimates_sum_gpu += FDP_ests_reshaped_gpu
                count += 1
    
    # Compute average FDP estimates
    if count > 0:
        FDP_estimates_avg_gpu = FDP_estimates_sum_gpu / count
    else:
        # Handle case with no data, though dataloader should not be empty
        FDP_estimates_avg_gpu = FDP_estimates_sum_gpu 
    
    FDP_estimates = FDP_estimates_avg_gpu.cpu().numpy()
    
    if show_plot:
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        def plot_boundary(ax, FDP_values, V_plot, T_stops_plot, alpha_plot, color='red'): # Renamed args to avoid clash
            """Plot the boundary between regions where FDP <= alpha and FDP > alpha"""
            below_alpha = FDP_values <= alpha_plot
            below_alpha = np.hstack((below_alpha, below_alpha[:, -1].reshape(-1, 1)))  # aux
            below_alpha = np.vstack((below_alpha, below_alpha[-1, :].reshape(1, -1)))  # aux

            delta_T = 1
            delta_v_plot = 1 / K

            for i_plot in range(len(T_stops_plot)):
                for j_plot in range(len(V_plot)):
                    if below_alpha[i_plot,j_plot] != below_alpha[i_plot+1,j_plot]:
                        ax.plot([V_plot[j_plot], V_plot[j_plot]+delta_v_plot], [T_stops_plot[i_plot]+delta_T/2, T_stops_plot[i_plot]+delta_T/2], color=color, linewidth=2)
                    if below_alpha[i_plot,j_plot] != below_alpha[i_plot,j_plot+1]:
                        ax.plot([V_plot[j_plot]+delta_v_plot, V_plot[j_plot]+delta_v_plot], [T_stops_plot[i_plot]-delta_T/2, T_stops_plot[i_plot]+delta_T/2], color=color, linewidth=2)
        
        im = plt.imshow(FDP_estimates, aspect='auto', origin='lower',
                       extent=[V_np.min(), V_np.max(), T_stops_np.min() - 0.5, T_stops_np.max() + 0.5],
                       cmap='viridis')
        
        plot_boundary(plt.gca(), FDP_estimates, V_np, T_stops_np, alpha, color='red')
        
        legend_elements = [Line2D([0], [0], color='red', lw=2, label=r'$\	ext{\widehat{FDP}}$ Boundary')] # Corrected LaTeX for FDP
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.colorbar(im, label=r'Average $\	ext{\widehat{FDP}}$') # Corrected LaTeX for FDP
        
        plt.title(r'Average $\	ext{\widehat{FDP}}$ Across $v$ and $T$ Values') # Corrected LaTeX for FDP
        plt.xlabel('$v$')
        plt.ylabel('$T$')
        plt.yticks(T_stops_np) 
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return FDP_estimates


def plot_real_FDR_heatmap(dataloader, alpha=0.1, T_stop_max=10, K=100, eps=1e-6, show_plot=True):
    """
    Plot a heatmap of average real FDR values across different v and T_stop values for all Phi matrices in the dataloader.
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader containing Phi matrices and beta vectors to evaluate
    alpha : float, default=0.1
        Target FDR threshold (shown as a contour line)
    T_stop_max : int, default=10
        Maximum T_stop value to evaluate
    K : int, default=100
        Number of v values to evaluate
    eps : float, default=1e-6
        Small value to avoid edge cases
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    numpy.ndarray
        Matrix of real FDP values
    """
    V_np = np.arange(0.5, 1, 1/K)
    V_np = np.append(V_np, 1.0 - eps)
    T_stops_np = np.arange(1, T_stop_max + 1) # These are 1-indexed T values

    real_FDR_sum = np.zeros((len(T_stops_np), len(V_np)))
    count = 0

    for batch_data in tqdm(dataloader, desc="Computing real FDR values"):
        Phis_batch = batch_data['phi']
        betas_batch = batch_data['beta']

        for Phi_single, beta_single in zip(Phis_batch, betas_batch):
            if isinstance(Phi_single, torch.Tensor):
                phi_np = Phi_single.cpu().numpy()
            else:
                phi_np = np.asarray(Phi_single) 

            if isinstance(beta_single, torch.Tensor):
                beta_np = beta_single.cpu().numpy()
            else:
                beta_np = np.asarray(beta_single)

            if beta_np.ndim > 1:
                beta_np = beta_np.squeeze()
            if beta_np.ndim == 0: # Handle scalar beta if p=1, otherwise it's an issue
                if phi_np.shape[0] == 1:
                    beta_np = beta_np.reshape(1)
                else: # This case should ideally not happen or be an error
                    print(f"Warning: Scalar beta encountered for multi-feature Phi (p={phi_np.shape[0]}). Skipping this sample.")
                    continue
            if len(beta_np) != phi_np.shape[0]:
                print(f"Warning: Mismatch in beta length ({len(beta_np)}) and Phi features ({phi_np.shape[0]}). Skipping this sample.")
                continue

            fdr_for_current_phi = np.zeros((len(T_stops_np), len(V_np)))

            for i, t_val in enumerate(T_stops_np): # t_val is 1-indexed T
                if t_val > phi_np.shape[1]:
                    # This Phi is shorter than t_val, contributes 0 FDR for this T and all v.
                    # fdr_for_current_phi[i, :] remains zeros.
                    continue

                phi_at_t = phi_np[:, t_val - 1] # Shape (p,)

                # Vectorized calculation over V_np (the K different v values)
                selected_matrix = phi_at_t[:, np.newaxis] > V_np # Shape (p, K)
                
                # Ensure beta_np[:, np.newaxis] is (p,1) for broadcasting with (p,K)
                false_discoveries_vector = np.sum((1 - beta_np[:, np.newaxis]) * selected_matrix, axis=0) # Shape (K,)
                total_selected_vector = np.sum(selected_matrix, axis=0) # Shape (K,)

                fdr_values_for_t = false_discoveries_vector / np.maximum(1, total_selected_vector) # Shape (K,)
                fdr_for_current_phi[i, :] = fdr_values_for_t
            
            real_FDR_sum += fdr_for_current_phi
            count += 1

    if count > 0:
        real_FDR_avg = real_FDR_sum / count
    else:
        real_FDR_avg = real_FDR_sum # Remains zeros if dataloader was empty

    if show_plot:
        plt.figure(figsize=(12, 8))
        
        def plot_boundary(ax, FDP_values, V_plot, T_stops_plot_param, alpha_plot, color='red'):
            """Plot the boundary between regions where FDP <= alpha and FDP > alpha"""
            below_alpha = FDP_values <= alpha_plot
            below_alpha = np.hstack((below_alpha, below_alpha[:, -1].reshape(-1, 1)))
            below_alpha = np.vstack((below_alpha, below_alpha[-1, :].reshape(1, -1)))

            delta_T = 1
            delta_v_plot = 1/K 
            if len(V_plot) > 1:
                delta_v_plot = V_plot[1] - V_plot[0]
            elif len(V_plot) == 1:
                 delta_v_plot = 1/K # Fallback if only one V point for some reason

            for i_plot in range(len(T_stops_plot_param)):
                for j_plot in range(len(V_plot)):
                    if below_alpha[i_plot,j_plot] != below_alpha[i_plot+1,j_plot]:
                        ax.plot([V_plot[j_plot] - delta_v_plot/2, V_plot[j_plot] + delta_v_plot/2], [T_stops_plot_param[i_plot]+delta_T/2, T_stops_plot_param[i_plot]+delta_T/2], color=color, linewidth=2)
                    if below_alpha[i_plot,j_plot] != below_alpha[i_plot,j_plot+1]:
                        ax.plot([V_plot[j_plot]+delta_v_plot/2, V_plot[j_plot]+delta_v_plot/2], [T_stops_plot_param[i_plot]-delta_T/2, T_stops_plot_param[i_plot]+delta_T/2], color=color, linewidth=2)

        im = plt.imshow(real_FDR_avg, aspect='auto', origin='lower',
                       extent=[V_np.min() - (1/K)/2, V_np.max() + (1/K)/2, T_stops_np.min() - 0.5, T_stops_np.max() + 0.5],
                       cmap='viridis')
        
        plot_boundary(plt.gca(), real_FDR_avg, V_np, T_stops_np, alpha, color='red')
        
        legend_elements = [Line2D([0], [0], color='red', lw=2, label=r'$\mathrm{FDP}$ Boundary')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.colorbar(im, label=r'Average $\mathrm{FDP}$')
        plt.title(r'Average $\mathrm{FDP}$ Across $v$ and $T$ Values')
        plt.xlabel(r'$v$')
        plt.ylabel(r'$T$')
        plt.xticks(np.linspace(V_np.min(), V_np.max(), num=5 if len(V_np)>=5 else len(V_np) ))
        plt.yticks(T_stops_np)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return real_FDR_avg


def plot_overlay_and_difference_FDP_heatmaps_3D(model, dataloader, alpha=0.1, T_stop_max=10, K=100, eps=1e-6, device=device, L=150):
    """
    Plot overlayed heatmaps/surface plots of model-predicted and real FDP values, and their difference.
    This version increases font sizes for better readability in the saved SVG file.
    """
    # Save original matplotlib settings
    original_params = plt.rcParams.copy()
    try:
        # CHANGE: Update rcParams for larger fonts on the SVG plot
        plt.rcParams.update({
            'font.size': 16,          # General font size
            'axes.labelsize': 18,     # Fontsize of the x, y, and z labels
            'axes.titlesize': 20,     # Fontsize of the axes title
            'xtick.labelsize': 14,    # Fontsize of the x tick labels
            'ytick.labelsize': 14,    # Fontsize of the y tick labels
            'legend.fontsize': 16,    # Fontsize of the legend
            'figure.titlesize': 22    # Fontsize of the figure title
        })

        # --- Data Preparation ---
        V_coords = np.arange(0.5, 1, 1/K)
        V_coords = np.append(V_coords, 1.0 - eps)
        T_coords = np.arange(1, T_stop_max + 1).astype(int)
        
        model_FDP_orig = plot_FDP_heatmap(model, dataloader, alpha, T_stop_max, K, eps, device, show_plot=False, L=L)
        real_FDP_orig = plot_real_FDR_heatmap(dataloader, alpha, T_stop_max, K, eps, show_plot=False)
        difference = np.maximum(0, real_FDP_orig - model_FDP_orig)

        model_FDP_plot = model_FDP_orig * 100
        real_FDP_plot = real_FDP_orig * 100
        
        fig = plt.figure(figsize=(22, 8))
        V_mesh, T_mesh = np.meshgrid(V_coords, T_coords)
        z_max_plot = max(np.max(model_FDP_plot), np.max(real_FDP_plot)) if model_FDP_plot.size > 0 and real_FDP_plot.size > 0 else 100
        if z_max_plot == 0: z_max_plot = 100

        # --- Plot 1: Overlayed 3D Surfaces ---
        ax_overlay = fig.add_subplot(1, 2, 1, projection='3d')
        colors_real_cmap = [(0.7, 0.7, 0.7), (0.0, 0.0, 0.0)]
        cmap_real = LinearSegmentedColormap.from_list("custom_gray", colors_real_cmap, N=2048)
        surf_real = ax_overlay.plot_surface(V_mesh, T_mesh, real_FDP_plot, cmap=cmap_real, vmin=0, vmax=z_max_plot,
                                            alpha=1.0, rstride=1, cstride=1, linewidth=0, antialiased=False)

        cmap_model = plt.cm.get_cmap('jet')
        surf_model = ax_overlay.plot_surface(V_mesh, T_mesh, model_FDP_plot, cmap=cmap_model, vmin=0, vmax=z_max_plot, 
                                             alpha=1.0, rstride=1, cstride=1, linewidth=0, antialiased=False)
        
        ax_overlay.set_xlabel(r'$v$')
        ax_overlay.set_ylabel(r'$T$')
        ax_overlay.set_zlabel(r'Average FDP $(\%)$')
        ax_overlay.set_zlim(0, z_max_plot) 
        ax_overlay.invert_xaxis()
        ax_overlay.set_yticks(T_coords)
        ax_overlay.view_init(elev=20, azim=-60)

        # Updated colorbar font sizes to match new rcParams
        cb_model = fig.colorbar(surf_model, ax=ax_overlay, shrink=0.6, aspect=10, pad=0.05, location='right')
        cb_model.set_label(r'$\widehat{\mathrm{FDP}} (\%)$', fontsize=18)
        
        cb_real = fig.colorbar(surf_real, ax=ax_overlay, shrink=0.6, aspect=10, pad=0.12, location='right')
        cb_real.set_label(r'FDP $(\%)$', fontsize=18)
        
        # --- Plot 2: Difference (2D Heatmap) ---
        ax_diff = fig.add_subplot(1, 2, 2)
        positive_colors = [(1, 0.9, 0.9), (1, 0.7, 0.7), (1, 0.5, 0.5), (1, 0.3, 0.3), (1, 0, 0)]
        custom_cmap_positive = LinearSegmentedColormap.from_list("custom_red_positive", positive_colors, N=100)

        im_diff = ax_diff.imshow(difference, aspect='auto', origin='lower',
                                 extent=[V_coords.min(), V_coords.max(), T_coords.min() - 0.5, T_coords.max() + 0.5],
                                 cmap=custom_cmap_positive, vmin=1e-9, vmax=difference.max() if difference.max() > 0 else 1)
        ax_diff.invert_xaxis()

        zero_mask = (difference == 0)
        if np.any(zero_mask):
            blue_color = [0, 0.3, 0.8, 0.8] 
            overlay_img = np.zeros((*difference.shape, 4))
            overlay_img[zero_mask] = blue_color
            ax_diff.imshow(overlay_img, aspect='auto', origin='lower',
                           extent=[V_coords.min(), V_coords.max(), T_coords.min() - 0.5, T_coords.max() + 0.5])
            legend_elements_ax_diff = [Patch(facecolor=blue_color[:3], alpha=blue_color[3], edgecolor='none', label='Zero Underestimation')]
            ax_diff.legend(handles=legend_elements_ax_diff, loc='upper right')
        
        ax_diff.set_title(r'Average Model Underestimation $(\mathrm{FDP} - \widehat{\mathrm{FDP}})^+$')
        ax_diff.set_xlabel(r'$v$')
        ax_diff.set_ylabel(r'$T$')
        ax_diff.set_yticks(T_coords) 
        ax_diff.grid(True, alpha=0.3)
        
        cbar_diff = fig.colorbar(im_diff, ax=ax_diff, shrink=0.8, aspect=20)
        cbar_diff.set_label('Average Underestimation Amount (0-1 scale)', size=18) # Also increase this label size
        if difference.max() > 0:
          cbar_diff.set_ticks(np.linspace(0, difference.max(), 5)) 
        else: 
          cbar_diff.set_ticks([0]) 
          if not np.any(zero_mask): 
              cbar_diff.set_ticks([0,1])

        # --- Create and save the standalone PDF of the left plot ---
        fig_left_only = plt.figure(figsize=(11, 8))
        ax_left_standalone = fig_left_only.add_subplot(111, projection='3d')
        surf_real_standalone = ax_left_standalone.plot_surface(V_mesh, T_mesh, real_FDP_plot, cmap=cmap_real, vmin=0, vmax=z_max_plot,
                                            alpha=1.0, rstride=1, cstride=1, linewidth=0, antialiased=False)
        surf_model_standalone = ax_left_standalone.plot_surface(V_mesh, T_mesh, model_FDP_plot, cmap=cmap_model, vmin=0, vmax=z_max_plot, 
                                             alpha=1.0, rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax_left_standalone.set_xlabel(r'$v$')
        ax_left_standalone.set_ylabel(r'$T$')
        ax_left_standalone.set_zlabel(r'Average FDP $(\%)$')
        ax_left_standalone.set_zlim(0, z_max_plot) 
        ax_left_standalone.invert_xaxis()
        ax_left_standalone.set_yticks(T_coords)
        ax_left_standalone.view_init(elev=20, azim=-60)

        cb_model_standalone = fig_left_only.colorbar(surf_model_standalone, ax=ax_left_standalone, shrink=0.6, aspect=10, pad=0.15)
        cb_model_standalone.set_label(r'$\widehat{\mathrm{FDP}} (\%)$', fontsize=18)
        cb_real_standalone = fig_left_only.colorbar(surf_real_standalone, ax=ax_left_standalone, shrink=0.6, aspect=10, pad=0.05)
        cb_real_standalone.set_label(r'FDP $(\%)$', fontsize=18)
        
        fig_left_only.savefig("overlayed.pdf", format="pdf", bbox_inches='tight')
        plt.close(fig_left_only)

        # --- Finalize and show plots ---
        plt.tight_layout(pad=1.5) 
        plt.savefig("fdp_overlay_difference_plot.svg", format="svg", bbox_inches='tight')
        plt.show()
        
        return model_FDP_orig, real_FDP_orig, difference

    finally:
        # Restore original matplotlib settings to not affect other plots
        plt.rcParams.update(original_params)


def plot_overlay_and_difference_FDP_heatmaps_3D(model, dataloader, alpha=0.1, T_stop_max=10, K=100, eps=1e-6, device=device, L=150):
    """
    Plot overlayed heatmaps/surface plots of model-predicted and real FDP values, and their difference.
    The first plot is a 3D overlay of two surfaces. The second is a 2D heatmap of the difference.
    The difference is computed as real_FDP - model_FDP, with non-negative values set to 0.
    """
    # Set larger font sizes for better SVG rendering
    plt.rcParams.update({
        'font.size': 16,           # Base font size
        'axes.titlesize': 18,      # Title font size
        'axes.labelsize': 20,      # Axis label font size
        'xtick.labelsize': 14,     # X-axis tick label size
        'ytick.labelsize': 14,     # Y-axis tick label size
        'legend.fontsize': 20,     # Legend font size
        'figure.titlesize': 20     # Figure title font size
    })
    
    # Create v grid
    V_coords = np.arange(0.5, 1, 1/K)
    V_coords = np.append(V_coords, 1.0 - eps)
    T_coords = np.arange(1, T_stop_max + 1).astype(int) # Ensure T_coords are integers
    
    # Get model-predicted FDP values (0-1 scale, without plotting)
    model_FDP_orig = plot_FDP_heatmap(model, dataloader, alpha, T_stop_max, K, eps, device, show_plot=False, L=L)
    
    # Get real FDP values (0-1 scale, without plotting)
    real_FDP_orig = plot_real_FDR_heatmap(dataloader, alpha, T_stop_max, K, eps, show_plot=False)
    
    # Compute difference (real - model), setting non-negative values to 0 (0-1 scale)
    difference = np.maximum(0, real_FDP_orig - model_FDP_orig)

    # Scale FDP values to percentages for 3D plotting
    model_FDP_plot = model_FDP_orig * 100
    real_FDP_plot = real_FDP_orig * 100
    
    # Create figure: 1 row, 2 columns (one 3D overlay, one 2D difference)
    fig = plt.figure(figsize=(22, 8)) # Adjusted figsize for two plots and colorbars

    # Create meshgrid for surface plots
    V_mesh, T_mesh = np.meshgrid(V_coords, T_coords)

    # Determine consistent Z-axis limits for the 3D FDP plot
    z_max_plot = max(np.max(model_FDP_plot), np.max(real_FDP_plot)) if model_FDP_plot.size > 0 and real_FDP_plot.size > 0 else 100 # Guard
    if z_max_plot == 0: z_max_plot = 100

    # --- Plot 1: Overlayed 3D Surfaces ---
    ax_overlay = fig.add_subplot(1, 2, 1, projection='3d')

    # Surface 1: Real FDP (bottom layer, 'bone_r' colormap)
    colors_real_cmap = [(0.7, 0.7, 0.7), (0.0, 0.0, 0.0)] # Renamed variable
    cmap_real = LinearSegmentedColormap.from_list("custom_gray", colors_real_cmap, N=2048)
    surf_real = ax_overlay.plot_surface(V_mesh, T_mesh, real_FDP_plot, cmap=cmap_real, vmin=0, vmax=z_max_plot,
                                        alpha=1.0, rstride=1, cstride=1, linewidth=0, antialiased=False)

    # Surface 2: Model FDP (top layer, 'jet' colormap)
    cmap_model = plt.cm.get_cmap('jet')
    surf_model = ax_overlay.plot_surface(V_mesh, T_mesh, model_FDP_plot, cmap=cmap_model, vmin=0, vmax=z_max_plot, 
                                         alpha=1.0, rstride=1, cstride=1, linewidth=0, antialiased=False)
    
    ax_overlay.set_title('Overlayed Average FDP Surfaces', fontsize=18)
    ax_overlay.set_xlabel(r'$v$', fontsize=16)
    ax_overlay.set_ylabel(r'$T$', fontsize=16)
    ax_overlay.set_zlabel(r'Average FDP $(\%)$', fontsize=16)
    ax_overlay.set_zlim(0, z_max_plot) 
    ax_overlay.invert_xaxis() # Invert v-axis
    ax_overlay.set_yticks(T_coords) # Set T-axis ticks to integers
    ax_overlay.view_init(elev=20, azim=-60) # Tilt the view
    
    # Set tick label sizes for 3D plot
    ax_overlay.tick_params(axis='x', labelsize=14)
    ax_overlay.tick_params(axis='y', labelsize=14)
    ax_overlay.tick_params(axis='z', labelsize=14)

    # Add colorbars for the 3D plot
    cb_model = fig.colorbar(surf_model, ax=ax_overlay, shrink=0.6, aspect=10, pad=0.05, location='right')
    cb_model.set_label(r'$\widehat{\mathrm{FDP}} (\%)$', fontsize=16)
    cb_model.ax.tick_params(labelsize=14)
    
    cb_real = fig.colorbar(surf_real, ax=ax_overlay, shrink=0.6, aspect=10, pad=0.12, location='right')
    cb_real.set_label(r'FDP $(\%)$', fontsize=16)
    cb_real.ax.tick_params(labelsize=14)
    
    # --- Plot 2: Difference (2D Heatmap) ---
    ax_diff = fig.add_subplot(1, 2, 2)
    
    positive_colors = [
        (1, 0.9, 0.9),  # very light red
        (1, 0.7, 0.7),  # light red
        (1, 0.5, 0.5),  # medium red
        (1, 0.3, 0.3),  # dark red
        (1, 0, 0)       # pure red
    ]
    n_bins_cmap = 100
    custom_cmap_positive = LinearSegmentedColormap.from_list("custom_red_positive", positive_colors, N=n_bins_cmap)

    im_diff = ax_diff.imshow(difference, aspect='auto', origin='lower',
                             extent=[V_coords.min(), V_coords.max(), T_coords.min() - 0.5, T_coords.max() + 0.5],
                             cmap=custom_cmap_positive, 
                             vmin=1e-9, 
                             vmax=difference.max() if difference.max() > 0 else 1)
    ax_diff.invert_xaxis() # Invert v-axis

    zero_mask = (difference == 0)
    if np.any(zero_mask):
        blue_color = [0, 0.3, 0.8, 0.8] 
        overlay_img = np.zeros((*difference.shape, 4))
        overlay_img[zero_mask] = blue_color
        
        ax_diff.imshow(overlay_img, aspect='auto', origin='lower',
                       extent=[V_coords.min(), V_coords.max(), T_coords.min() - 0.5, T_coords.max() + 0.5])
        
        legend_elements_ax_diff = [Patch(facecolor=blue_color[:3], alpha=blue_color[3],
                                   edgecolor='none',
                                   label='Zero Underestimation')]
        ax_diff.legend(handles=legend_elements_ax_diff, loc='upper right', fontsize=14)
    
    ax_diff.set_title(r'Average Model Underestimation $(\mathrm{FDP} - \widehat{\mathrm{FDP}})^+$', fontsize=18)
    ax_diff.set_xlabel(r'$v$', fontsize=16)
    ax_diff.set_ylabel(r'$T$', fontsize=16)
    ax_diff.set_yticks(T_coords) 
    ax_diff.tick_params(axis='x', labelsize=14)
    ax_diff.tick_params(axis='y', labelsize=14)
    ax_diff.grid(True, alpha=0.3)
    
    cbar_diff = fig.colorbar(im_diff, ax=ax_diff, shrink=0.8, aspect=20, label='Average Underestimation Amount (0-1 scale)')
    cbar_diff.set_label('Average Underestimation Amount (0-1 scale)', fontsize=16)
    cbar_diff.ax.tick_params(labelsize=14)
    
    if difference.max() > 0:
      cbar_diff.set_ticks(np.linspace(0, difference.max(), 5)) 
    else: 
      cbar_diff.set_ticks([0]) 
      if not np.any(zero_mask): 
          cbar_diff.set_ticks([0,1])

    plt.tight_layout(pad=1.5) 
    plt.savefig("fdp_overlay_difference_plot.svg", format="svg", bbox_inches='tight')
    plt.show()
    
    # Reset matplotlib rcParams to default values
    plt.rcParams.update(plt.rcParamsDefault)
    
    return model_FDP_orig, real_FDP_orig, difference


def plot_overlay_and_difference_FDP_heatmaps_3D_interactive(model, dataloader, alpha=0.1, T_stop_max=10, K=100, eps=1e-6, device=device, L=150):
    """
    Plot interactive overlayed 3D surface plots of model-predicted and real FDP values using Plotly.
    The plot will show two surfaces: one for real FDP and one for model-predicted FDP.
    The difference is computed as real_FDP - model_FDP.
    """
    # Create v grid
    V_coords = np.arange(0.5, 1, 1/K)
    V_coords = np.append(V_coords, 1.0 - eps)
    T_coords = np.arange(1, T_stop_max + 1).astype(int)

    # Get model-predicted FDP values (0-1 scale)
    # Assuming plot_FDP_heatmap and plot_real_FDR_heatmap are defined in the same file
    # and can return data without plotting by setting show_plot=False
    model_FDP_orig = plot_FDP_heatmap(model, dataloader, alpha, T_stop_max, K, eps, device, show_plot=False, L=L)
    
    # Get real FDP values (0-1 scale)
    real_FDP_orig = plot_real_FDR_heatmap(dataloader, alpha, T_stop_max, K, eps, show_plot=False)
    
    # Compute difference (real - model)
    difference = real_FDP_orig - model_FDP_orig # For informational return, not directly plotted here

    # Scale FDP values to percentages for 3D plotting
    model_FDP_plot = model_FDP_orig * 100
    real_FDP_plot = real_FDP_orig * 100
    
    # Create meshgrid for surface plots
    V_mesh, T_mesh = np.meshgrid(V_coords, T_coords)

    # Create Plotly figure
    fig = go.Figure()

    # Surface 1: Real FDP
    fig.add_trace(go.Surface(
        z=real_FDP_plot,
        x=V_mesh,
        y=T_mesh,
        name='Real FDP',
        colorscale='Greys', # Equivalent to 'bone_r' or a grayscale map
        showscale=True,
        opacity=1.0,
        colorbar=dict(title='FDP (%)', x=1.10, len=0.7) # Adjust colorbar position
    ))

    # Surface 2: Model FDP
    fig.add_trace(go.Surface(
        z=model_FDP_plot,
        x=V_mesh,
        y=T_mesh,
        name='Model FDP',
        colorscale='Jet', # Standard 'jet' colormap
        showscale=True,
        opacity=1.0,
        colorbar=dict(title='Predicted FDP (%)', x=1.25, len=0.7) # Adjust colorbar position
    ))

    # Determine consistent Z-axis limits
    z_max_plot = max(np.max(model_FDP_plot), np.max(real_FDP_plot)) if model_FDP_plot.size > 0 and real_FDP_plot.size > 0 else 100
    if z_max_plot == 0:
        z_max_plot = 100

    fig.update_layout(
        title='Interactive Overlayed Average FDP Surfaces',
        scene=dict(
            xaxis_title='v',
            yaxis_title='T',
            zaxis_title='Average FDP (%)',
            xaxis=dict(autorange='reversed'), # Invert v-axis
            yaxis=dict(tickvals=T_coords.tolist()), # Ensure integer T-ticks
            zaxis=dict(range=[0, z_max_plot]),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.75) # Initial camera view, adjust as needed
            )
        ),
        width=1000, # Adjust figure width
        height=800, # Adjust figure height
        margin=dict(l=50, r=150, b=50, t=50, pad=4) # Adjust margins for colorbars
    )

    fig.show()
    fig.write_html("fdp_overlay_interactive.html")
    
    return model_FDP_orig, real_FDP_orig, difference