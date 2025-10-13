import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
import pickle
import random

try:
    import h5py
except ImportError:
    h5py = None

def is_json_folder(path):
    """
    Check if the given path is a directory containing JSON files.
    """
    if not os.path.isdir(path):
        return False
    return any(f.endswith('.json') for f in os.listdir(path))

def get_phi_shape(data_path):
    """
    Determine the phi shape by loading the first Phi matrix from the data source.
    """
    if is_json_folder(data_path):
        json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON files found in {data_path}")
        with open(os.path.join(data_path, json_files[0]), 'r') as f:
            data = json.load(f)
        if 'Phi_mat' in data:
            phi_mat = np.array(data['Phi_mat'])
        else:
            raise ValueError("JSON file does not contain a Phi_mat field")
    
    elif data_path.endswith('.npz'):
        with np.load(data_path, allow_pickle=True) as data:
            if 'phi_mats' in data:
                phi_mat = data['phi_mats'][0]
            else:
                raise ValueError("The .npz file does not contain phi_mats array")

    elif data_path.endswith(('.h5', '.hdf5')):
        if h5py is None:
            raise ImportError("h5py is not installed. Please install it to read HDF5 files.")
        with h5py.File(data_path, 'r') as f:
            if 'phi_mats' in f:
                phi_mat = f['phi_mats'][0]
            else:
                raise ValueError("The HDF5 file does not contain a phi_mats dataset")
    else:
        raise ValueError(f"Unsupported data path or format: {data_path}")

    return phi_mat.shape


class FDPEstimationDataset(Dataset):
    def __init__(self, data_path, lazy_loading=False):
        """
        Initialize the dataset from a .npz, .h5, or folder of JSON files.
        
        Parameters
        ----------
        lazy_loading : bool, default=False
            If False, loads the entire dataset into RAM. If True, loads data on-demand.
        """
        self.data_path = data_path
        self.lazy_loading = lazy_loading
        self.data_file = None # For lazy-loaded HDF5 or memory-mapped NPZ
        self.has_X_y_data = False

        if is_json_folder(data_path):
            self.file_type = 'json'
            self._init_json_folder(data_path)
        elif data_path.endswith('.npz'):
            self.file_type = 'npz'
            self._init_npz(data_path)
        elif data_path.endswith(('.h5', '.hdf5')):
            self.file_type = 'hdf5'
            self._init_hdf5(data_path)
        else:
            raise ValueError(f"Unsupported data path or format: {data_path}")
        

    def _init_json_folder(self, json_folder):
        self.json_files = sorted(
            [f for f in os.listdir(json_folder) if f.endswith('.json')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        if not self.json_files:
            raise ValueError(f"No JSON files found in {json_folder}")
        
        # Check for X_mats and y_vecs in the first file
        with open(os.path.join(self.data_path, self.json_files[0]), 'r') as f:
            data = json.load(f)
            if 'X_mat' in data and 'y_vec' in data:
                self.has_X_y_data = True

        if not self.lazy_loading:
            self._load_all_data_from_json()

    def _load_all_data_from_json(self):
        phi_mats, betas, v_values, T_stop_values, fdp_values, L_values = [], [], [], [], [], []
        if self.has_X_y_data:
            X_mats, y_vecs = [], []

        for file_name in self.json_files:
            with open(os.path.join(self.data_path, file_name), 'r') as f:
                data = json.load(f)
                phi_mats.append(np.array(data['Phi_mat']))
                betas.append(np.array(data['beta']))
                v_values.append(float(data['v']))
                T_stop_values.append(float(data['T_stop']))
                fdp_values.append(float(data['FDP']))
                L_values.append(int(data['L']))
                if self.has_X_y_data:
                    x_mat_data = data.get('X_mat')
                    X_mats.append(np.array(x_mat_data) if x_mat_data is not None else None)
                    y_vec_data = data.get('y_vec')
                    y_vecs.append(np.array(y_vec_data) if y_vec_data is not None else None)

        self.phi_mats = np.empty(len(phi_mats), dtype=object)
        self.phi_mats[:] = phi_mats
        self.betas = np.empty(len(betas), dtype=object)
        self.betas[:] = betas
        self.v_values = np.array(v_values)
        self.T_stop_values = np.array(T_stop_values)
        self.fdp_values = np.array(fdp_values)
        self.L_values = np.array(L_values)
        if self.has_X_y_data:
            self.X_mats = np.empty(len(X_mats), dtype=object)
            self.X_mats[:] = X_mats
            self.y_vecs = np.empty(len(y_vecs), dtype=object)
            self.y_vecs[:] = y_vecs

    def _init_npz(self, npz_file):
        if self.lazy_loading:
            self.data_file = np.load(npz_file, mmap_mode='r')
            if 'X_mats' in self.data_file and 'y_vecs' in self.data_file:
                self.has_X_y_data = True
        else:
            # Eager loading: open file, copy all data to RAM, and close file.
            with np.load(npz_file, allow_pickle=True) as data:
                self.phi_mats = np.array(data['phi_mats'])
                self.betas = np.array(data['betas'])
                self.v_values = np.array(data['v_values'])
                self.T_stop_values = np.array(data['T_stop_values'])
                self.fdp_values = np.array(data['fdp_values'])
                self.L_values = np.array(data['L_values'])
                if 'X_mats' in data and 'y_vecs' in data:
                    self.has_X_y_data = True
                    self.X_mats = np.array(data['X_mats'])
                    self.y_vecs = np.array(data['y_vecs'])

    def _init_hdf5(self, hdf5_file):
        if h5py is None:
            raise ImportError("h5py is not installed. Please install it to read HDF5 files.")
        
        if self.lazy_loading:
            self.data_file = h5py.File(hdf5_file, 'r')
            required = ['betas', 'phi_mats', 'v_values', 'T_stop_values', 'fdp_values']
            if not all(k in self.data_file for k in required) or not ('L_values' in self.data_file):
                self.data_file.close()
                raise ValueError("HDF5 file is missing required datasets.")
            if 'X_mats' in self.data_file and 'y_vecs' in self.data_file:
                self.has_X_y_data = True
        else:
            # Eager loading: open file, copy all data to RAM, and close file.
            with h5py.File(hdf5_file, 'r') as f:
                self.phi_mats = f['phi_mats'][:]
                self.betas = f['betas'][:]
                self.v_values = f['v_values'][:]
                self.T_stop_values = f['T_stop_values'][:]
                self.fdp_values = f['fdp_values'][:]
                self.L_values = f['L_values'][:]
                if 'X_mats' in f and 'y_vecs' in f:
                    self.has_X_y_data = True
                    self.X_mats = f['X_mats'][:]
                    self.y_vecs = f['y_vecs'][:]

    def __len__(self):
        if not self.lazy_loading:
            return len(self.phi_mats)
        if self.file_type == 'json':
            return len(self.json_files)
        elif self.file_type in ['npz', 'hdf5']:
            return len(self.data_file['phi_mats'])
        return 0

    def __getitem__(self, idx):
        # Eager loading: data is already in memory.
        if not self.lazy_loading:
            phi = self.phi_mats[idx]
            beta = self.betas[idx]
            v = self.v_values[idx]
            T_stop = self.T_stop_values[idx]
            fdp = self.fdp_values[idx]
            L = self.L_values[idx]
            X_mat = self.X_mats[idx] if self.has_X_y_data else None
            y_vec = self.y_vecs[idx] if self.has_X_y_data else None
        
        # Lazy loading: fetch data from disk.
        else:
            original_idx = idx
            if self.file_type == 'json':
                with open(os.path.join(self.data_path, self.json_files[original_idx]), 'r') as f:
                    data = json.load(f)
                phi = np.array(data['Phi_mat'])
                beta = np.array(data['beta'])
                v = float(data['v'])
                T_stop = float(data['T_stop'])
                fdp = float(data['FDP'])
                L = int(data['L'])
                X_mat = np.array(data.get('X_mat')) if self.has_X_y_data else None
                y_vec = np.array(data.get('y_vec')) if self.has_X_y_data else None
            else: # NPZ (memmap) or HDF5
                phi = np.array(self.data_file['phi_mats'][original_idx])
                beta = np.array(self.data_file['betas'][original_idx])
                v = self.data_file['v_values'][original_idx]
                T_stop = self.data_file['T_stop_values'][original_idx]
                fdp = self.data_file['fdp_values'][original_idx]
                L = self.data_file['L_values'][original_idx]
                X_mat = np.array(self.data_file['X_mats'][original_idx]) if self.has_X_y_data else None
                y_vec = np.array(self.data_file['y_vecs'][original_idx]) if self.has_X_y_data else None

        item = {
            'phi': torch.tensor(phi, dtype=torch.float32),
            'beta': torch.tensor(beta, dtype=torch.float32),
            'v': torch.tensor(v, dtype=torch.float32),
            'T_stop': torch.tensor(T_stop, dtype=torch.float32),
            'fdp': torch.tensor(fdp, dtype=torch.float32),
            'L': torch.tensor(L, dtype=torch.float32)
        }
        if self.has_X_y_data and X_mat is not None and y_vec is not None:
            item['X_mat'] = torch.tensor(X_mat, dtype=torch.float32)
            item['y_vec'] = torch.tensor(y_vec, dtype=torch.float32)
            
        return item

    def close(self):
        """Close any open file handles."""
        if hasattr(self, 'data_file') and self.data_file is not None:
            if hasattr(self.data_file, 'close'):
                self.data_file.close()

    def __del__(self):
        self.close()

def collate_fn_variable_size(batch):
    """
    A collate function that handles variable size tensors.
    """
    if not batch:
        return {}
    
    # These keys will be collected into a list of tensors
    variable_size_keys = {'X_mat', 'y_vec'}
    
    # Get all keys from the first item, assuming all items have the same keys
    all_keys = batch[0].keys()
    
    standard_keys = [key for key in all_keys if key not in variable_size_keys]
    
    collated_batch = {}
    
    # Stack standard keys
    for key in standard_keys:
        collated_batch[key] = torch.stack([d[key] for d in batch])
        
    # Collect variable size keys
    for key in variable_size_keys:
        if key in all_keys:
            collated_batch[key] = [d[key] for d in batch]
            
    return collated_batch


def get_separate_data_loaders(train_data_path, test_data_path, batch_size=32, lazy_loading=False):
    train_dataset = FDPEstimationDataset(train_data_path, lazy_loading=lazy_loading)
    test_dataset = FDPEstimationDataset(test_data_path, lazy_loading=lazy_loading)
    
    # Use a custom collate function if the dataset contains X_mat and y_vec, 
    # as they can have variable shapes.
    collate_fn = collate_fn_variable_size if train_dataset.has_X_y_data or test_dataset.has_X_y_data else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_loader, test_loader


def get_partial_loader(loader, percentage, shuffle=True):
    dataset = loader.dataset
    total_size = len(dataset)
    subset_size = int(total_size * percentage)
    
    if shuffle:
        indices = random.sample(range(total_size), subset_size)
    else:
        indices = list(range(subset_size))
    
    subset_dataset = Subset(dataset, indices)
    
    partial_loader = DataLoader(
        subset_dataset,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle if hasattr(loader, 'shuffle') else False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn
    )
    
    return partial_loader


def save_dataloaders(train_loader, test_loader, path='data_loaders.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({
            'train_loader': train_loader,
            'test_loader': test_loader
        }, f)
    print(f"Data indices saved to {path}")

def load_dataloaders(path='data_loaders.pkl', data_dir=None):
    if data_dir is None:
        raise ValueError("data_dir must be provided to recreate the dataset")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['train_loader'], data['test_loader']
