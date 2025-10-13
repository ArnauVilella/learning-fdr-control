# Learning False Discovery Rate Control via Model-Based Neural Networks

This repository contains the code for the paper "Learning False Discovery Rate Control via Model-Based Neural Networks". For a preprint, please email me at avp@connect.ust.hk.

## Dataset Generation

### Artificial Datasets

The scripts to generate the artificial datasets are located in the `slurm_files/` directory. These are designed for use on a computing cluster (like Slurm) for large-scale experiments as described in the paper.

- Each distribution will generate a separate `.h5` file.
- After generation, these files need to be merged using the `join_h5.py` script before training.

For generating smaller datasets with a single distribution, you can use the functions available in `trexselector_deep/generate_data.py`.

### Genomics Dataset

1.  **Preprocessing:** Before generating the genomics dataset, you must first run `filtering.R` and then `pruning.R` to preprocess the raw HAPGEN2 matrices.
2.  **Generation:** The `slurm_files/` directory also contains the scripts for generating the final deep learning dataset from the preprocessed data. For smaller-scale generation, `trexselector_deep/generate_data.py` can also be used.

## Experiments

The Jupyter notebooks (`.ipynb` files) in the root directory contain the different experiments presented in the paper.
