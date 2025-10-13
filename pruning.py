import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import norm

# Paths
X_input_dir = "genomics_data/X_matrices"
X_output_dir = "genomics_data/preprocessed_matrices"
beta_input_dir = "genomics_data/betas"
beta_output_dir = "genomics_data/preprocessed_betas"
os.makedirs(X_output_dir, exist_ok=True)
os.makedirs(beta_output_dir, exist_ok=True)

# Response vector: 100 controls (0s), 200 cases (1s)
y = np.array([0]*100 + [1]*200)

def cochran_armitage(snp, y):
    """
    Cochran-Armitage trend test for SNP vs binary phenotype.
    SNP coded as 0/1/2.
    """
    # counts by genotype and phenotype
    geno_vals = [0,1,2]
    n_cases = np.array([np.sum((snp == g) & (y == 1)) for g in geno_vals])
    n_controls = np.array([np.sum((snp == g) & (y == 0)) for g in geno_vals])
    n_total = n_cases + n_controls
    N = np.sum(n_total)

    # scores (trend: 0,1,2)
    scores = np.array(geno_vals, dtype=float)

    # proportions
    p_cases = np.sum(n_cases) / N
    p_controls = 1 - p_cases

    # numerator: sum(scores * (observed_cases - expected_cases))
    expected_cases = n_total * p_cases
    num = np.sum(scores * (n_cases - expected_cases))

    # denominator
    var = p_cases * p_controls * (
        np.sum(n_total * scores**2) -
        (np.sum(n_total * scores)**2) / N
    )
    if var == 0:
        return 1.0
    z = num / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))  # two-sided
    return p

cluster_counts = []

for i in range(1, 101):
    print(f"Processing matrix {i}/100...")
    X_path = os.path.join(X_input_dir, f"X_{i}.txt")
    if not os.path.exists(X_path):
        print(f"Warning: {X_path} not found, skipping.")
        continue
    X = np.loadtxt(X_path, dtype=np.int16)

    # Cochran-Armitage p-values for SNPs
    p_values = np.array([cochran_armitage(X[:, j], y) for j in range(X.shape[1])])

    # Correlation matrix between SNPs
    corr = np.corrcoef(X, rowvar=False)

    # Convert correlation to distance (1 - |r|)
    dist = 1 - np.abs(corr)

    # Hierarchical clustering (single linkage)
    Z = linkage(dist[np.triu_indices_from(dist, k=1)], method="single")

    # Cut dendrogram at distance = 0.25 (since corr cutoff is 0.75)
    labels = fcluster(Z, t=0.25, criterion="distance")

    # For each cluster, pick SNP with lowest p-value
    selected_snps = []
    for cluster_id in np.unique(labels):
        members = np.where(labels == cluster_id)[0]
        best_snp = members[np.argmin(p_values[members])]
        selected_snps.append(best_snp)

    X_pruned = X[:, selected_snps]

    # Save pruned matrix
    np.savetxt(os.path.join(X_output_dir, f"X_{i}_pruned.txt"), X_pruned, fmt="%d")

    # Propagate pruning to beta vectors
    beta_path = os.path.join(beta_input_dir, f"beta_{i}.txt")
    if os.path.exists(beta_path):
        beta = np.loadtxt(beta_path, dtype=np.int16)
        print(sum(beta))
        
        # For each cluster, get the maximum beta coefficient instead of the one from the representative SNP.
        beta_pruned_values = []
        for cluster_id in np.unique(labels):
            members = np.where(labels == cluster_id)[0]
            beta_pruned_values.append(np.max(beta[members]))
        beta_pruned = np.array(beta_pruned_values)

        print(sum(beta_pruned))

        np.savetxt(os.path.join(beta_output_dir, f"beta_{i}_pruned.txt"), beta_pruned, fmt="%d")
    else:
        print(f"Warning: {beta_path} not found, skipping beta pruning for this matrix.")


    cluster_counts.append(len(np.unique(labels)))

# Report cluster statistics
print("Cluster statistics across matrices:")
print(f"Average clusters: {np.mean(cluster_counts):.2f}")
print(f"Min clusters: {np.min(cluster_counts)}")
print(f"Max clusters: {np.max(cluster_counts)}")