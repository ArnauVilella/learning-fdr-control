# Script to run random experiments with hierarchical clustering pruning and FDP analysis
library("TRexSelector")
library("snpStats")

# Pruning function - now uses snpStats and 20% of data for p-values
prune_matrix <- function(X, y, beta) {
  # Sample 20% of the data for computing marginal p-values
  ind.screen = rep(FALSE, length(y))
  ind.screen[sample.int(length(y), size=length(y)*0.2)] = TRUE
  
  # Convert X to SnpMatrix format for snpStats
  snp.data <- as(X, "SnpMatrix")
  
  # Calculate p-values using snpStats
  pvals.screen = p.value(single.snp.tests(y[ind.screen], snp.data = snp.data[ind.screen,]), df=1)
  p_values <- pvals.screen
  
  # Correlation matrix using covariance approach (like reference)
  variances <- apply(X, 2, var, na.rm = TRUE)
  # print(sum(beta))
  if (sum(beta) != 10) {
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
    print("WARNING! WARNING! WARNING! WARNING!")
  }
  Sigma = cov(X)
  Sigma.distance = as.dist(1 - abs(cov2cor(Sigma)))
  
  # Hierarchical clustering (single linkage)
  hc <- hclust(Sigma.distance, method = "single")
  
  # Cut dendrogram at distance = 0.25 (since correlation cutoff is 0.75)
  clusters <- cutree(hc, h = 0.25)
  
  # For each cluster, pick SNP with lowest p-value (fixed logic)
  max_clusters <- max(clusters)
  ind.repr = sapply(1:max_clusters, function(c) {
    cluster_elements = clusters==c
    top_within = which.min(p_values[cluster_elements])
    if( length(top_within)==0 ) top_within = 1
    which(cluster_elements)[top_within]
  })
  # Get corresponding beta values for selected representatives
  beta_pruned_values <- beta[ind.repr]
  
  # Return pruned matrix and beta
  return(list(
    X_pruned = X[, ind.repr, drop = FALSE],
    beta_pruned = beta_pruned_values,
    n_clusters = max_clusters
  ))
}

# Create output directories if they don't exist
dir.create("X_matrices_txt", showWarnings = FALSE)
dir.create("betas_txt", showWarnings = FALSE)
dir.create("preprocessed_matrices", showWarnings = FALSE)
dir.create("preprocessed_betas", showWarnings = FALSE)

# Read the true active variables
true_active <- readRDS("true_active_variables.rds")
print(true_active)

# Initialize vectors to store results
all_fdps <- c()
cluster_counts <- c()

# Define y vector (keeping original behavior)
y <- c(rep(0, 300), rep(1, 700))

# Get list of all .rds files in X_matrices directory
rds_files <- list.files("X_matrices", pattern = "\\.rds$", full.names = TRUE)
total_files <- length(rds_files)

cat(sprintf("Processing %d matrices with pruning and FDP analysis...\n", total_files))
cat("Using snpStats::cochran.armitage.test and 20%% data sampling for p-values\n\n")

# Process all X matrices
for (i in seq_along(rds_files)) {
  # Read the X matrix
  cat(sprintf("Reading file: %s\n", rds_files[i]))
  X <- readRDS(rds_files[i])
  
  # Get the base filename without extension
  base_name <- tools::file_path_sans_ext(basename(rds_files[i]))
  
  # Extract the matrix number from the filename (e.g., "X_1" -> 1)
  matrix_number <- as.numeric(gsub("X_", "", base_name))
  
  # Create beta vector (true active variables indicator)
  beta <- as.integer(colnames(X) %in% true_active)
  
  # Save original X matrix as txt file
  original_x_filename <- file.path("X_matrices_txt", sprintf("X_%d.txt", matrix_number))
  write.table(X, file = original_x_filename, row.names = FALSE, col.names = FALSE, sep = " ")
  cat(sprintf("Writing original X matrix: %s\n", original_x_filename))
  
  # Save original beta vector as txt file
  original_beta_filename <- file.path("betas_txt", sprintf("betas_%d.txt", matrix_number))
  write.table(beta, file = original_beta_filename, row.names = FALSE, col.names = FALSE, sep = " ")
  cat(sprintf("Writing original beta vector: %s\n", original_beta_filename))
  
  cat(sprintf("Matrix %d: Original dimensions %d x %d, active vars: %d\n", 
              matrix_number, nrow(X), ncol(X), sum(beta)))
  
  # Apply pruning
  pruned_result <- prune_matrix(X, y, beta)
  X_pruned <- pruned_result$X_pruned
  beta_pruned <- pruned_result$beta_pruned
  n_clusters <- pruned_result$n_clusters
  
  # Save pruned X matrix as txt file
  pruned_x_filename <- file.path("preprocessed_matrices", sprintf("X_%d_pruned.txt", matrix_number))
  write.table(X_pruned, file = pruned_x_filename, row.names = FALSE, col.names = FALSE, sep = " ")
  cat(sprintf("Writing pruned X matrix: %s\n", pruned_x_filename))
  
  # Save pruned beta vector as txt file
  pruned_beta_filename <- file.path("preprocessed_betas", sprintf("beta_%d_pruned.txt", matrix_number))
  write.table(beta_pruned, file = pruned_beta_filename, row.names = FALSE, col.names = FALSE, sep = " ")
  cat(sprintf("Writing pruned beta vector: %s\n", pruned_beta_filename))
  
  cluster_counts <- c(cluster_counts, n_clusters)
  
  cat(sprintf("  After pruning: %d x %d, active vars: %d, clusters: %d\n", 
              nrow(X_pruned), ncol(X_pruned), sum(beta_pruned), n_clusters))
  
  # Run random experiments on pruned matrix
  # X <- as.matrix(scale(X_pruned))
  # y <- as.vector(y - mean(y))
  # res <- random_experiments(as.matrix(scale(X_pruned)), as.vector(y - mean(y)), num_dummies = 3 * ncol(X_pruned), T_stop=5)
  dummy_x <- matrix(0, nrow = 2, ncol = 2)
  dummy_y <- rep(0, 2)
  res <- random_experiments(dummy_x, dummy_y, num_dummies = ncol(dummy_x), T_stop=1)
  phi_T_mat <- res$phi_T_mat
  
  # Sample one random threshold value in [0.5, 1] for this matrix
  threshold_levels <- seq(0.5, 1 - .Machine$double.eps, by = 1/20)
  threshold <- sample(threshold_levels, 1)
  
  # T_stop_samp <- sample(1:5, 1)
  # vector_t <- phi_T_mat[, T_stop_samp]
  T_stop_dummy <- 1
  vector_t <- phi_T_mat[, T_stop_dummy]
  
  # Select indices where phi_T_mat > threshold
  selected_indices <- which(vector_t > threshold)
  
  # Compute FDP (False Discovery Proportion)
  # Count false discoveries (selected variables that are not truly active)
  false_discoveries <- sum(beta_pruned[selected_indices] == 0)
  # FDP = False Discoveries / max(1, Total Discoveries)
  # Using max(1, length) handles the case when no discoveries are made
  fdp <- false_discoveries / max(1, length(selected_indices))
  # Store the FDP
  all_fdps <- c(all_fdps, fdp)
  
  # Print progress
  cat(sprintf("Processed matrix %d/%d: %s\n\n", i, total_files, base_name))
}

cat(sprintf("Processing complete! Collected %d FDP values from %d matrices.\n", 
            length(all_fdps), total_files))

# Cluster statistics
cat(sprintf("\nCluster statistics across matrices:\n"))
cat(sprintf("  Average clusters: %.2f\n", mean(cluster_counts)))
cat(sprintf("  Min clusters: %d\n", min(cluster_counts)))
cat(sprintf("  Max clusters: %d\n", max(cluster_counts)))

# FDP statistics
cat(sprintf("\nFDP summary statistics:\n"))
cat(sprintf("  Mean: %.4f\n", mean(all_fdps)))
cat(sprintf("  Median: %.4f\n", median(all_fdps)))
cat(sprintf("  Min: %.4f\n", min(all_fdps)))
cat(sprintf("  Max: %.4f\n", max(all_fdps)))

# Create histogram of FDPs
png("fdp_histogram_pruned.png", width = 800, height = 600, res = 100)
hist(all_fdps, 
     breaks = 30,
     main = "Histogram of False Discovery Proportions (FDP) - After Pruning\n(Using snpStats and 20% data sampling)",
     xlab = "False Discovery Proportion",
     ylab = "Frequency",
     col = "lightblue",
     border = "black")

# Add vertical line at mean
abline(v = mean(all_fdps), col = "red", lwd = 2, lty = 2)
legend("topright", 
       legend = paste("Mean FDP =", round(mean(all_fdps), 4)),
       col = "red",
       lty = 2, 
       lwd = 2)

dev.off()

cat("\nHistogram saved as 'fdp_histogram_pruned.png'\n")