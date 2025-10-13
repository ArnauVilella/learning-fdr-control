# Script to perform SNP level filtering on raw X matrices
library("snpStats")

# Create output directory if it doesn't exist
dir.create("X_matrices", showWarnings = FALSE)

# Read the true active variables (for reference in output)
true_active <- readRDS("true_active_variables.rds")
cat("Loaded true active variables:\n")
print(true_active)
cat("\n")

# Get list of all .rds files in raw_X_matrices directory
rds_files <- list.files("raw_X_matrices", pattern = "\\.rds$", full.names = TRUE)
total_files <- length(rds_files)

cat(sprintf("Found %d raw matrices to filter...\n\n", total_files))

# SNP filtering thresholds (from tutorial)
call_rate_threshold <- 0.95
maf_threshold <- 0.01
# maf_threshold <- 0.0005
hardy_threshold <- 10^-6

# Process all raw X matrices
for (i in seq_along(rds_files)) {
  # Read the raw X matrix
  cat(sprintf("Processing file %d/%d: %s\n", i, total_files, basename(rds_files[i])))
  X_raw <- readRDS(rds_files[i])
  
  # Get the base filename without extension
  base_name <- tools::file_path_sans_ext(basename(rds_files[i]))
  
  # Extract the matrix number from the filename (e.g., "X_1" -> 1)
  matrix_number <- as.numeric(gsub("X_|_no_preprocessing", "", base_name))
  
  cat(sprintf("  Original dimensions: %d x %d\n", nrow(X_raw), ncol(X_raw)))
  
  # Convert to SnpMatrix for filtering
  genotypes <- as(X_raw, "SnpMatrix")
  
  # Step 1: SNP summary statistics (MAF, call rate, etc.)
  snpsum.col <- col.summary(genotypes)
  
  # Step 2: Filter on MAF and call rate
  use <- with(snpsum.col, (!is.na(MAF) & MAF > maf_threshold) & Call.rate >= call_rate_threshold)
  use[is.na(use)] <- FALSE  # Remove NA's as well
  
  n_removed_maf_call <- ncol(genotypes) - sum(use)
  cat(sprintf("  Removed %d SNPs due to low MAF or call rate\n", n_removed_maf_call))
  
  # Subset genotypes and SNP summary data for SNPs that pass call rate and MAF criteria
  genotypes <- genotypes[, use]
  snpsum.col <- snpsum.col[use, ]
  
  # Step 3: Filter on Hardy-Weinberg equilibrium
  HWEuse <- with(snpsum.col, !is.na(z.HWE) & (abs(z.HWE) < abs(qnorm(hardy_threshold/2))))
  HWEuse[is.na(HWEuse)] <- FALSE  # Remove NA's as well
  
  n_removed_hwe <- ncol(genotypes) - sum(HWEuse)
  cat(sprintf("  Removed %d SNPs due to high HWE\n", n_removed_hwe))
  
  # Subset genotypes and SNP summary data for SNPs that pass HWE criteria
  genotypes <- genotypes[, HWEuse]
  snpsum.col <- snpsum.col[HWEuse, ]
  
  # Convert back to numeric matrix
  X_filtered <- as(genotypes, "numeric")
  
  cat(sprintf("  Final dimensions: %d x %d\n", nrow(X_filtered), ncol(X_filtered)))
  cat(sprintf("  Total SNPs removed: %d (%.1f%%)\n", 
              ncol(X_raw) - ncol(X_filtered),
              100 * (ncol(X_raw) - ncol(X_filtered)) / ncol(X_raw)))
  
  # Check how many true active variables remain
  if (!is.null(colnames(X_filtered))) {
    n_active_remaining <- sum(colnames(X_filtered) %in% true_active)
    cat(sprintf("  True active variables remaining: %d/%d\n", 
                n_active_remaining, length(true_active)))
  }
  
  # Save filtered matrix
  output_file <- file.path("X_matrices", sprintf("X_%d.rds", matrix_number))
  saveRDS(X_filtered, file = output_file)
  cat(sprintf("  Saved to: %s\n\n", output_file))
}

cat("SNP level filtering complete!\n")
cat(sprintf("Filtered matrices saved in 'X_matrices/' directory\n"))