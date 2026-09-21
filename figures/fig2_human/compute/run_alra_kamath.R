#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(ALRA); library(Matrix); library(hdf5r)})
set.seed(42)

root <- Sys.getenv("M2H_SOURCE_ROOT")
if (root == "") stop("M2H_SOURCE_ROOT is not set")
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
frozen <- file.path(dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))), "frozen")

h5 <- H5File$new(file.path(root, "Data", "Human", "Kamath", "Human_Kamath_Dopamine.h5ad"), mode = "r")
data_vec <- h5[["X"]][["data"]]$read()
indices <- h5[["X"]][["indices"]]$read()
indptr <- h5[["X"]][["indptr"]]$read()
var_names <- h5[["var"]][["_index"]]$read()
obs_names <- h5[["obs"]][["index"]]$read()
h5$close_all()

X <- new("dgRMatrix", j = as.integer(indices), p = as.integer(indptr), x = as.numeric(data_vec),
         Dim = c(length(obs_names), length(var_names)))
X <- as(X, "CsparseMatrix")
rownames(X) <- obs_names
colnames(X) <- var_names

lib_sizes <- rowSums(X)
X <- log1p(sweep(as.matrix(X), 1, lib_sizes / 10000, "/"))

alra_mat <- alra(X)$A_norm_rank_k_cor_sc

tables <- list(
  Kamath_alra_imputed.csv = c("SOX6", "CALB1", "ANXA1", "ALDH1A1", "GAD2"),
  Kamath_alra_tafa1_markers.csv = c("ASTN2", "PCDH17", "CNTN5", "TAFA1", "LMO3", "TPBG")
)
for (name in names(tables)) {
  genes <- tables[[name]]
  stopifnot(all(genes %in% colnames(alra_mat)))
  out <- data.frame(barcode = obs_names)
  for (g in genes) out[[paste0(g, "_alra")]] <- alra_mat[, g]
  write.csv(out, file.path(frozen, name), row.names = FALSE)
}
