suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(future)
})
set.seed(42)
plan("sequential")
options(future.globals.maxSize = 24 * 1024^3)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
repo_root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir <- Sys.getenv(
  "DOPABASE_DATA_DIR",
  unset = file.path(repo_root, "human_integration", "data")
)
work_dir <- Sys.getenv(
  "DOPABASE_WORK_DIR",
  unset = file.path(repo_root, "human_integration", "work")
)
exp_dir  <- file.path(work_dir, "seurat_export")
out_dir  <- file.path(work_dir, "rpca_out")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("reading export...\n")
counts <- readMM(file.path(exp_dir, "counts.mtx"))
genes  <- readLines(file.path(exp_dir, "genes.txt"))
cells  <- readLines(file.path(exp_dir, "cells.txt"))
rownames(counts) <- genes; colnames(counts) <- cells
meta <- read.csv(file.path(exp_dir, "metadata.csv"), row.names = 1)
meta <- meta[cells, , drop = FALSE]

obj <- CreateSeuratObject(counts = counts, meta.data = meta)
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$study)

cat("normalize / HVG / scale / pca...\n")
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, nfeatures = 3000, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = 50, verbose = FALSE)

layers <- Layers(obj[["RNA"]], search = "counts")
ref_idx <- grep("Kamath", layers)
cat("reference layer index:", ref_idx, "of", paste(layers, collapse=","), "\n")

obj <- IntegrateLayers(
  object = obj, method = RPCAIntegration,
  orig.reduction = "pca", new.reduction = "integrated.rpca",
  reference = ref_idx, dims = 1:30, k.anchor = 5, verbose = FALSE
)
obj <- JoinLayers(obj)

emb <- Embeddings(obj, "integrated.rpca")[, 1:30]
write.csv(emb, file.path(out_dir, "rpca_embedding.csv"))

obj <- FindNeighbors(obj, reduction = "integrated.rpca", dims = 1:30, verbose = FALSE)
obj <- FindClusters(obj, resolution = 1.0, verbose = FALSE)
write.csv(data.frame(cell = colnames(obj), leiden_rpca = obj$seurat_clusters),
          file.path(out_dir, "rpca_clusters.csv"), row.names = FALSE)

cat("DONE -> ", out_dir, "\n")
