import scanpy as sc
import numpy as np

adata = sc.read_h5ad("/data1/data/corpus/panglao_human.h5ad")
gene_counts = (adata.X > 0).sum(axis=0)

if hasattr(gene_counts, "A1"):
    gene_counts = gene_counts.A1

gene_names = adata.var_names
sorted_indices = np.argsort(-gene_counts)
top10k_genes = gene_names[sorted_indices[:10000]]
top5k_genes = gene_names[sorted_indices[:5000]]

adata_10k = adata[:, top10k_genes].copy()
adata_5k = adata[:, top5k_genes].copy()

adata_10k.write("/data1/data/corpus/panglao_human10k.h5ad")
adata_5k.write("/data1/data/corpus/panglao_human5k.h5ad")
