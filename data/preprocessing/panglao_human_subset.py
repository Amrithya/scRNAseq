import scanpy as sc
import numpy as np

adata = sc.read_h5ad("/data1/data/corpus/panglao_human10k.h5ad")
adata_5k = sc.read_h5ad("/data1/data/corpus/panglao_human5k.h5ad")

print("10k genes dataset shape:", adata.shape)  # (n_cells, 10000)
print("5k genes dataset shape:", adata_5k.shape)
