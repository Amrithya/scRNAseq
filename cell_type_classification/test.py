import scanpy as sc

adata = sc.read_h5ad("Zheng68K.h5ad")

print("Shape of the data matrix:", adata.shape)
