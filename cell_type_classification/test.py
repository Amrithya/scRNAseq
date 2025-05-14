import os
import scanpy as sc

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'pbmc68k(2).h5ad')
adata = sc.read_h5ad(file_path)

print(adata)
print(adata.X.shape)

print(adata.X)
print(adata.obs['cell_type'].value_counts())

sc.pp.neighbors(adata)
sc.tl.umap(adata)