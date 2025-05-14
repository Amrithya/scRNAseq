import scanpy as sc

file_path = '/data1/data/corpus/panglao_human.h5ad'
adata = sc.read_h5ad(file_path)

print(adata)
print(adata.X.shape)

print(adata.X)
