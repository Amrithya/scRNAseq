import scanpy as sc

file_path = "/data1/data/corpus/panglao_human10k.h5ad"
adata = sc.read_h5ad(file_path)

print(adata)
print(adata.X.shape)

print(adata.obs.head())

sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata)

sc.pl.umap(adata, color=["leiden_res0_25", "leiden_res0_5", "leiden_res1"],
           legend_loc="on data")
