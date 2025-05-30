import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import scanpy as sc

rep_np = np.load('/data1/data/corpus/conv1d_representations.npy')

if len(rep_np.shape) == 3:
    rep_np = rep_np.mean(axis=1)

adata = sc.read_h5ad('/data1/data/corpus/Zheng68K.h5ad')
print("Available columns in adata.obs:", adata.obs.columns)

labels = adata.obs['celltype'].values

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(rep_np)

plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)
for lbl in unique_labels:
    idxs = labels == lbl
    plt.scatter(embedding[idxs, 0], embedding[idxs, 1], label=lbl, s=10, alpha=0.8)

plt.title("UMAP of Performer Representations Colored by Cell Type")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('umap_performer_representations_by_cell_type.png', dpi=300)
plt.show()
