import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle as pkl
import umap.umap_ as umap
import torch
import matplotlib.pyplot as plt

representations = torch.load('performer_cls_representations.pt')

rep_np = representations.cpu().numpy()

print("Representation shape:", rep_np.shape)

if len(rep_np.shape) == 3:
    rep_np = rep_np.mean(axis=1)  # shape becomes [batch_size, dim]

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(rep_np)

plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)
plt.title("UMAP of Performer Representations")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig('umap_performer_representations.png')