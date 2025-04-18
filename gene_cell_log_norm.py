import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

adata = sc.read_h5ad('pbmc68k(2).h5ad')

sc.pp.normalize_total(adata, target_sum=1e4)  
sc.pp.log1p(adata)  
X = adata.X 

cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
y = le.fit_transform(cell_type_series)

clf = LogisticRegression(penalty="l1",solver="liblinear")
clf.fit(X, y)

y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Training Accuracy after log-normalization: {acc:.4f}")
