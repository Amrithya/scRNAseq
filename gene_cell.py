import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


adata = sc.read_h5ad('pbmc68k(2).h5ad')

X = adata.X

cell_type_df = pd.DataFrame(adata.obs['cell_type'])
cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
encoded_cell_types = le.fit_transform(cell_type_series)
cell_type_df['encoded'] = encoded_cell_types
cell_type_df.drop(columns=['cell_type'], inplace=True)
X_normalized = normalize(X, norm='l1', axis=1)

clf=LogisticRegression(penalty="l1",solver="liblinear")
clf.fit(X_normalized,cell_type_df['encoded'])

y_pred = clf.predict(X_normalized)
decoded_preds = le.inverse_transform(y_pred)

y_true = le.transform(adata.obs['cell_type']) 
acc = accuracy_score(y_true, y_pred)
print(f"Training Accuracy: {acc:.4f}")