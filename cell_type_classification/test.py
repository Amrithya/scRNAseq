import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import shap
import os
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Reading data...")
adata = sc.read_h5ad('/data1/data/corpus/pbmc68k(2).h5ad')

print("Normalizing and log-transforming...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X = adata.X

print("Encoding labels...")
cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
y = le.fit_transform(cell_type_series)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("Fitting logistic regression...")
clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear")
clf.fit(X_train, y_train)

print("Computing SHAP values...")
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)

print("shap_values shape",shap_values.shape)