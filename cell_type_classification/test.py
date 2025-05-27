import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import shap
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

adata = sc.read_h5ad('/data1/data/corpus/pbmc68k(2).h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X = adata.X

cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
y = le.fit_transform(cell_type_series)

clf = LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial", max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    n_classes = len(shap_values)
    n_samples, n_features = shap_values[0].shape
    stacked = np.hstack(shap_values)
else:
    if shap_values.ndim == 3:
        n_samples, n_classes, n_features = shap_values.shape
        stacked = shap_values.reshape(n_samples, n_classes * n_features)
    elif shap_values.ndim == 2:
        n_samples, n_features = shap_values.shape
        n_classes = 1
        stacked = shap_values
    else:
        raise ValueError("Unexpected SHAP output shape")

feature_names = list(adata.var_names)[:n_features]

if n_classes > 1:
    col_names = [f"{gene}_class_{cls}" for cls in range(n_classes) for gene in feature_names]
else:
    col_names = feature_names

assert stacked.shape[1] == len(col_names), f"Mismatch: {stacked.shape[1]} SHAP cols vs {len(col_names)} headers"

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

results_file = os.path.join(results_dir, "shap_values_all_classes.csv")
df = pd.DataFrame(stacked, columns=col_names)
df.to_csv(results_file, index=False)
