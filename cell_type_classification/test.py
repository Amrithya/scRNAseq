import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import shap
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

adata = sc.read_h5ad('/data1/data/corpus/pbmc68k(2).h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X = adata.X

cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
y = le.fit_transform(cell_type_series)

clf = LogisticRegression(penalty="l1",solver="liblinear")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

stacked = np.hstack(shap_values)

col_names = [
    f"{gene}_class_{cls}" for cls in range(len(shap_values)) for gene in adata.var_names
]

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
results_file = os.path.join(results_dir, "shap_values_all_classes.csv")
df = pd.DataFrame(stacked, columns=col_names)
df.to_csv(results_file, index=False)