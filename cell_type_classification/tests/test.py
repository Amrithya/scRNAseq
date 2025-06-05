import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from scipy.sparse import issparse
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score 
import scanpy as sc
import scipy.sparse
from sklearn.preprocessing import LabelEncoder
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', '..', 'data', 'Zheng68K.h5ad')
if os.path.exists(file_path):
    print("Loading balanced and preprocessed data on local")
    adata = sc.read_h5ad(file_path)
else:
    print("No file")
X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
feature_names = adata.var_names
le = LabelEncoder()
y = le.fit_transform(adata.obs['celltype'])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label encoding mapping:")
for label, encoded in label_mapping.items():
    print(f"{label}: {encoded}")
    