import os
import torch
import shap
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import xgboost as xgb
import torch.nn as nn
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from scipy.sparse import issparse
import scipy.sparse

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score 
import helper as h



def shap_explain(clf, X_test, y_test, feature_names):
    """
    Function to explain model predictions using SHAP and save results.

    Returns:
    - shap_values: SHAP values for the test data
    - model: Trained model
    - explainer: SHAP explainer object
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("Explaining model predictions using SHAP")

    explainer = shap.Explainer(clf, X_test)
    y_pred = clf.predict(X_test)

    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")
    sample_idx = correct_indices[0]

    shap_values = explainer(X_test[sample_idx].reshape(1, -1))

    if hasattr(shap_values, "values") and shap_values.values.ndim == 2:
        pred_class = y_pred[sample_idx]
        sample_shap_values = shap_values.values[pred_class]
    else:
        sample_shap_values = shap_values.values[0]

    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sample_shap_values
    }).sort_values(by='shap_value', key=abs, ascending=False)

    csv_path = os.path.join(results_dir, f'shap_values_sample_{sample_idx}_class_{y_pred[sample_idx]}.csv')
    df_shap.to_csv(csv_path, index=False)
    print(f"Saved SHAP values for sample {sample_idx} to {csv_path}")

    return shap_values, explainer


adata = ad.read_h5ad('/data1/data/corpus/Zheng68K.h5ad')  
X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
feature_names = adata.var_names
le = LabelEncoder()
y = le.fit_transform(adata.obs['celltype'])
X_train, y_train, X_test, y_test = h.split_data(X, y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
lr = h.train_logistic_regression(X_train, y_train)
shap_values, explainer = shap_explain(lr, X_test, y_test, feature_names)

