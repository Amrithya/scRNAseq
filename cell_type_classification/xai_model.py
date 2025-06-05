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

    print("Explaining model predictions using SHAP for one correctly predicted sample")

    explainer = shap.Explainer(clf, X_test)
    y_pred = clf.predict(X_test)

    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")
    sample_idx = correct_indices[0]

    shap_values = explainer(X_test[sample_idx].reshape(1, -1))

    if hasattr(shap_values, "values"):
        vals = shap_values.values
        print(f"shap_values.values shape: {vals.shape}")
        pred_class = y_pred[sample_idx]
        if vals.ndim == 3:
            num_samples = vals.shape[0]
            if vals.shape[1] == len(feature_names):
                sample_shap_values = vals[sample_idx, :, pred_class]
            elif vals.shape[2] == len(feature_names):
                sample_shap_values = vals[sample_idx, pred_class, :]
            else:
                raise ValueError("Unexpected shape of SHAP values for multiclass")
        elif vals.ndim == 2:
            sample_shap_values = vals[sample_idx]
        else:
            raise ValueError(f"Unexpected SHAP values ndim: {vals.ndim}")
    else:
        raise ValueError("SHAP values object has no attribute 'values'")

    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sample_shap_values
    }).sort_values(by='shap_value', key=abs, ascending=False)

    csv_path = os.path.join(results_dir, f'shap_values_sample_{sample_idx}_class_{y_pred[sample_idx]}.csv')
    df_shap.to_csv(csv_path, index=False)
    print(f"Saved SHAP values for sample {sample_idx} to {csv_path}")

    return shap_values, explainer


def shap_explain_all(clf, X_test, y_test, feature_names):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("Explaining model predictions using SHAP for all correctly samples")

    explainer = shap.Explainer(clf, X_test)
    y_pred = clf.predict(X_test)

    correct_indices = np.where(y_pred == y_test)[0]    
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")
    
    X_correct = X_test[correct_indices]
    correct_labels = y_test[correct_indices]

    print("Shape of correct_labels:",correct_labels.shape)


    explainer = shap.Explainer(clf, X_test) 
    shap_values_correct = explainer(X_correct)

    print("shap_values_correct[0].values.shape")
    print(shap_values_correct[0].values.shape)

    print(f"Computed SHAP values for {len(correct_indices)} correctly predicted samples.")

    all_dfs = []
    for i, idx in enumerate(correct_indices):

        pred_class = y_pred[idx]
        shap_vals = shap_values_correct[i].values[:, pred_class]

        if shap_vals.ndim == 2:
            pred_class = y_pred[idx]
            shap_vals = shap_vals[pred_class]

        assert len(shap_vals) == len(feature_names), \
            f"SHAP value length {len(shap_vals)} doesn't match feature count {len(feature_names)}"

        df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_vals,
            'sample_index': idx,
            'true_label': correct_labels[i]
        })

        all_dfs.append(df)
    
    result_df = pd.concat(all_dfs)
    result_path = os.path.join(results_dir, 'shap_values_all_correct.csv')
    result_df.to_csv(result_path, index=False)
    print(f"Saved SHAP values for all correctly predicted samples to {result_path}")

    return shap_values_correct, correct_indices, explainer



adata = ad.read_h5ad('/data1/data/corpus/Zheng68K.h5ad')  
X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
feature_names = adata.var_names
le = LabelEncoder()
y = le.fit_transform(adata.obs['celltype'])
X_train, y_train, X_test, y_test = h.split_data(X, y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
lr = h.train_logistic_regression(X_train, y_train)
#shap_values, explainer = shap_explain(lr, X_test, y_test, feature_names)
shap_values_correct, correct_indices, explainer = shap_explain_all(lr, X_test, y_test, feature_names)

