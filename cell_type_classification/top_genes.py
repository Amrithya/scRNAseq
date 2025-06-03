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


def get_top_genes(adata):
    """
    Extracts the top n_genes based on variance from the AnnData object.

    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the gene expression data.
    n_genes : int
        The number of top genes to extract based on variance.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the top n_genes and their variance.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    X = adata.X
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['celltype'])
    X_train, X_test, y_train, y_test = h.split_data(X, y)
    
    cls = LogisticRegression(penalty="l1", C=0.1,solver="liblinear")
    cls.fit(X_train, y_train)

    top_k_list = [100, 250, 500, 1000]
    top_genes = pd.DataFrame()
    for top_k in top_k_list:
        importances = np.mean(np.abs(cls.coef_), axis=0)
        indices = np.argsort(importances)[-top_k:][::-1]
        top_genes_k = pd.DataFrame({
            'gene': adata.var_names[indices],
        })
        
        top_genes_k['top_k'] = top_k
        top_genes = pd.concat([top_genes, top_genes_k], ignore_index=True)
        X_train_k = X_train[:, indices]
        X_test_k = X_test[:, indices]
        cls_k = LogisticRegression(penalty="l1", C=0.1, solver="liblinear")
        cls_k.fit(X_train_k, y_train)
        y_pred_train_k = cls_k.predict(X_train_k)
        y_pred_test_k = cls_k.predict(X_test_k)
        acc_train_k = accuracy_score(y_train, y_pred_train_k)
        acc_test_k = accuracy_score(y_test, y_pred_test_k)
        f1_train_k = f1_score(y_train, y_pred_train_k, average='weighted')
        f1_test_k = f1_score(y_test, y_pred_test_k, average='weighted')
        precision_train_k = precision_score(y_train, y_pred_train_k, average='weighted')    
        precision_test_k = precision_score(y_test, y_pred_test_k, average='weighted')
        recall_train_k = recall_score(y_train, y_pred_train_k, average='weighted')
        recall_test_k = recall_score(y_test, y_pred_test_k, average='weighted')
        df = pd.DataFrame({
            'top_k': [top_k],
            'accuracy_train': [acc_train_k],
            'accuracy_test': [acc_test_k],
            'f1_train': [f1_train_k],
            'f1_test': [f1_test_k],
            'precision_train': [precision_train_k],
            'precision_test': [precision_test_k],
            'recall_train': [recall_train_k],
            'recall_test': [recall_test_k]
        })
        df.to_csv(os.path.join(results_dir, f'top_genes_{top_k}.csv'), index=False)
        print(f"Top {top_k} genes extracted and saved to results directory.")
        
    top_genes.to_csv(os.path.join(results_dir, 'top_genes_all.csv'), index=False)
    print("All top genes extracted and saved to results directory.")
    
adata = ad.read_h5ad('/data1/data/corpus/Zheng68K.h5ad')  
get_top_genes(adata)  