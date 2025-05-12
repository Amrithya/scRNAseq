import os
import torch
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def log_norm(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    cell_type_series = adata.obs['cell_type']
    le = LabelEncoder()
    y = le.fit_transform(cell_type_series)
    X = adata.X
    return X, y, le

def preprocess_data(adata, samp=False, cluster=False):

    """
    Preprocess the input AnnData object by normalizing and log-transforming the data.
    Also encodes the 'cell_type' column in adata.obs to numerical labels.

    Parameters:
    -----------
    adata : AnnData
        The raw AnnData object containing gene expression matrix and metadata.

    Returns:
    --------
    adata : AnnData
        The processed AnnData object.
    y : np.ndarray
        Encoded class labels.
    """

    X, y,le = log_norm(adata)
    if samp == False :
        X_train, y_train, X_test, y_test = split_data(X,y)
    else:
        X_balanced, y_balanced = do_smote(X, y, cluster)
        X_train, y_train, X_test, y_test = split_data(X_balanced,y_balanced)
    return X_train, y_train, X_test, y_test, le

def preprocess_data(device, adata, samp=False, cluster=False):
    X, y,le = log_norm(adata)
    input_size = X.shape[1]
    output_size = len(le.classes_)
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y.toarray(), dtype=torch.long)
    if samp == False :
        X_train, y_train, X_test, y_test = split_data(X_tensor,y_tensor)
    else:
        X_balanced, y_balanced = do_smote(X_tensor, y_tensor, cluster)
        X_train, y_train, X_test, y_test = split_data(X_balanced,y_balanced) 
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    y_train_np = np.array(y_train).flatten()
    classes = np.unique(y_train_np)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_np)
    weights = torch.tensor(weights, dtype=torch.float)

    return train_data, test_data, weights, le, input_size, output_size


def split_data(X,y):
    SAMPLING_FRACS = [1.0]
    for frac in SAMPLING_FRACS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022) #update Aug 2023: hold train/val across all runs #same train/val set split for each frac in k
        for index_train, index_val in sss.split(X, y):
            index_train_small = np.random.choice(index_train, round(index_train.shape[0]*frac), replace=False)
            X_train, y_train = X[index_train_small], y[index_train_small]
            X_test, y_test = X[index_val], y[index_val]
    return X_train, y_train, X_test, y_test


def train_rf(X,y):

    """
    Train a Random Forest classifier.

    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Class labels.

    Returns:
    --------
    model : RandomForestClassifier
        The trained random forest model.
    """
     
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model

def train_logistic_regression(X, y):
    """
    Train a Logistic Regression model with L1 regularization.

    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Class labels.

    Returns:
    --------
    model : LogisticRegression
        Trained logistic regression model.
    """

    model = LogisticRegression(penalty="l1", C=0.1,solver="liblinear")
    print(model)
    model.fit(X, y)
    return model

def train_sgd(X,y):
    """
    Train a SGDClassifier with max_iter=1000, tol=1e-3

    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Class labels.

    Returns:
    --------
    model : SGDClassifier
        Trained SGDClassifier.
    """

    model = SGDClassifier(max_iter=1000, tol=1e-3)
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    """
    Train an XGBoost classifier.

    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Class labels.

    Returns:
    --------
    model : XGBClassifier
        Trained XGBoost model.
    """

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model

def evaluate_model(clf, X, y, label_encoder=None, mode="",model="",down_samp=""):

    """
    Evaluate the given model on the data and log the results.

    Parameters:
    -----------
    clf : classifier
        Trained classifier to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        True labels.
    label_encoder : LabelEncoder, optional
        Used to decode labels for reporting.
    mode : str
        "train" or "test", used for tracking output.
    model : str
        Model name, e.g. "lr" or "rf".
    down_samp : bool or str
        Whether dataset was downsampled.

    Returns:
    --------
    None
    """

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    results = {
        'Mode': [mode],
        'Overall Accuracy': [acc],
        'Model':[model],
        'Down sampling':[down_samp]
    }
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    class_accuracies = {}
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        correct_preds = np.sum(y_pred[label_indices] == y[label_indices])
        class_accuracy = correct_preds / len(label_indices)
        class_name = label_encoder.inverse_transform([label])[0] if label_encoder else label
        class_accuracies[class_name] = class_accuracy
        results[f"{class_name}"] = [class_accuracy]

    if mode == "test":
        val_macro_f1 = f1_score(y, y_pred, average="macro")
        results['Macro F1'] = [val_macro_f1]
        plot_filename = os.path.join(results_dir, f"confusion_matrix_{model}_{'down' if down_samp else 'full'}.png")
        plot_confusion_matrix(y,y_pred,label_encoder,plot_filename)
    df = pd.DataFrame(results)

    results_file = os.path.join(results_dir, f"results_file_{model}_{'down' if down_samp else 'full'}.csv")
    if os.path.exists(results_file):
        os.remove(results_file)
    df.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
    print(f"Results saved to {results_file}")

def plot_confusion_matrix(y_true, y_pred, label_encoder=None,save_path=None):

    """
    Generate and save/show a confusion matrix as a heatmap.

    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    label_encoder : LabelEncoder, optional
        If provided, used to decode class indices to names.
    save_path : str, optional
        If provided, the plot will be saved to this path.

    Returns:
    --------
    None
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    labels = label_encoder.classes_ if label_encoder else np.unique(y_true)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    else:
        plt.show()

def do_smote(X, y, cluster=False):
    """
    Apply SMOTE to balance the dataset and train a model.

    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Class labels.
    Returns:
    --------
    X_balanced, y_balanced
    """

    print("Before SMOTE:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(save_path, exist_ok=True)
    if cluster:
        adata_file = sc.read_h5ad('/data1/data/corpus/pbmc68k_balanced_data.h5ad')
    else:
        adata_file = os.path.join(save_path, "pbmc68k_balanced_data.h5ad")
    if os.path.exists(adata_file):
            print("Balanced data exists..")
            adata = ad.read_h5ad(adata_file)
            X_balanced = adata.X
            y_balanced = adata.obs['label'].values
    else:
            smote = SMOTE(random_state=2022)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            adata = ad.AnnData(X_balanced)
            adata.obs['label'] = pd.Categorical(y_balanced)
            adata.write(adata_file)
    print("After SMOTE:")
    print(f"X shape: {X_balanced.shape}")
    print(f"y shape: {y_balanced.shape}")
    print("Class distribution after SMOTE:")
    print(pd.Series(y_balanced).value_counts())
    return X_balanced, y_balanced
        

def shap_explain(clf,X_train, X_test):
    """
    Function to train a Logistic Regression model and explain it using SHAP.
    
    Returns:
    - shap_values: SHAP values for the test data
    - model: Trained logistic regression model
    - explainer: SHAP explainer object
    """

    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.force_plot(shap_values[0])
    shap.dependence_plot("mean radius", shap_values, X_test)


