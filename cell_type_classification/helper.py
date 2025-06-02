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


def load_data(samp,cluster):
    if cluster:
        if samp:
            train_path = '/data1/data/corpus/Zheng68K_smote_data_train.h5ad'
            test_path = '/data1/data/corpus/Zheng68K_smote_data_test.h5ad'

            if os.path.exists(train_path) and os.path.exists(test_path):
                print("Loading balanced and preprocessed data on cluster")
                
                adata_train = sc.read_h5ad(train_path)
                adata_test = sc.read_h5ad(test_path)
                le = LabelEncoder()
                X_train = adata_train.X
                y_train = adata_train.obs['label'].values
                print(f"X_train shape: {X_train.shape}")
                print(f"y_train shape: {y_train.shape}")

                X_test = adata_test.X
                y_test = adata_test.obs['label'].values
            else:
                print("Preprocessing raw data with SMOTE on cluster")
                adata = sc.read_h5ad('/data1/data/corpus/Zheng68K.h5ad')
                X_train, y_train, X_test, y_test, le = preprocess_data(adata, samp, cluster)
                adata_train = sc.AnnData(X_train)
                adata_train.obs['label'] = y_train  
                adata_test = sc.AnnData(X_test)
                adata_test.obs['label'] = y_test

                adata_train.write('/data1/data/corpus/Zheng68K_smote_data_train.h5ad')
                adata_test.write('/data1/data/corpus/Zheng68K_smote_data_test.h5ad')
        else:
            print("Preprocessing raw data on cluster")
            adata = sc.read_h5ad('/data1/data/corpus/Zheng68K.h5ad')
            X_train, y_train, X_test, y_test, le = preprocess_data(adata, samp, cluster)

    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if samp:
            file_path = os.path.join(current_dir, '..', 'data', 'Zheng68K_smote_data.h5ad')
            if os.path.exists(file_path):
                print("Loading balanced and preprocessed data on local")
                adata = sc.read_h5ad(file_path)
                X = adata.X
                cell_type_series = adata.obs['label']
                le = LabelEncoder()
                y = le.fit_transform(cell_type_series)
                X_train, y_train, X_test, y_test = split_data(X,y)
            else:
                print("Preprocessing raw data with SMOTE on local")
                adata = sc.read_h5ad(os.path.join(current_dir, '..', 'data', 'Zheng68K.h5ad'))
                X_train, y_train, X_test, y_test, le = preprocess_data(adata, samp, cluster)
        else:
            print("Preprocessing raw data without SMOTE on local")
            adata = sc.read_h5ad(os.path.join(current_dir, '..', 'data', 'Zheng68K.h5ad'))
            X_train, y_train, X_test, y_test, le = preprocess_data(adata, samp, cluster)
        
    return X_train, y_train, X_test, y_test, le

def log_norm(adata):
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    cell_type_series = adata.obs['celltype']
    le = LabelEncoder()
    y = le.fit_transform(cell_type_series)
    X = adata.X
    return X, y, le

def preprocess_data(adata, samp, cluster):

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
        X_train, y_train, X_test, y_test = split_data(X,y)
        X_train, y_train = do_smote(X_train, y_train)
    return X_train, y_train, X_test, y_test, le

def preprocess_data_nn(device, X_train, y_train, X_test, y_test, le):
    print("Preprocessing data for neural network")
    input_size = X_train.shape[1]
    output_size = len(le.classes_)
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    y_train_np = y_train.numpy().flatten()
    classes = np.unique(y_train_np)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_np)
    weights = torch.tensor(weights, dtype=torch.float)

    return train_data, test_data, weights, le, input_size, output_size


def split_data(X,y):
    print("Splitting data into train and test sets")
    SAMPLING_FRACS = [1.0]
    for frac in SAMPLING_FRACS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022)
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
    print("Training Logistic Regression model")
    model = LogisticRegression(penalty="l1", C=0.1,solver="liblinear")
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
    print("Training SGDClassifier model")
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
    print("Training XGBoost model")
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model

def evaluate_model(clf, X, y, label_encoder=None, mode="",model="",samp=""):

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
    samp : bool or str
        Whether dataset was downsampled.

    Returns:
    --------
    None
    """
    print(f"Evaluating {model} model on {mode} data")
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    results = {
        'Mode': [mode],
        'Overall Accuracy': [acc],
        'Model':[model],
        'Down sampling':[samp]
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
        macro_accuracy = np.mean(list(class_accuracies.values()))
        results['Macro Accuracy'] = [macro_accuracy]
        results['Micro F1'] = [f1_score(y, y_pred, average='micro')]
        results['Macro F1'] = [f1_score(y, y_pred, average='macro')]
        results['Micro Precision'] = [precision_score(y, y_pred, average='micro')]
        results['Macro Precision'] = [precision_score(y, y_pred, average='macro')]
        results['Micro Recall'] = [recall_score(y, y_pred, average='micro')]
        results['Macro Recall'] = [recall_score(y, y_pred, average='macro')]
    
    df = pd.DataFrame(results)
    
    results_file = os.path.join(results_dir, f"results_file_{model}_{'smote' if samp else 'raw'}.csv")
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

def do_smote(X, y):
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
    X_train, y_train
    """

    print("Before SMOTE:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    smote = SMOTE(random_state=2022)
    X_train, y_train = smote.fit_resample(X, y)
    adata = ad.AnnData(X_train)
    adata.obs['label'] = pd.Categorical(y_train)
    print("After SMOTE:")
    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train).value_counts())
    adata.write('/data1/data/corpus/pbmc68k_balanced_data2.h5ad')
    return X_train, y_train
        

def shap_explain(clf, X_train, X_test, model):
    """
    Function to explain model predictions using SHAP and save results.

    Returns:
    - shap_values: SHAP values for the test data
    - model: Trained model
    - explainer: SHAP explainer object
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        print("Explaining model predictions using SHAP")
        explainer = shap.Explainer(clf, X_train)
        shap_values = explainer(X_test)

        shap.summary_plot(shap_values, X_test)
        shap.force_plot(shap_values[0])
        shap.dependence_plot("mean radius", shap_values, X_test)

        shap_df = pd.DataFrame(shap_values.values[:5], columns=X_test.columns)
        shap_df["instance"] = range(1, 6)
        shap_df["method"] = "SHAP"

        results_file = os.path.join(results_dir, f"results_file_{model}_xai.csv")
        shap_df.to_csv(results_file, index=False)
        return shap_values, model, explainer

    except Exception as e:
        print(f"Error during SHAP explanation: {e}")
        return None, model, None


def lime_explain(clf, X_train, X_test, le, model):
    """
    Function to explain a classifier's predictions using LIME and save results.

    Returns:
    - lime_explainer: LIME explainer object
    """
    try:
        print("Explaining model predictions using LIME")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)

        lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=le.classes_.tolist(),
            mode="classification"
        )

        lime_data = []
        for i in range(5):
            exp = lime_explainer.explain_instance(X_test.iloc[i].values, clf.predict_proba, num_features=10)
            exp.show_in_notebook(show_table=True)

            for feature, weight in exp.as_list():
                lime_data.append({
                    "instance": i + 1,
                    "feature": feature,
                    "weight": weight,
                    "method": "LIME"
                })

        lime_df = pd.DataFrame(lime_data)

        results_file = os.path.join(results_dir, f"results_file_{model}_xai.csv")
        if os.path.exists(results_file):
            lime_df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            lime_df.to_csv(results_file, index=False)

        return lime_explainer

    except Exception as e:
        print(f"Error during LIME explanation: {e}")
        return None