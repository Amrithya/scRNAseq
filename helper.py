import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(adata):

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

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    cell_type_series = adata.obs['cell_type']
    le = LabelEncoder()
    y = le.fit_transform(cell_type_series)
    return adata,y,le

def balance_dataset(adata, X, y):

    """
    Downsamples each class in the dataset to the size of the smallest class for balancing.

    Parameters:
    -----------
    adata : AnnData
        AnnData object (used for updating obs with new labels).
    X : array-like
        Feature matrix (cells x genes).
    y : array-like
        Class labels.

    Returns:
    --------
    X_balanced : np.ndarray
        Downsampled feature matrix.
    y_balanced : np.ndarray
        Downsampled label array.
    """

    assert len(y) == X.shape[0], f"Mismatch: len(y)={len(y)}, X.shape[0]={X.shape[0]}"

    adata.obs['label'] = y
    min_size = pd.Series(y).value_counts().min()

    balanced_indices = []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        selected = np.random.choice(label_indices, min_size, replace=False)
        balanced_indices.extend(selected)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced

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

    model = LogisticRegression(penalty="l1", solver="liblinear")
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

def evaluate_model(clf, X, y, label_encoder=None, mode="",model="",down_samp="", results_file=""):

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
    results_file : str
        Path to CSV file for storing evaluation metrics.

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
        plot_filename = f"confusion_matrix_{model}_{'down' if down_samp else 'full'}.png"
        plot_confusion_matrix(y,y_pred,label_encoder,plot_filename)
    df = pd.DataFrame(results)
    df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)
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