import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from helper import train_logistic_regression, evaluate_model,balance_dataset,preprocess_data,train_rf,train_sgd,train_xgboost


def run_model(model,down_samp,adata,y,le):
    """
    Trains and evaluates a classification model (Logistic Regression or Random Forest)
    on the given AnnData object. Optionally performs downsampling to balance class labels.

    Parameters:
    -----------
    model : str
        The model to train, either "lr" for Logistic Regression or "rf" for Random Forest.
    
    down_samp : bool
        Whether to perform downsampling to balance class distributions.
        
    adata : anndata.AnnData
        The AnnData object containing gene expression data (adata.X) and metadata.
    
    y : np.ndarray
        The encoded labels for classification.
    
    le : sklearn.preprocessing.LabelEncoder
        LabelEncoder instance used to map original class names to integers.

    Returns:
    --------
    None
    """
    X = adata.X
    #print(model,down_samp,adata,y,le)
    if down_samp == False :
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    else:
        X_balanced,y_balanced = balance_dataset(adata,X,y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced,
            y_balanced,
            test_size=0.2,
            random_state=42,
            stratify=y_balanced
        )

    if model == "lr":
        lr = train_logistic_regression(X_train, y_train)
        evaluate_model(lr, X_train, y_train, le,"train",model,down_samp)
        evaluate_model(lr, X_test, y_test, le,"test",model,down_samp)
    elif model == "rf":
        rf = train_rf(X_train, y_train)
        evaluate_model(rf, X_train, y_train, le,"train",model,down_samp)
        evaluate_model(rf, X_test, y_test, le,"test",model,down_samp)
    elif model == "sgd":
        rf = train_sgd(X_train, y_train)
        evaluate_model(rf, X_train, y_train, le,"train",model,down_samp)
        evaluate_model(rf, X_test, y_test, le,"test",model,down_samp)
    elif model == "xg":
        xg = train_xgboost(X_train, y_train)
        evaluate_model(xg, X_train, y_train, le,"train",model,down_samp)
        evaluate_model(xg, X_test, y_test, le,"test",model,down_samp)


if __name__ == "__main__":

    """
    Main script for training and evaluating classification models on single-cell gene expression data.

    This script performs the following:
    1. Loads the PBMC68k dataset from an .h5ad file.
    2. Preprocesses the data (normalization, log transformation, label encoding).
    3. Accepts command-line arguments to:
        - Choose the model type (logistic regression or random forest).
        - Enable or disable downsampling for class balancing.
        - Specify the output CSV file to store evaluation results.
    4. Trains the model on the training data.
    5. Evaluates the model on both training and test data.
    6. Saves the performance metrics and confusion matrix to disk.

    Example usage:
        python gene_final.py -m rf -d -s results.csv
        python gene_final.py --model lr --output results.csv
    """

    adata = sc.read_h5ad('/data1/data/corpus/pbmc68k(2).h5ad')

    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #file_path = os.path.join(current_dir, '..', 'data', 'pbmc68k(2).h5ad')
    #adata = sc.read_h5ad(file_path)

    adata,y,le = preprocess_data(adata)

    cmdline_parser = argparse.ArgumentParser('Training')

    cmdline_parser.add_argument('-m', '--model',
                                default="lr",
                                help='model name',
                                type=str)
    
    cmdline_parser.add_argument('-d', '--down_samp',
                                action='store_true',
                                help='enable down sampling')
    
    cmdline_parser.add_argument('-s', '--output',
                                default="evaluation_results.csv",
                                help='output_file',
                                type=str)
    
    args, unknowns = cmdline_parser.parse_known_args()
    
    run_model(args.model, args.down_samp, adata, y,le)
    

