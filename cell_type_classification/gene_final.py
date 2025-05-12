import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import argparse
from sklearn.preprocessing import LabelEncoder
import helper as h

def run_model(model,X_train, y_train, X_test, y_test, down_samp, le):
    """
    Trains and evaluates a classification model (Logistic Regression or Random Forest)
    on the given AnnData object. Optionally performs downsampling to balance class labels.

    Parameters:
    -----------
    model : str
        The model to train, either "lr" for Logistic Regression or "rf" for Random Forest.

    X_train : np.ndarray
        The training feature matrix.
    y_train : np.ndarray
        The training labels.
    X_test : np.ndarray
        The test feature matrix.
    y_test : np.ndarray
        The test labels.

    Returns:
    --------
    None
    """
    if model == "lr":
        lr = h.train_logistic_regression(X_train, y_train)
        h.evaluate_model(lr, X_train, y_train,"train",model,down_samp)
        h.evaluate_model(lr, X_test, y_test,"test",model,down_samp)
        #h.shap_explainer(lr, X_train, X_test)
    elif model == "rf":
        rf = h.train_rf(X_train, y_train)
        h.evaluate_model(rf, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(rf, X_test, y_test, le,"test",model,down_samp)
        #h.shap_explainer(lr, X_train, X_test)

    elif model == "sgd":
        rf = h.train_sgd(X_train, y_train)
        h.evaluate_model(rf, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(rf, X_test, y_test, le,"test",model,down_samp)
        #h.shap_explainer(lr, X_train, X_test)
    elif model == "xg":
        xg = h.train_xgboost(X_train, y_train)
        h.evaluate_model(xg, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(xg, X_test, y_test, le,"test",model,down_samp)
        #h.shap_explainer(lr, X_train, X_test)


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
        python gene_final.py -m rf -d -c  # if you run on cluster,include -c
        python gene_final.py --model lr --down_samp
    """

    cmdline_parser = argparse.ArgumentParser('Training')

    cmdline_parser.add_argument('-m', '--model',
                                default="lr",
                                help='model name',
                                type=str)
    
    cmdline_parser.add_argument('-d', '--down_samp',
                                action='store_true',
                                help='enable down sampling')
    
    cmdline_parser.add_argument('-c', '--cluster',
                                action='store_true',
                                help='dataset file location')
    
    cmdline_parser.add_argument('-s', '--output',
                                default="evaluation_results.csv",
                                help='output_file',
                                type=str)
    
    args, unknowns = cmdline_parser.parse_known_args()

    if args.cluster:
        adata = sc.read_h5ad('/data1/data/corpus/pbmc68k(2).h5ad')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', 'data', 'pbmc68k(2).h5ad')
        adata = sc.read_h5ad(file_path)

    X_train, y_train, X_test, y_test, le = h.preprocess_data(adata, args.down_samp, args.cluster)
    
    run_model(args.model, X_train, y_train, X_test, y_test, args.down_samp, le)
    

