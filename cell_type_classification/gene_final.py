import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import helper as h

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
        SAMPLING_FRACS = [1.0]
        for frac in SAMPLING_FRACS:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022) #update Aug 2023: hold train/val across all runs #same train/val set split for each frac in k
            for index_train, index_val in sss.split(X, y):
                index_train_small = np.random.choice(index_train, round(index_train.shape[0]*frac), replace=False)
                X_train, y_train = X[index_train_small], y[index_train_small]
                X_test, y_test = X[index_val], y[index_val]
    else:
        X_balanced,y_balanced = h.balance_dataset(adata,X,y)
        SAMPLING_FRACS = [1.0]
        for frac in SAMPLING_FRACS:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022) #update Aug 2023: hold train/val across all runs #same train/val set split for each frac in k
            for index_train, index_val in sss.split(X_balanced, y_balanced):
                index_train_small = np.random.choice(index_train, round(index_train.shape[0]*frac), replace=False)
                X_train, y_train = X_balanced[index_train_small], y_balanced[index_train_small]
                X_test, y_test = X_balanced[index_val], y_balanced[index_val]

    if model == "lr":
        lr = h.train_logistic_regression(X_train, y_train)
        h.evaluate_model(lr, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(lr, X_test, y_test, le,"test",model,down_samp)
    elif model == "rf":
        rf = h.train_rf(X_train, y_train)
        h.evaluate_model(rf, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(rf, X_test, y_test, le,"test",model,down_samp)
    elif model == "sgd":
        rf = h.train_sgd(X_train, y_train)
        h.evaluate_model(rf, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(rf, X_test, y_test, le,"test",model,down_samp)
    elif model == "xg":
        xg = h.train_xgboost(X_train, y_train)
        h.evaluate_model(xg, X_train, y_train, le,"train",model,down_samp)
        h.evaluate_model(xg, X_test, y_test, le,"test",model,down_samp)


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

    adata,y,le = h.preprocess_data(adata)
    
    run_model(args.model, args.down_samp, adata, y,le)
    

