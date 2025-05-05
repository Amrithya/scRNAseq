import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)

file_path = os.path.join(current_dir,'..', 'data', 'Zheng68K.h5ad')
file_path = os.path.abspath(file_path)


file_path1 = os.path.join(current_dir, '..','..', 'data', 'pbmc68k(2).h5ad')
file_path1 = os.path.abspath(file_path1)

adata = sc.read_h5ad(file_path)
adata1 = sc.read_h5ad(file_path1)

print(adata.head())

print(adata1.head())

