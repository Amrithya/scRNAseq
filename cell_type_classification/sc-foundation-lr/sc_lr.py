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
file_path = os.path.join(current_dir, '..','..', 'data', 'pbmc68k(2).h5ad')
file_path = os.path.abspath(file_path)

print(file_path)
if os.path.exists(file_path):
    adata = sc.read_h5ad(file_path)
else:
    print(f"File not found: {file_path}")


if(file_path == "/home/amrithya/Desktop/codes/scRNAseq/data/pbmc68k(2).h5ad"):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    data = adata.X
    cell_type_series = adata.obs['cell_type']
    le = LabelEncoder()
    label = le.fit_transform(cell_type_series)
else:
    data = adata.X
    label = adata.obs.celltype

SAMPLING_FRACS = [1.0]

ks = []
fracs = []
cs=[]
train_accs = []
test_accs = []
test_f1s = []
for frac in SAMPLING_FRACS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022) #update Aug 2023: hold train/val across all runs #same train/val set split for each frac in k
        for index_train, index_val in sss.split(data, label):
            index_train_small = np.random.choice(index_train, round(index_train.shape[0]*frac), replace=False)
            X_train, y_train = data[index_train_small], label[index_train_small]
            X_test, y_test = data[index_val], label[index_val]

        print("Loaded data...")

        c = 0.1
        lr = LogisticRegression(penalty="l1", C=c, solver="liblinear") #random_state=0, 
        lr.fit(X_train, y_train)
        train_acc = lr.score(X_train, y_train)
        test_acc = lr.score(X_test, y_test)
        print("train set accuracy: " + str(np.around(train_acc, 4)))
        print("test set accuracy: " + str(np.around(test_acc, 4)))
        val_macro_f1 = f1_score(y_test, lr.predict(X_test), average="macro")
        print("test set macro F1: " + str(np.around(val_macro_f1, 4)))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_f1s.append(val_macro_f1)
        
        print("\n")
