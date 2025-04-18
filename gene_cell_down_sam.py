import scanpy as sc
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

adata = sc.read_h5ad('pbmc68k(2).h5ad')
X = adata.X

cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
encoded_labels = le.fit_transform(cell_type_series)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X_normalized = adata.X
y = encoded_labels

NREPS = 1
SAMPLING_FRACS = [0.5, 0.2]

train_accs = []
test_accs = []
test_f1s = []

for k in range(NREPS):
    for frac in SAMPLING_FRACS:
        print(f"\nRepetition {k}, Sampling Fraction: {frac}")
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in sss.split(X_normalized, y):
            np.random.seed(k)
            train_idx_downsampled = np.random.choice(
                train_idx,
                size=round(len(train_idx) * frac),
                replace=False
            )
            X_train = X_normalized[train_idx_downsampled]
            y_train = y[train_idx_downsampled]
            X_test = X_normalized[test_idx]
            y_test = y[test_idx]

            lr = LogisticRegression(penalty="l1",solver="liblinear")
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            train_acc = accuracy_score(y_train, lr.predict(X_train))
            test_acc = accuracy_score(y_test, y_pred)
            val_macro_f1 = f1_score(y_test, y_pred, average="macro")

            print("Train set accuracy: " + str(np.around(train_acc, 4)))
            print("Test set accuracy: " + str(np.around(test_acc, 4)))
            print("Test set macro F1: " + str(np.around(val_macro_f1, 4)))

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_f1s.append(val_macro_f1)
