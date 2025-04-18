import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


adata = sc.read_h5ad('pbmc68k(2).h5ad')
X = adata.X

cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
encoded_labels = le.fit_transform(cell_type_series)

sc.pp.normalize_total(adata, target_sum=1e4)  
sc.pp.log1p(adata)  

X_normalized = adata.X 

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized,
    encoded_labels,
    test_size=0.2,
    random_state=42,
)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
decoded_preds = le.inverse_transform(y_pred)
decoded_true = le.inverse_transform(y_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

y_test_labels = le.inverse_transform(y_test)
mask = (y_test_labels == 'CD4+ T Helper2') | (y_test_labels == 'CD34+')
X_filtered = X_test[mask]
y_filtered = y_test[mask]
score_filtered = clf.score(X_filtered, y_filtered)
print(f"Score on hard cells: {score_filtered:.4f}")