import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle as pkl
import umap.umap_ as umap
import matplotlib.pyplot as plt

embedding_path = "inference_embeddings.npy"
embeddings = np.load(embedding_path)  

label_path = "./ckpts/label.pkl"
with open(label_path, 'rb') as f:
    labels = pkl.load(f)

if hasattr(labels, 'numpy'):
    labels = labels.numpy()

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Logistic Regression Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
embedding_2d = umap_model.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
plt.title("UMAP of PerformerLM Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(scatter, label="Cell Type")
plt.tight_layout()

plt.savefig("umap_performerlm.png", dpi=300)
