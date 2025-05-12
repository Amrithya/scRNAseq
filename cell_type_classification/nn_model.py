import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import scanpy as sc
import os
import helper as h

def train_nn(X, y, lr_rate, dropout_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    le = LabelEncoder()
    y_tensor = torch.tensor(le.fit_transform(y), dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    y_train_np = np.array(y_train).flatten()
    classes = np.unique(y_train_np)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_np)
    weights = torch.tensor(weights, dtype=torch.float)

    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class NNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout_rate):
            super(NNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(hidden_size // 2, output_size)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x

    input_size = X.shape[1]
    hidden_size = 256
    output_size = len(le.classes_)
    num_epochs = 10

    model = NNet(input_size, hidden_size, output_size, dropout_rate).to(device)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    print(f"\nTraining with learning rate: {lr_rate}, dropout rate: {dropout_rate}")
    l1_lambda = 1e-5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total * 100
    print(f"Test Accuracy with learning rate {lr_rate}, dropout {dropout_rate}: {test_accuracy:.2f}%")
    return test_accuracy


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser('Training')
    cmdline_parser.add_argument('-c', '--cluster', action='store_true', help='Use cluster path')
    args, _ = cmdline_parser.parse_known_args()

    if args.cluster:
        adata = sc.read_h5ad('/data1/data/corpus/pbmc68k(2).h5ad')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', 'data', 'pbmc68k(2).h5ad')
        adata = sc.read_h5ad(file_path)

    X = adata.X
    adata, y, le = h.preprocess_data(adata)

    lr_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.0, 0.3, 0.5]
    results = []

    for dropout in dropout_rates:
        for lr in lr_rates:
            accuracy = train_nn(X, y, lr, dropout)
            results.append((lr, dropout, accuracy))

    print("\nSummary of Results:")
    for lr, dropout, acc in results:
        print(f"LR: {lr}, Dropout: {dropout}, Test Accuracy: {acc:.2f}%")