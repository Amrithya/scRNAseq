import os
import torch
import argparse
import numpy as np
import helper as h
import scanpy as sc
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import cell_type_classification.lrp_nn as lrp_nn


class NNet_LRP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(NNet_LRP, self).__init__()
        self.fc1 = lrp_nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = lrp_nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = lrp_nn.Linear(hidden_size // 2, output_size)

    def forward(self, x, explain=False, rule="alpha1beta0"):
        self.explain = explain
        self.rule = rule

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def relprop(self, R):
        R = self.fc3.relprop(R, rule=self.rule)
        R = self.dropout2(R)
        R = self.relu2(R)
        R = self.fc2.relprop(R, rule=self.rule)
        R = self.dropout1(R)
        R = self.relu1(R)
        R = self.fc1.relprop(R, rule=self.rule)
        return R


def train_nn(device, train_data, test_data, lr_rate, weights, input_size, output_size, dropout_rate, hidden_size):
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = NNet_LRP(input_size, hidden_size, output_size, dropout_rate).to(device)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    print(f"\nTraining with Hidden size: {hidden_size}, learning rate: {lr_rate}, dropout rate: {dropout_rate}")
    l1_lambda = 1e-5
    num_epochs = 10

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
    print(f"Hidden size {hidden_size}, learning rate {lr_rate}, dropout {dropout_rate} => "
          f"Train Accuracy: {epoch_accuracy:.2f}% :: Test Accuracy: {test_accuracy:.2f}%")

    return test_accuracy, epoch_accuracy, model


def explain_prediction(model, input_tensor, device, rule="alpha1beta0"):
    model.eval()
    input_tensor = input_tensor.to(device).requires_grad_(True)

    output = model(input_tensor, explain=True, rule=rule)
    output_selected = output[torch.arange(input_tensor.size(0)), output.max(1)[1]]
    output_selected = output_selected.sum()
    output_selected.backward()

    relevance = model.relprop(output)
    return relevance
