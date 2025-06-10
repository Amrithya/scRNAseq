import os
import copy
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

class LRP:
    '''
    Implementation of Layer-wise Relevance Propagation for your neural network
    '''
    def __init__(self, model, rule='epsilon', epsilon=1e-7):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.rule = rule
        
    def forward(self, x):
        self.local_input = x.clone().requires_grad_(True)
        return self.model(self.local_input)
    
    def relprop(self, x, R=None):
        if hasattr(self, 'local_input') and self.local_input.grad is not None:
            self.local_input.grad.zero_()

        self.local_input = x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            output = self.model(self.local_input)
            if R is None:
                R = torch.zeros_like(output)
                R.scatter_(1, output.argmax(dim=1, keepdim=True), 1.0)
            output.backward(R, retain_graph=True)
            print("local_input.grad is", self.local_input.grad)
            print("local_input requires_grad:", self.local_input.requires_grad)
            print("output shape:", output.shape)
            print("R shape:", R.shape)
            relevance = self.local_input.grad * self.local_input.data
            self.local_input.grad = None
        return relevance.detach()
    __call__ = relprop


def train_nn(device, train_data, test_data, lr_rate, weights, input_size, output_size, dropout_rate, hidden_size):
    
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
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    num_epochs = 10

    save_path = "/data1/data/corpus/scMODEL/lrp_nn_model_Zheng68K.pth"

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model = NNet(input_size, hidden_size, output_size, dropout_rate).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded existing model from {save_path}")
        model.eval()
        train_correct, train_total = 0, 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                train_accuracy = train_correct / train_total * 100
                
                test_correct, test_total = 0, 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        test_correct += (predicted == labels).sum().item()
                        test_total += labels.size(0)
                test_accuracy = test_correct / test_total * 100
                
        print(f"Input size {input_size}, Hidden size {hidden_size}, learning rate {lr_rate}, dropout {dropout_rate}=> Train Accuracy:{train_accuracy:.2f} :: Test Accuracy: {test_accuracy:.2f}%")

        return model, test_accuracy, train_accuracy
    
    else:
        model = NNet(input_size, hidden_size, output_size, dropout_rate).to(device)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)

        print(f"\nTraining with Hidden size: {hidden_size},learning rate: {lr_rate}, dropout rate: {dropout_rate}")
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

        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'dropout_rate': dropout_rate,
            'lr_rate': lr_rate,
            'weights': weights.cpu(),
            }, save_path)
        print(f"Model saved to {save_path}")

        print(f"Input size {input_size}, Hidden size {hidden_size}, learning rate {lr_rate}, dropout {dropout_rate}=> Train Accuracy:{epoch_accuracy:.2f} :: Test Accuracy: {test_accuracy:.2f}%")
        return model, test_accuracy, epoch_accuracy


    

    