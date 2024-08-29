import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim

# Wczytanie danych z pliku CSV
data = pd.read_csv('data/F1_FULL_DATA.csv')

# Przygotowanie danych do modelowania
features = data[['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']]
target_top10 = (data['FinishPosition'] <= 10).astype(int)
target_top3 = (data['FinishPosition'] <= 3).astype(int)

# Podział danych na zbiór treningowy i testowy dla top 10
X_train, X_test, y_train_top10, y_test_top10 = train_test_split(features, target_top10, test_size=0.2, random_state=42)
# Podział danych na zbiór treningowy i testowy dla top 3
_, _, y_train_top3, y_test_top3 = train_test_split(features, target_top3, test_size=0.2, random_state=42)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konwersja danych do tensorów
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor_top10 = torch.tensor(y_train_top10.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor_top10 = torch.tensor(y_test_top10.values, dtype=torch.float32).unsqueeze(1)
y_train_tensor_top3 = torch.tensor(y_train_top3.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor_top3 = torch.tensor(y_test_top3.values, dtype=torch.float32).unsqueeze(1)

# Definicja sieci neuronowej
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return torch.sigmoid(out)

# Inicjalizacja modelu
input_size = X_train_tensor.shape[1]
model_top10 = NeuralNetwork(input_size)
model_top3 = NeuralNetwork(input_size)

# Definicja funkcji straty i optymalizatora
criterion = nn.BCELoss()
optimizer_top10 = optim.Adam(model_top10.parameters(), lr=0.001)
optimizer_top3 = optim.Adam(model_top3.parameters(), lr=0.001)

# Trenowanie modelu dla top 10
num_epochs = 100
for epoch in range(num_epochs):
    model_top10.train()
    optimizer_top10.zero_grad()
    outputs = model_top10(X_train_tensor)
    loss = criterion(outputs, y_train_tensor_top10)
    loss.backward()
    optimizer_top10.step()

# Trenowanie modelu dla top 3
for epoch in range(num_epochs):
    model_top3.train()
    optimizer_top3.zero_grad()
    outputs = model_top3(X_train_tensor)
    loss = criterion(outputs, y_train_tensor_top3)
    loss.backward()
    optimizer_top3.step()

# Ewaluacja modelu
def evaluate_model(model, X_test_tensor, y_test_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs > threshold).float()
        accuracy = accuracy_score(y_test_tensor, predictions)
        precision = precision_score(y_test_tensor, predictions)
        recall = recall_score(y_test_tensor, predictions)
        f1 = f1_score(y_test_tensor, predictions)
    return accuracy, precision, recall, f1

# Ocena modelu dla top 10
accuracy_top10, precision_top10, recall_top10, f1_top10 = evaluate_model(model_top10, X_test_tensor, y_test_tensor_top10)

# Ocena modelu dla top 3
accuracy_top3, precision_top3, recall_top3, f1_top3 = evaluate_model(model_top3, X_test_tensor, y_test_tensor_top3)

print(f"Top 10 - Accuracy: {accuracy_top10}, Precision: {precision_top10}, Recall: {recall_top10}, F1 Score: {f1_top10}")
print(f"Top 3 - Accuracy: {accuracy_top3}, Precision: {precision_top3}, Recall: {recall_top3}, F1 Score: {f1_top3}")
