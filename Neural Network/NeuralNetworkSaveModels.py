import argparse
import logging
import os
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn import Module, Linear, ReLU, LogSoftmax
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#==================================================================================================
#                                            LOGGER
#==================================================================================================

logger = logging.getLogger()
lprint = logger.info

def setup_logger():
    log_formatter = logging.Formatter('%(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)

def print_text_separator():
    lprint('--------------------------------------------------------')

#==================================================================================================
#                                         NEURAL NETWORKS
#==================================================================================================
class RaceNet(Module):
    def __init__(self, input_size: int, classes: int):
        super(RaceNet, self).__init__()
        
        self.fc1 = Linear(in_features=input_size, out_features=128)
        self.relu1 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=64)
        self.relu2 = ReLU()
        self.fc3 = Linear(in_features=64, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output = self.logSoftmax(x)
        return output

#==================================================================================================
#                                         TRAINING WRAPPER 
#==================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Ścieżka do pliku CSV z danymi')
    parser.add_argument('-o', '--output_path', type=str, default='/models', help='Ścieżka do katalogu, gdzie zostaną zapisane wyniki (domyślnie: /models)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Rozmiar batcha (domyślnie: 32)')
    parser.add_argument('-t', '--train_split', type=float, default=0.8, help='Procent danych przeznaczonych na trening (domyślnie: 0.7)')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3, help='Początkowa szybkość uczenia (domyślnie: 1e-3)')
    parser.add_argument('--epochs', type=int, default=10, help='Liczba epok treningowych (domyślnie: 10)')
    return parser.parse_args()

def get_data_loaders(dataset_path: str, train_split: float, batch_size: int):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Encode 'Driver' column
    label_encoder = LabelEncoder()
    df['Driver'] = label_encoder.fit_transform(df['Driver'])

    # Prepare the data
    features = ['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']
    X = df[features]
    y_points = df['FinishPosition'] <= 10
    y_podium = df['FinishPosition'] <= 3

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create datasets
    tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y_points = torch.tensor(y_points.values, dtype=torch.long)
    tensor_y_podium = torch.tensor(y_podium.values, dtype=torch.long)
    
    dataset_points = TensorDataset(tensor_X, tensor_y_points)
    dataset_podium = TensorDataset(tensor_X, tensor_y_podium)
    
    # Split datasets
    train_size_points = int(len(dataset_points) * train_split)
    val_size_points = len(dataset_points) - train_size_points
    
    train_size_podium = int(len(dataset_podium) * train_split)
    val_size_podium = len(dataset_podium) - train_size_podium
    
    train_dataset_points, val_dataset_points = random_split(dataset_points, [train_size_points, val_size_points])
    train_dataset_podium, val_dataset_podium = random_split(dataset_podium, [train_size_podium, val_size_podium])
    
    # Create data loaders
    train_loader_points = DataLoader(train_dataset_points, batch_size=batch_size, shuffle=True)
    val_loader_points = DataLoader(val_dataset_points, batch_size=batch_size)
    
    train_loader_podium = DataLoader(train_dataset_podium, batch_size=batch_size, shuffle=True)
    val_loader_podium = DataLoader(val_dataset_podium, batch_size=batch_size)
    
    return train_loader_points, val_loader_points, train_loader_podium, val_loader_podium

def train_network(initial_learning_rate: float, epochs: int,
                  train_loader: DataLoader, validation_loader: DataLoader,
                  output_path: str, task: str):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    lprint(f'Device -> {device_name}')
    device = torch.device(device_name)
    input_size = next(iter(train_loader))[0].shape[1]
    model = RaceNet(input_size, 2).to(device)  # Two classes: True or False
    opt = Adam(model.parameters(), lr=initial_learning_rate)
    lossFn = torch.nn.NLLLoss()
    lprint(f'Initializing training for {task}')
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    lprint("[INFO] training the network...")
    startTime = time.time()
    # loop over our epochs
    for e in range(0, epochs):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in train_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in validation_loader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
                
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / len(train_loader)
        avgValLoss = totalValLoss / len(validation_loader)
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_loader.dataset)
        valCorrect = valCorrect / len(validation_loader.dataset)
        # update our training history
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["train_acc"].append(trainCorrect)
        history["val_loss"].append(avgValLoss.cpu().detach().numpy())
        history["val_acc"].append(valCorrect)
        # print the model training and validation information
        lprint(f"[INFO] EPOCH: {e + 1}/{epochs}")
        lprint(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
        lprint(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}\n")

    # finish measuring how long training took
    endTime = time.time()
    lprint(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save the model
    model_save_path = os.path.join(output_path, f'{task}_model.pth')
    torch.save(model.state_dict(), model_save_path)
    lprint(f"[INFO] Model saved to: {model_save_path}")
    
    # Save the training history
    history_save_path = os.path.join(output_path, f'{task}_training_history.csv')
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_save_path, index=False)
    lprint(f"[INFO] Training history saved to: {history_save_path}")
    
    return model, history

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, precision, recall, f1

def visualisation_of_history(history, task):
    plt.title(f'Accuracy - {task}')
    plt.plot(history['train_acc'], '-', label='Train')
    plt.plot(history['val_acc'], '--', label='Validation')
    plt.legend()
    plt.show()

    plt.title(f'Loss - {task}')
    plt.plot(history['train_loss'], '-', label='Train')
    plt.plot(history['val_loss'], '--', label='Validation')
    plt.legend()
    plt.show()

def main(args):
    setup_logger()
    train_loader_points, val_loader_points, train_loader_podium, val_loader_podium = get_data_loaders(args.dataset_path, args.train_split, args.batch_size)
    
    model_points, history_points = train_network(args.initial_learning_rate, args.epochs, train_loader_points, val_loader_points, args.output_path, 'points')
    visualisation_of_history(history_points, 'Points')
    
    model_podium, history_podium = train_network(args.initial_learning_rate, args.epochs, train_loader_podium, val_loader_podium, args.output_path, 'podium')
    visualisation_of_history(history_podium, 'Podium')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_points, precision_points, recall_points, f1_points = evaluate_model(model_points, val_loader_points, device)
    lprint(f"Top 10 - Accuracy: {accuracy_points}, Precision: {precision_points}, Recall: {recall_points}, F1 Score: {f1_points}")

    accuracy_podium, precision_podium, recall_podium, f1_podium = evaluate_model(model_podium, val_loader_podium, device)
    lprint(f"Top 3 - Accuracy: {accuracy_podium}, Precision: {precision_podium}, Recall: {recall_podium}, F1 Score: {f1_podium}")

if __name__ == '__main__':
    main(parse_arguments())
