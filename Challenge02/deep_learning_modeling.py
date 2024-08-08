# CNN-LSTM

# library import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import tqdm
import os

class ElecData(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X
        self.y = y
        self.window_size = window_size
        
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        input_window = self.X[idx:idx+self.window_size]
        target = self.y[idx+self.window_size]
        
        return input_window.astype(np.float32), target.astype(np.float32)

# model - CNN-LSTM
# modeling
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=24,
                                kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=24, hidden_size=64,
                             num_layers=5, batch_first=True)
        self.linear1 = nn.Linear(832, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = torch.flatten(x)
        return x

# Dataloader
# datapreprocessing
# minmax norm
def minmax(X, maximum=None, minimum=None):
    if maximum:
        return (X - minimum) / (maximum - minimum), maximum, minimum
    else:
        maximum = X.max(axis=0)
        minimum = X.min(axis=0)
        return (X - minimum) / (maximum - minimum), maximum, minimum

# dataset
columns = pd.read_csv("./2022_train.csv").columns
dataset1 = pd.read_csv("./2022_train.csv").values.astype(float)
dataset2 = pd.read_csv("./2023_train.csv").values[:1000].astype(float)
total_dataset = np.concatenate((dataset1, dataset2))
total_dataset = np.clip(total_dataset, 0, None)  # 발전량은 음수가 될 수 없기에 조정

norm_dataset, maximum, minimum = minmax(total_dataset)
with open("gen_min.pickle","wb") as f:
    pickle.dump(minimum, f)

with open("gen_max.pickle", "wb") as f:
    pickle.dump(maximum, f)

# training
for i in range(14, 30):

    X = norm_dataset[:, 1:14]
    y = norm_dataset[:, i]
    
    train_X = X[:-300]
    train_y = y[:-300]

    test_X = X[-300:]
    test_y = y[-300:]

    # dataloader assgin
    trainset = ElecData(train_X, train_y, 24)
    testset = ElecData(test_X, test_y, 24)

    train_loader = DataLoader(trainset, shuffle=False, batch_size=64, drop_last=True)
    test_loader = DataLoader(testset, shuffle=False, batch_size=64, drop_last=True)

    print(f"[{columns[i]}]")

    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(200):
        # sum_loss = 0//////
        for data, label in train_loader:
            data = data.to(device); label = label.to(device)
            
            optimizer.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, label)
            # sum_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        # print(loss)
    
    # evaluation
    total_loss = 0
    total_prediction = []
    total_label = []

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            total_prediction.extend(list(pred))
            total_label.extend(list(label))

            loss = F.mse_loss(pred, label)
            total_loss += loss
        
        print(total_loss.item() / len(test_loader)) # 정규화 된 상태의 값
        
        plt.figure(figsize=(10, 5))
        plt.title(f"MSE: {total_loss.item() / len(test_loader)}")
        plt.plot([x.item() for x in total_prediction], label="pred")
        plt.plot([x.item() for x in total_label], label="target")
        plt.legend()
        plt.savefig(f"./model_result/cnnlstm/cnn-lstm_model_{columns[i]}.pdf")
        plt.close()

    torch.save(model.state_dict(), f"./model_result/cnnlstm/cnn-lstm_model_{columns[i]}.pt")
