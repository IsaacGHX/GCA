from sklearn.preprocessing import MinMaxScaler
import math
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import copy
import matplotlib.pyplot as plt
import numpy as np
import os

def train_generator(train_x_slide, train_y_slide, val_x_slide, val_y_slide, y_scaler,
                    learning_rate, batch_size, num_epochs, modelG, device, patience=50):

    trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_slide), batch_size=batch_size, shuffle=True)

    modelG = modelG.to(device)
    optimizerG = optim.Adam(modelG.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    histG = np.zeros(num_epochs)
    hist_val_loss = np.zeros(num_epochs)

    best_mse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0  # Early stopping counter
    print('Start training Generator on', device)

    for epoch in range(num_epochs):
        lossdata_G = []

        modelG.train()
        for x, y in trainDataloader:
            x, y = x.to(device), y.to(device)

            pred = modelG(x).squeeze()
            loss_G = criterion(pred, y.squeeze())

            optimizerG.zero_grad()
            loss_G.backward()
            optimizerG.step()
            lossdata_G.append(loss_G.item())

        histG[epoch] = np.mean(lossdata_G)

        # Evaluate on validation set
        modelG.eval()
        with torch.no_grad():
            pred_y_val = modelG(val_x_slide.to(device)).cpu().numpy()
            y_val_pred = y_scaler.inverse_transform(pred_y_val)
            y_val_true = y_scaler.inverse_transform(val_y_slide)
            mse_val = mean_squared_error(y_val_true, y_val_pred)
            hist_val_loss[epoch] = mse_val

        print(f'Epoch {epoch + 1}/{num_epochs} | Generator Loss: {histG[epoch]:.4f} | Validation MSE: {mse_val:.4f}')

        # Early stopping
        if mse_val < best_mse:
            best_mse = mse_val
            best_model_state = copy.deepcopy(modelG.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} with patience {patience} epochs without improvement.")
            break

    return best_model_state, histG, hist_val_loss
