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


def train_gan(train_x_slide, train_y_gan, val_x_slide, val_y_slide, y_scaler, window_size, learning_rate, batch_size,
              num_epochs, modelG, modelD, device, patience=50):
    trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), batch_size=batch_size, shuffle=True)

    modelG = modelG.to(device)
    modelD = modelD.to(device)
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(modelG.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    optimizerD = optim.Adam(modelD.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    histG = np.zeros(num_epochs)
    histD = np.zeros(num_epochs)
    hist_val_loss = np.zeros(num_epochs)

    best_mse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0  # Early stopping counter
    print('Start training on', device)
    feature_num=train_x_slide.shape[2]
    target_num=train_y_gan.shape[2]
    print("feature numbers:",feature_num)
    print("target numbers:",target_num)
    for epoch in range(num_epochs):

        lossdata_G, lossdata_D = [], []
        for x, y in trainDataloader:
            x, y = x.to(device), y.to(device)

            # 训练判别器
            fake_data_G = torch.cat([y[:, :window_size, :], modelG(x).reshape(-1, 1, target_num)], axis=1)
            #print(y.shape)
            lossD_real = criterion(modelD(y), torch.ones_like(modelD(y)).to(device))
            lossD_fake = criterion(modelD(fake_data_G), torch.zeros_like(modelD(y)).to(device))
            loss_D = lossD_real + lossD_fake

            modelD.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizerD.step()
            lossdata_D.append(loss_D.item())

            # 训练生成器
            output_fake = modelD(fake_data_G)
            loss_G = criterion(output_fake, torch.ones_like(output_fake).to(device))
            loss_G += F.mse_loss(modelG(x).squeeze(), y[:, -1, :].squeeze())

            modelG.zero_grad()
            loss_G.backward()
            optimizerG.step()
            lossdata_G.append(loss_G.item())

        histG[epoch] = np.mean(lossdata_G)
        histD[epoch] = np.mean(lossdata_D)

        modelG.eval()
        pred_y_val = modelG(val_x_slide.to(device)).cpu().detach().numpy()
        y_val_pred = y_scaler.inverse_transform(pred_y_val)
        y_val_true = y_scaler.inverse_transform(val_y_slide)
        mse_val_g = mean_squared_error(y_val_true, y_val_pred)
        hist_val_loss[epoch] = mse_val_g
        print(
            f'Epoch {epoch + 1}/{num_epochs} | Generator Loss: {histG[epoch]:.4f} | Discriminator Loss: {histD[epoch]:.4f} | Validation MSE: {hist_val_loss[epoch]:.4f}')

        # Early stopping logic
        if mse_val_g < best_mse:
            best_mse = mse_val_g
            best_model_state = copy.deepcopy(modelG.state_dict())
            epochs_without_improvement = 0  # Reset counter if validation loss improves
        else:
            epochs_without_improvement += 1

        # If validation loss does not improve for `patience` epochs, stop early
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} with patience {patience} epochs without improvement.")
            break

        modelG.train()

    return best_model_state, histG, histD, hist_val_loss

