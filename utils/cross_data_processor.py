import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler



def load_data_all(data_path,target_columns, feature_columns,train_split):
    data = pd.read_csv(data_path)

    # 选择多个目标列
    y = data.iloc[:,target_columns].values  # 获取多个目标字段
    # 获取目标列名称
    target_column_names = data.columns[target_columns]
    print("target:",target_column_names)

    # Feature selection
    x = data.iloc[:,feature_columns].values
    feature_column_names = data.columns[feature_columns]
    print("features:", feature_column_names)

    # Data splitting
    train_size = int(data.shape[0] * train_split)

    train_x, test_x = x[:train_size], x[train_size:]

    train_y, test_y = y[:train_size], y[train_size:]

    # Normalization
    x_scaler = MinMaxScaler(feature_range=(0, 1))


    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_x = x_scaler.fit_transform(train_x)

    test_x= x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y)

    test_y = y_scaler.transform(test_y)

    return train_x, test_x,train_y, test_y, y_scaler




def create_sequences_combine(x, y, window_size,start):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(start, x.shape[0]):
        tmp_x = x[i - window_size: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window_size: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan

