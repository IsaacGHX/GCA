import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import matplotlib.pyplot as plt

def validate(model, val_x, val_y):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # 使用模型进行预测
        predictions = model(val_x).cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())
        return mse_loss


def plot_generator_losses(data_G1, data_G2, data_G3,output_dir):
    """
    绘制 G1、G2、G3 的损失曲线。

    Args:
        data_G1 (list): G1 的损失数据列表，包含 [histD1_G1, histD2_G1, histD3_G1, histG1]。
        data_G2 (list): G2 的损失数据列表，包含 [histD1_G2, histD2_G2, histD3_G2, histG2]。
        data_G3 (list): G3 的损失数据列表，包含 [histD1_G3, histD2_G3, histD3_G3, histG3]。
    """

    all_data = [data_G1, data_G2, data_G3]

    plt.figure(figsize=(15, 5))  # 可选：设置图形大小

    # 循环绘制 G1、G2、G3 的损失曲线
    for i, data in enumerate(all_data):
        plt.subplot(1, 3, i + 1)  # 创建子图
        plt.plot(data[0], label=f"G{i+1} against D1 Loss")
        plt.plot(data[1], label=f"G{i+1} against D2 Loss")
        plt.plot(data[2], label=f"G{i+1} against D3 Loss")
        plt.plot(data[3], label=f"combined G{i+1} Loss")

        plt.xlabel("Epoch")
        plt.ylabel(f"G{i+1} Loss")
        plt.title(f"G{i+1} Loss over Epochs")
        plt.legend()

    # 如果需要显示整个图形，可以添加 plt.show()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"))


def plot_discriminator_losses(data_D1, data_D2, data_D3,output_dir):

    all_data = [data_D1, data_D2, data_D3]

    plt.figure(figsize=(15, 5))  # 可选：设置图形大小

    # 循环绘制 G1、G2、G3 的损失曲线
    for i, data in enumerate(all_data):
        plt.subplot(1, 3, i + 1)  # 创建子图
        plt.plot(data[0], label=f"D{i+1} against G1 Loss")
        plt.plot(data[1], label=f"D{i+1} against G2 Loss")
        plt.plot(data[2], label=f"D{i+1} against G3 Loss")
        plt.plot(data[3], label=f"combined D{i+1} Loss")

        plt.xlabel("Epoch")
        plt.ylabel(f"D{i+1} Loss")
        plt.title(f"D{i+1} Loss over Epochs")
        plt.legend()

    # 如果需要显示整个图形，可以添加 plt.show()
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"))



def visualize_overall_loss(histG1, histG2, histG3,histD1, histD2, histD3,output_dir):

    plt.figure(figsize=(12, 6))  # 可选：设置图形大小
    plt.plot(histG1, label="G1 Loss")
    plt.plot(histG2, label="G2 Loss")
    plt.plot(histG3, label="G3 Loss")
    plt.plot(histD1, label="D1 Loss")
    plt.plot(histD2, label="D2 Loss")
    plt.plot(histD3, label="D3 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_losses.png"))




def plot_mse_loss(hist_MSE_G1, hist_MSE_G2, hist_MSE_G3, hist_val_loss1, hist_val_loss2, hist_val_loss3, num_epochs,output_dir):
    """
    绘制训练过程中和验证集上的MSE损失变化曲线

    参数：
    hist_MSE_G1, hist_MSE_G2, hist_MSE_G3 : 训练过程中各生成器的MSE损失
    hist_val_loss1, hist_val_loss2, hist_val_loss3 : 验证集上各生成器的MSE损失
    num_epochs : 训练的epoch数
    """
    plt.figure(figsize=(12, 6))

    # 绘制训练集MSE损失曲线
    plt.plot(range(num_epochs), hist_MSE_G1, label="Train MSE G1", color='blue')
    plt.plot(range(num_epochs), hist_MSE_G2, label="Train MSE G2", color='green')
    plt.plot(range(num_epochs), hist_MSE_G3, label="Train MSE G3", color='red')

    # 绘制验证集MSE损失曲线
    plt.plot(range(num_epochs), hist_val_loss1, label="Val MSE G1", color='blue', alpha=0.5)
    plt.plot(range(num_epochs), hist_val_loss2, label="Val MSE G2", color='green', alpha=0.5)
    plt.plot(range(num_epochs), hist_val_loss3, label="Val MSE G3", color='red', alpha=0.5)

    plt.title("MSE Loss for Generators (Train and Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(os.path.join(output_dir, "mse_losses.png"))





def inverse_transform(predictions, scaler):
    """ 使用y_scaler逆转换预测结果 """
    return scaler.inverse_transform(predictions)

def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)  # 新增
    return mse, mae, rmse, mape, per_target_mse

def plot_fitting_curve(true_values, predicted_values, output_dir, model_name):
    """绘制拟合曲线并保存结果"""
    plt.figure(figsize=(10,6))
    plt.plot(true_values, label='True Values')
    plt.plot(predicted_values, label='Predicted Values')
    plt.title(f'{model_name} Fitting Curve')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{output_dir}/{model_name}_fitting_curve.png')
    plt.close()

def save_metrics(metrics, output_dir, model_name):
    """保存MSE, MAE, RMSE, MAPE到文件"""
    with open(f'{output_dir}/{model_name}_metrics.txt', 'w') as f:
        f.write("MSE: {}\n".format(metrics[0]))
        f.write("MAE: {}\n".format(metrics[1]))
        f.write("RMSE: {}\n".format(metrics[2]))
        f.write("MAPE: {}\n".format(metrics[3]))

# 在训练结束后，加载最佳模型并进行预测和评估
def evaluate_best_models(modelG1, modelG2, modelG3, best_model_state1, best_model_state2, best_model_state3, train_x1, train_x2, train_x3, train_y,  test_x1, test_x2, test_x3, test_y, y_scaler, output_dir):
    # 加载最佳模型权重
    modelG1.load_state_dict(best_model_state1)
    modelG2.load_state_dict(best_model_state2)
    modelG3.load_state_dict(best_model_state3)

    # 将模型设置为评估模式
    modelG1.eval()
    modelG2.eval()
    modelG3.eval()


    train_y=inverse_transform(train_y, y_scaler)
    test_y=inverse_transform(test_y, y_scaler)
    # 在训练集上进行预测
    with torch.no_grad():
        train_pred1 = modelG1(train_x1).cpu().numpy()
        train_pred2 = modelG2(train_x2).cpu().numpy()
        train_pred3 = modelG3(train_x3).cpu().numpy()

    # 逆转换到原始数据
    train_pred1_inv = inverse_transform(train_pred1, y_scaler)
    train_pred2_inv = inverse_transform(train_pred2, y_scaler)
    train_pred3_inv = inverse_transform(train_pred3, y_scaler)

    # 计算训练集指标
    train_metrics1 = compute_metrics(train_y, train_pred1_inv)
    train_metrics2 = compute_metrics(train_y, train_pred2_inv)
    train_metrics3 = compute_metrics(train_y, train_pred3_inv)

    # 输出训练集指标
    print(
        f"Train Metrics for G1: MSE={train_metrics1[0]:.4f}, MAE={train_metrics1[1]:.4f}, RMSE={train_metrics1[2]:.4f}, MAPE={train_metrics1[3]:.4f}")
    print(
        f"Train Metrics for G2: MSE={train_metrics2[0]:.4f}, MAE={train_metrics2[1]:.4f}, RMSE={train_metrics2[2]:.4f}, MAPE={train_metrics2[3]:.4f}")
    print(
        f"Train Metrics for G3: MSE={train_metrics3[0]:.4f}, MAE={train_metrics3[1]:.4f}, RMSE={train_metrics3[2]:.4f}, MAPE={train_metrics3[3]:.4f}")

    # 绘制训练集拟合曲线
    plot_fitting_curve(train_y, train_pred1_inv, output_dir, 'G1_Train')
    plot_fitting_curve(train_y, train_pred2_inv, output_dir, 'G2_Train')
    plot_fitting_curve(train_y, train_pred3_inv, output_dir, 'G3_Train')

    # 保存训练集结果
    # save_metrics(train_metrics1, output_dir, 'G1_Train')
    # save_metrics(train_metrics2, output_dir, 'G2_Train')
    # save_metrics(train_metrics3, output_dir, 'G3_Train')


    # 在测试集上进行预测
    with torch.no_grad():
        test_pred1 = modelG1(test_x1).cpu().numpy()
        test_pred2 = modelG2(test_x2).cpu().numpy()
        test_pred3 = modelG3(test_x3).cpu().numpy()

    # 逆转换到原始数据
    test_pred1_inv = inverse_transform(test_pred1, y_scaler)
    test_pred2_inv = inverse_transform(test_pred2, y_scaler)
    test_pred3_inv = inverse_transform(test_pred3, y_scaler)

    # 计算测试集指标
    test_metrics1 = compute_metrics(test_y, test_pred1_inv)
    test_metrics2 = compute_metrics(test_y, test_pred2_inv)
    test_metrics3 = compute_metrics(test_y, test_pred3_inv)

    # 输出测试集指标
    print(
        f"Test Metrics for G1: MSE={test_metrics1[0]:.4f}, MAE={test_metrics1[1]:.4f}, RMSE={test_metrics1[2]:.4f}, MAPE={test_metrics1[3]:.4f}")
    print(
        f"Test Metrics for G2: MSE={test_metrics2[0]:.4f}, MAE={test_metrics2[1]:.4f}, RMSE={test_metrics2[2]:.4f}, MAPE={test_metrics2[3]:.4f}")
    print(
        f"Test Metrics for G3: MSE={test_metrics3[0]:.4f}, MAE={test_metrics3[1]:.4f}, RMSE={test_metrics3[2]:.4f}, MAPE={test_metrics3[3]:.4f}")

    # 绘制测试集拟合曲线
    plot_fitting_curve(test_y, test_pred1_inv, output_dir, 'G1_Test')
    plot_fitting_curve(test_y, test_pred2_inv, output_dir, 'G2_Test')
    plot_fitting_curve(test_y, test_pred3_inv, output_dir, 'G3_Test')

    # 保存测试集结果
    # save_metrics(test_metrics1, output_dir, 'G1_Test')
    # save_metrics(test_metrics2, output_dir, 'G2_Test')
    # save_metrics(test_metrics3, output_dir, 'G3_Test')
    return {
        "train_mse": [train_metrics1[0], train_metrics2[0], train_metrics3[0]],
        "train_mae": [train_metrics1[1], train_metrics2[1], train_metrics3[1]],
        "train_rmse": [train_metrics1[2], train_metrics2[2], train_metrics3[2]],
        "train_mape": [train_metrics1[3], train_metrics2[3], train_metrics3[3]],
        "train_mse_per_target": [train_metrics1[4], train_metrics2[4], train_metrics3[4]],
        "test_mse": [test_metrics1[0], test_metrics2[0], test_metrics3[0]],
        "test_mae": [test_metrics1[1], test_metrics2[1], test_metrics3[1]],
        "test_rmse": [test_metrics1[2], test_metrics2[2], test_metrics3[2]],
        "test_mape": [test_metrics1[3], test_metrics2[3], test_metrics3[3]],
        "test_mse_per_target": [test_metrics1[4], test_metrics2[4], test_metrics3[4]]
    }



