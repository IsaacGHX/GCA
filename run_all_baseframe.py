import pandas as pd
import argparse
import torch
from baseframe import *  # 记得你的新 pipeline 名是 single_baseframe_pipeline
from utils.baseframe_trainer import *

def run_multiple_experiments(data_path, output_dir, window_size, model, batch_size,
                             num_epochs, learning_rate, train_split, random_seed, device):

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    #output_dir = f'{output_dir}/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running experiments on device={device}...")

    # feature_columns = list(range(22, 35))
    feature_columns=[]
    target_columns = [[i] for i in range(1, 22)]

    # 逐个组合进行实验
    for target in target_columns:
        target_feature_columns = feature_columns + target
        print("Using features:", target_feature_columns)

        results = run_baseframe_pipeline(
            data_path=data_path,
            window_size=window_size,
            model=model,
            feature_columns=target_feature_columns,
            target_column=target,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            train_split=train_split,
            random_seed=random_seed,
            output_dir=output_dir,
            device=device
        )

        result_row = {
            'target_column': target,
            'train_mse': results['train_mse'],
            'train_mae': results['train_mae'],
            'train_rmse': results['train_rmse'],
            'train_mape': results['train_mape'],
            'test_mse': results['test_mse'],
            'test_mae': results['test_mae'],
            'test_rmse': results['test_rmse'],
            'test_mape': results['test_mape']
        }
        output_path = f'{output_dir}/{model}{window_size}_without_GT.csv'
        pd.DataFrame([result_row]).to_csv(output_path, mode='a', header=not pd.io.common.file_exists(output_path), index=False)

    print(f"All experiments completed. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple experiments for baseframe generator training.')
    parser.add_argument('--data_path', type=str, default="database/npdc_with_trend.csv")
    parser.add_argument('--output_dir', type=str, default='out_put/baseframe_without_GT')  # 注意路径别再带 gan
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--model', type=str, default='gru')
    parser.add_argument('--device', type=int, default=0, help="GPU index (i for GPU i)")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=3407)

    args = parser.parse_args()

    run_multiple_experiments(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        model=args.model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        random_seed=args.random_seed,
        device=args.device
    )
