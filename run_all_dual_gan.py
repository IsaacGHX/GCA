import os
import pandas as pd
import argparse
from itertools import combinations
from dual_gan import *  # 改为你新写的dual_gan.py文件

def run_dual_experiments(data_path, output_dir,
                         window_size1, window_size2, distill,
                         num_epoch, batch_size, train_split,
                         random_seed, device):

    results_file = os.path.join(output_dir, "gca_dual_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_file):
        df = pd.DataFrame(columns=[
            "feature_columns", "target_columns",
            "train_mse", "train_mae", "train_rmse", "train_mape",
            "test_mse", "test_mae", "test_rmse", "test_mape"
        ])
        df.to_csv(results_file, index=False)

    feature_columns = list(range(22, 35))
    target_columns = [[i] for i in range(1, 22)]

    for target in target_columns:
        target_feature_columns = feature_columns + target
        print("Using features:", target_feature_columns)

        results = dual_gan_pipeline(
            data_path=data_path,
            output_dir=output_dir,
            feature_columns=target_feature_columns,
            target_columns=target,
            window_size1=window_size1,
            window_size2=window_size2,
            model_shorter_term=args.model_shorter_term,
            model_longer_term=args.model_longer_term,
            distill=distill,
            num_epoch=num_epoch,
            batch_size=batch_size,
            train_split=train_split,
            random_seed=random_seed,
            device=device
        )

        result_row = {
            "feature_columns": feature_columns,
            "target_columns": target,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "train_rmse": results["train_rmse"],
            "train_mape": results["train_mape"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "test_rmse": results["test_rmse"],
            "test_mape": results["test_mape"]
        }

        df = pd.DataFrame([result_row])
        df.to_csv(results_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for dual GAN model")
    parser.add_argument('--data_path', type=str, default="database/npdc_with_trend.csv")
    parser.add_argument('--output_dir', type=str, default="out_put/dual_gan")
    parser.add_argument('--window_size1', type=int, default=5)
    parser.add_argument('--window_size2', type=int, default=10)
    parser.add_argument('--model_shorter_term', type=str, default="gru")
    parser.add_argument('--model_longer_term', type=str, default="lstm")

    parser.add_argument('--distill', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=3407)

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    run_dual_experiments(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_size1=args.window_size1,
        window_size2=args.window_size2,
        model_shorter_term=args.model_shorter_term,
        model_longer_term=args.model_longer_term,
        distill=args.distill,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        train_split=args.train_split,
        random_seed=args.random_seed,
        device=args.device
    )