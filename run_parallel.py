import os
import itertools
from subprocess import Popen

# 配置
window_sizes = [5, 10, 20]
models = ["gru", "lstm", "transformer"]
devices = [0, 1]  # 可用 GPU 编号

# 通用参数
data_path = "database/npdc_with_trend.csv"
output_dir = "out_put/single_gan"
batch_size = 64
num_epochs = 1024
learning_rate = 3e-5
train_split = 0.8
random_seed = 3407

os.makedirs(output_dir, exist_ok=True)

# 所有实验组合
experiments = list(itertools.product(window_sizes, models))

# 跑一个实验，分配到对应 GPU
def run_experiment(i, window_size, model):
    device = devices[i % len(devices)]  # 轮流分配 GPU
    #output_path = os.path.join(output_dir, f"{model}_{window_size}.csv")

    cmd = [
        "python", "run_all_single_gan.py",  # 你的主脚本名
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--window_size", str(window_size),
        "--model", model,
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate),
        "--train_split", str(train_split),
        "--random_seed", str(random_seed),
        "--device", str(device)
    ]

    print(f"Launching on GPU:{device} → {model}_{window_size}")
    return Popen(cmd)

# 并发控制：每批最多 len(devices) 个
processes = []
for i, (window_size, model) in enumerate(experiments):
    p = run_experiment(i, window_size, model)
    processes.append(p)

    if (i + 1) % len(devices) == 0:
        for p in processes:
            p.wait()
        processes = []

# 等待最后一批完成
for p in processes:
    p.wait()

print("✅ 所有实验已完成！")
