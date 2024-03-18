import sys

sys.path.append("../")
import optuna
import torch as th
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from models.net import Net  # 确保你的模型定义可以被正确导入
from datetime import datetime
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import wandb as wandb_logger
import wandb

n_features = 100


def objective(trial):
    TMSTAMP = datetime.now().strftime("%m%d_%H%M")
    logger = wandb_logger.WandbLogger(
        project="BM-PROJECT",
        name=TMSTAMP,
    )
    df_train = pd.read_feather("../data/Xtrain.feather")
    y_insample = pd.read_feather("../data/Ytrain.feather")

    clip = trial.suggest_int("clip", 0, 10)
    y_insample = np.clip(y_insample, -clip, clip)

    n_samples, n_features = df_train.shape
    n_train_test_split = int(n_samples * 0.8)  # 计算 80% 的位置
    X_train_val = np.arange(n_train_test_split)  # 训练+验证集索引
    X_test_idx = np.arange(n_train_test_split, n_samples)  # 测试集索引

    X_train = df_train.iloc[X_train_val]
    y_train = y_insample.iloc[X_train_val]
    X_test = df_train.iloc[X_test_idx]
    y_test = y_insample.iloc[X_test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    train_set = TensorDataset(
        th.tensor(X_train.values).float(), th.tensor(y_train.values).float()
    )
    val_set = TensorDataset(
        th.tensor(X_val.values).float(), th.tensor(y_val.values).float()
    )
    test_set = TensorDataset(
        th.tensor(X_test.values).float(), th.tensor(y_test.values).float()
    )

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=12)

    # 模型参数
    model_param = {
        "input_size": n_features,
        "output_size": 1,
        "hidden_sizes": trial.suggest_categorical(
            "hidden_sizes", [[50, 50], [100, 50], [100, 100], [50, 50], [60], [80, 60]]
        ),
        "act": trial.suggest_categorical("act", ["relu", "leakyrelu", "selu", "silu"]),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "loss_fn": trial.suggest_categorical(
            "loss_fn", ["mse", "corr", "ccc", "huber"]
        ),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-3),
    }
    logger.experiment.config.update(model_param)
    logger.experiment.config.update(
        {
            "tmstamp": TMSTAMP,
            "clip": clip,
        }
    )

    model = Net(**model_param)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_ic",
        mode="max",
        dirpath=f"./checkpoints/{TMSTAMP}/",
        filename="{epoch}-{val_ic:.2f}-{train_ic:.2f}",
        save_top_k=1,
    )

    earlystop_callback = EarlyStopping(
        monitor="val_ic",
        mode="max",
        patience=5,
    )
    # 配置 PyTorch Lightning 训练器
    trainer = pl.Trainer(
        max_epochs=100,  # 减少为了演示
        callbacks=[checkpoint_callback, earlystop_callback],
        devices=[2],  # 适配GPU可用性
        logger=logger,
    )

    # 训练模型
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # 测试模型
    trainer.test(dataloaders=test_loader, ckpt_path="best", verbose=False)
    # 假设 test_result 包含了测试集的 IC 和 R2 分数
    res = trainer.callback_metrics
    wandb.finish()
    return res["test_ic"], res["test_r2"]


# 创建一个 Optuna 学习实验，目标是最大化 IC 和 R2
# study = optuna.create_study(
#     study_name="nn",
#     directions=["maximize", "maximize"],
#     storage="sqlite:///net_study.db",
# )
study = optuna.load_study(study_name="nn", storage="sqlite:///net_study.db")
study.optimize(objective, n_trials=100)  # 运行的试验次数

print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Values: IC={}, R2={}".format(trial.values[0], trial.values[1]))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
