import sys

sys.path.append("../")
import numpy as np
import torch as th
import pandas as pd
from sklearn.model_selection import KFold
import argparse
import wandb
from datetime import datetime
from pytorch_lightning.loggers import wandb as wandb_logger
from torch.utils import data
from models.net import Net
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
)

import multiprocessing

# 全局变量定义
TMSTAMP = datetime.now().strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g", type=int, default=0)
parser.add_argument("--batch", "-b", type=int, default=512)
parser.add_argument("--clip", type=int, default=5)
args = parser.parse_args()
clip = args.clip
batch_size = args.batch

df_train = pd.read_feather("../data/Xtrain.feather")
y_train = pd.read_feather("../data/Ytrain.feather")

clip = 5
y_train = np.clip(y_train, -clip, clip)

n_samples, n_features = df_train.shape
n_train_test_split = int(n_samples * 0.8)  # 计算 80% 的位置
X_train_val = np.arange(n_train_test_split)  # 训练+验证集索引
X_test = np.arange(n_train_test_split, n_samples)  # 测试集索引
kf = KFold(n_splits=4, shuffle=False)
train_idxs = np.arange(len(X_train_val))


def train_model(fold, gpu, batch_size, clip):
    train_idx, val_idx = list(kf.split(train_idxs))[fold]
    X_train, y_train, X_val, y_val = (
        df_train.iloc[train_idx].values,
        y_train.iloc[train_idx].values,
        df_train.iloc[val_idx].values,
        y_train.iloc[val_idx].values,
    )

    train_dataset = TensorDataset(
        th.tensor(X_train.values).float(), th.tensor(y_train).float()
    )
    val_dataset = TensorDataset(
        th.tensor(X_val.values).float(), th.tensor(y_val).float()
    )
    test_dataset = TensorDataset(
        th.tensor(df_train.iloc[X_test].values).float(),
        th.tensor(y_train.iloc[X_test].values).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
    )

    # 模型和训练器配置
    model_param = {
        "input_size": df_train.shape[1],
        "output_size": 1,
        "hidden_sizes": [100, 50],
        "act": "LeakyReLU",
        "lr": 1e-4,
        "loss_fn": "corr",
        "dropout_rate": 0.05,
        "weight_decay": 0.02,
    }
    model = Net(**model_param)

    logger = WandbLogger(project="BM-PROJECT", name=f"{TMSTAMP}_fold{fold}")
    logger.experiment.config.update(model_param)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{TMSTAMP}/fold{fold}/",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    trainer = pl.Trainer(
        gpus=[gpu],
        callbacks=[checkpoint_callback, earlystop_callback],
        logger=logger,
        max_epochs=100,
        precision=16,
    )

    # 训练和测试
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":

    processes = []
    for fold in range(4):  # 创建4个进程，每个进程训练一个模型
        p = multiprocessing.Process(
            target=train_model, args=(fold, fold, args.batch, args.clip)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 等待所有进程完成
