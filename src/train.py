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


TMSTAMP = datetime.now().strftime("%m%d_%H%M")

args = argparse.ArgumentParser()
args.add_argument("--fold", "-f", type=int, default=0)
args.add_argument("--gpu", "-g", type=int, default=0)
args.add_argument("--batch", "-b", type=int, default=512)
args = args.parse_args()


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
train_idx, val_idx = list(kf.split(train_idxs))[args.fold]


train_dataset = data.TensorDataset(
    th.tensor(df_train.iloc[train_idx].values).float(),
    th.tensor(y_train.iloc[train_idx].values).float(),
)
val_dataset = data.TensorDataset(
    th.tensor(df_train.iloc[val_idx].values).float(),
    th.tensor(y_train.iloc[val_idx].values).float(),
)
test_dataset = data.TensorDataset(
    th.tensor(df_train.iloc[X_test].values).float(),
    th.tensor(y_train.iloc[X_test].values).float(),
)
train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12
)
val_loader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=12
)
test_loader = data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=12
)

experiment_name = f"NET_f{args.fold}"
logger = wandb_logger.WandbLogger(
    project="BM-PROJECT",
    name=TMSTAMP,
)
logger.experiment.config.update(
    {
        "tmstamp": TMSTAMP,
        "batch_size": batch_size,
        "fold": args.fold,
        "clip": clip,
    }
)
model_param = {
    "input_size": n_features,
    "output_size": 1,
    "hidden_sizes": [
        200,
        100,
        50,
    ],
    "act": "LeakyReLU",
    "lr": 1e-4,
    "loss_fn": "mse",
    "dropout_rate": 0.2,
    "weight_decay": 2e-3,
}


model = Net(**model_param)
experiment_name = TMSTAMP

checkpoint_callback = ModelCheckpoint(
    monitor="val_ic",
    mode="max",
    dirpath=f"./checkpoints/{TMSTAMP}/{args.fold}/",
    filename="{epoch}-{val_ic:.2f}-{train_ic:.2f}",
    save_top_k=1,
)

earlystop_callback = EarlyStopping(
    monitor="val_ic",
    mode="max",
    patience=5,
)

gas_callback = GradientAccumulationScheduler(scheduling={10: 2, 20: 3, 30: 4, 40: 10})

logger.experiment.config.update(model_param)

trainer = pl.Trainer(
    devices=[args.gpu],
    callbacks=[checkpoint_callback, earlystop_callback, gas_callback],
    logger=logger,
    max_epochs=100,
    precision="16-mixed",
    # fast_dev_run=True,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(dataloaders=test_loader, ckpt_path="best")
