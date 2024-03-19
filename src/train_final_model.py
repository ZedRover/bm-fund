from typing import *
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
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
from glob import glob
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


df_train = pd.read_feather("../data/Xtrain.feather").values
y_train = pd.read_feather("../data/Ytrain.feather").values.ravel()

df_test = pd.read_feather("../data/Xtest.feather").values

# df_train, df_test, y_train, y_test = train_test_split(
#     df_train, y_train, test_size=0.2, shuffle=False
# )


def train_gbm_model(config_path: str) -> np.ndarray:
    param = json.load(open(config_path))
    clip = param["clip"]
    del param["clip"]
    x_tr, x_va, y_tr, y_va = train_test_split(
        df_train, y_train, test_size=0.2, shuffle=True
    )
    y_tr = np.clip(y_tr, -clip, clip)
    train_data = lgb.Dataset(x_tr, label=y_tr)
    valid_data = lgb.Dataset(x_va, label=y_va)
    raw_param = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_threads": 60,
    }
    param.update(raw_param)
    print(param)
    gbm = lgb.train(
        param,
        train_data,
        valid_sets=[valid_data],
        verbose_eval=False,
        early_stopping_rounds=6000,
    )
    y_pred = gbm.predict(df_test)
    # y_true = y_test
    # ic = spearmanr(y_true, y_pred).correlation
    # r2 = r2_score(y_true, y_pred)
    # print(f"ic: {ic}, r2: {r2}")

    return y_pred


def train_net(config_dir: str) -> np.ndarray:
    param = json.load(open(config_dir))
    clip = param["clip"]
    del param["clip"]
    x_tr, x_va, y_tr, y_va = train_test_split(
        df_train, y_train, test_size=0.2, shuffle=True
    )
    y_tr = np.clip(y_tr, -clip, clip)
    train_dataset = data.TensorDataset(th.tensor(x_tr).float(), th.tensor(y_tr).float())
    val_dataset = data.TensorDataset(th.tensor(x_va).float(), th.tensor(y_va).float())
    test_dataset = data.TensorDataset(
        th.tensor(df_test).float(), th.tensor(df_test).float()
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=12
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=12
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=12
    )

    TMSTAMP = datetime.now().strftime("%m%d_%H%M")
    logger = wandb_logger.WandbLogger(
        project="BM-PROJECT-FINAL",
        name=TMSTAMP,
    )
    logger.experiment.config.update(
        {
            "tmstamp": TMSTAMP,
            "batch_size": 512,
            "clip": clip,
        }
    )
    model_param = {
        "input_size": 100,
        "output_size": 1,
        "hidden_sizes": param["hidden_sizes"],
        "act": param["act"],
        "lr": param["lr"],
        "loss_fn": param["loss_fn"],
        "dropout_rate": param["dropout_rate"],
        "weight_decay": param["weight_decay"],
    }
    logger.experiment.config.update(model_param)

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
    trainer = pl.Trainer(
        max_epochs=100,  # 减少为了演示
        callbacks=[checkpoint_callback, earlystop_callback],
        devices=[3],  # 适配GPU可用性
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model.eval()
    y_pred = []
    for batch in test_loader:
        x, _ = batch
        y_pred.append(model(x).detach().cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    wandb.finish()
    # y_true = y_test
    # ic = spearmanr(y_true, y_pred).correlation
    # r2 = r2_score(y_true, y_pred)
    # print(f"ic: {ic}, r2: {r2}")
    return y_pred


if __name__ == "__main__":
    lgb_dir = "../notebook/opt_jsons"
    nn_dir = "../notebook/opt_nn_jsons"
    lgb_config_dirs = glob(f"{lgb_dir}/*.json")
    nn_config_dirs = glob(f"{nn_dir}/*.json")
    y_hats = []
    for config_dir in lgb_config_dirs:
        y_hats.append(train_gbm_model(config_dir))

    for config_dir in nn_config_dirs:
        y_hats.append(train_net(config_dir))

    y_hats = [y_hat.reshape(-1, 1) for y_hat in y_hats]
    y_pred = np.mean(y_hats, axis=0)
    # y_true = y_test.reshape(-1, 1)
    # ic = spearmanr(y_true, y_pred).correlation
    # r2 = r2_score(y_true, y_pred)
    # print(f"ic: {ic}, r2: {r2}")
    np.save("../data/y_hats.npy", y_pred)
