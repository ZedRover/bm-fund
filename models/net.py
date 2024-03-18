import functools
from typing import Any, Dict, List, Tuple

import torch as th
import pytorch_lightning as pl
from lion_pytorch import Lion
from torch import Tensor, nn, optim
from audtorch.metrics.functional import pearsonr as pearsonr_
from torcheval.metrics.functional import r2_score


def nanstd(input_tensor: Tensor, dim: int = 0, keepdim: bool = True) -> Tensor:
    return th.sqrt(
        th.nanmean(
            (input_tensor - th.nanmean(input_tensor, dim=dim, keepdim=keepdim)) ** 2
        )
    )


class CorrLoss(nn.Module):
    """Pearsonr correlation loss."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, real: Tensor) -> Tensor:
        mean_y, std_y = th.nanmean(real), nanstd(real)
        mean_yhat, std_yhat = th.nanmean(pred), nanstd(pred)
        if std_y == 0 or std_yhat == 0:
            print("std_y or std_yhat is zero")
            return th.tensor([0.0], device=pred.device)
        return 1 - th.mean((pred - mean_yhat) * (real - mean_y) / (std_yhat * std_y))


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient Loss using nanmean."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Calculate means using nanmean
        mean_input, mean_target = th.nanmean(input), th.nanmean(target)
        # Calculate variances using nanmean
        var_input = th.nanmean((input - mean_input) ** 2)
        var_target = th.nanmean((target - mean_target) ** 2)
        # Calculate Pearson correlation coefficient using audtorch's function
        pearson_r = pearsonr_(input, target, batch_first=False)
        # Calculate the Concordance Correlation Coefficient (CCC)
        ccc = (2 * pearson_r * th.sqrt(var_input) * th.sqrt(var_target)) / (
            var_input + var_target + (mean_input - mean_target) ** 2
        )
        # Return 1 minus the CCC as the loss to be minimized
        return 1 - ccc


class R2Loss(nn.Module):
    """R2 score loss."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, real: Tensor) -> Tensor:
        return 1 - r2_score(pred, real)


def pearsonr(yhat: Tensor, y: Tensor, batch_first=False) -> Tensor:
    yhat = yhat.squeeze()
    y = y.squeeze()
    idx = ~th.isnan(y)
    yhat = yhat[idx]
    y = y[idx]
    return pearsonr_(yhat, y, batch_first=batch_first)


def metric_callback(yhat: Tensor, y: Tensor, stage: str = "train"):
    ic = pearsonr(yhat, y, batch_first=False)
    r2 = r2_score(yhat, y)
    return {
        f"{stage}_ic": ic,
        f"{stage}_r2": r2,
    }


loss_fn_dict: Dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "corr": CorrLoss(),
    "huber": nn.HuberLoss(delta=1),
    "ccc": CCCLoss(),
    "r2": R2Loss(),
}

act_fn_dict: Dict[str, nn.Module] = {
    "leakyrelu": nn.LeakyReLU(negative_slope=0.1),
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "silu": nn.SiLU(),
}


class Net(pl.LightningModule):
    model_name = "Net"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int] = (256,),
        act: str = "LeakyReLU",
        lr: float = 1e-3,
        loss_fn: str = "mse",
        dropout_rate: float = -1,
        weight_decay: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()

        self.kwargs = kwargs
        self.loss_fn = loss_fn
        self.loss = loss_fn_dict.get(loss_fn.lower(), CorrLoss())
        self.act = act_fn_dict.get(act.lower(), nn.LeakyReLU(negative_slope=0.1))
        self.lr = lr
        self.weight_decay = weight_decay

        hidden_sizes = [input_size] + list(hidden_sizes)
        dnn_layers = []
        drop_input = nn.Dropout(dropout_rate)
        dnn_layers.append(drop_input)
        hidden_units = input_size
        for _i, (_input_size, hidden_units) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:], strict=True)
        ):
            fc = nn.Linear(_input_size, hidden_units)
            bn = nn.BatchNorm1d(hidden_units)
            dnn_layers.append(nn.Sequential(fc, bn, self.act))
        drop_output = nn.Dropout(dropout_rate)
        dnn_layers.append(drop_output)
        dnn_layers.append(nn.Linear(hidden_units, output_size))
        self.dnn_layers = nn.ModuleList(dnn_layers)

        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )

    def configure_optimizers(self) -> Any:
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, x):
        cur_output = x
        for _, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output

    def _get_reconstruction_loss(self, yhat: Tensor, y: Tensor) -> Tensor:
        return self.loss(yhat, y)

    def training_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        yhat = self(x)
        loss = self._get_reconstruction_loss(yhat, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log_dict(metric_callback(yhat, y, "train"))
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        yhat = self(x)
        loss = self._get_reconstruction_loss(yhat, y)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log_dict(metric_callback(yhat, y, "val"))
        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        yhat = self(x)
        self.log_dict(metric_callback(yhat, y, "test"))
        return None
