from collections import deque
from time import time
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr as pearsonr_
from fast_soft_sort.pytorch_ops import soft_rank
from torch import Tensor, nn, optim

METRICS_1D_5D = [
    "test_ic_1d",
    "test_ic_5d",
    "test_q90_1d",
    "test_q90_5d",
    "test_q99_1d",
    "test_q99_5d",
]


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def nanstd(input_tensor: Tensor, dim: int = 0, keepdim: bool = True) -> Tensor:
    return th.sqrt(
        th.nanmean(
            (input_tensor - th.nanmean(input_tensor, dim=dim, keepdim=keepdim)) ** 2
        )
    )


def nanmax(tensor, dim=None, keepdim=False):
    min_value = th.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output


def nanmin(tensor, dim=None, keepdim=False):
    max_value = th.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


# def nanstd(tensor, dim=None, keepdim=False):
#     output = nanvar(tensor, dim=dim, keepdim=keepdim)
#     output = output.sqrt()
#     return output


def nanprod(tensor, dim=None, keepdim=False):
    output = tensor.nan_to_num(1).prod(dim=dim, keepdim=keepdim)
    return output


def nancumprod(tensor, dim=None, keepdim=False):
    output = tensor.nan_to_num(1).cumprod(dim=dim, keepdim=keepdim)
    return output


def nancumsum(tensor, dim=None, keepdim=False):
    output = tensor.nan_to_num(0).cumsum(dim=dim, keepdim=keepdim)
    return output


def nanargmin(tensor, dim=None, keepdim=False):
    max_value = th.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdim)
    return output


def nanargmax(tensor, dim=None, keepdim=False):
    min_value = th.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdim)
    return output


def calc_topk(input: Tensor, target: Tensor, topk: int = 10) -> Tensor:
    """Use this function must be under the condition that the batch size of test_loader is 6,000."""
    idx = ~th.isnan(target)
    input, target = input[idx], target[idx]
    top_q_indices = th.topk(input, int(len(target) * topk / 100), largest=True).indices
    q_ret = th.nanmean(target[top_q_indices]).item()
    return round(q_ret * 250, 3)


def pearsonr(yhat: Tensor, y: Tensor, batch_first=False) -> Tensor:
    yhat = yhat.squeeze()
    y = y.squeeze()
    idx = ~th.isnan(y)
    yhat = yhat[idx]
    y = y[idx]
    return pearsonr_(yhat, y, batch_first=batch_first)


# @timer_func
def metric_callback(
    input: Tensor,
    target: Tensor,
    stage: str = "train",
    name_y: List[str] | Tuple = ("1d", "5d"),
) -> Dict:
    y1, y2 = target[:, 0].squeeze(), target[:, 1].squeeze()
    if input.shape[1] == 2:
        y1_hat, y2_hat = input[:, 0].squeeze(), input[:, 1].squeeze()
    else:
        y1_hat, y2_hat = input.squeeze(), input.squeeze()
    ic1 = pearsonr(y1_hat, y1, batch_first=False)
    ic2 = pearsonr(y2_hat, y2, batch_first=False)
    q9_1 = calc_topk(y1_hat, y1, topk=10)
    q9_2 = calc_topk(y2_hat, y2, topk=10)
    q99_1 = calc_topk(y1_hat, y1, topk=1)
    q99_2 = calc_topk(y2_hat, y2, topk=1)
    res = {
        f"{stage}_ic_{name_y[0]}": ic1,
        f"{stage}_ic_{name_y[1]}": ic2,
        f"{stage}_q90_{name_y[0]}": q9_1,
        f"{stage}_q90_{name_y[1]}": q9_2,
        f"{stage}_q99_{name_y[0]}": q99_1,
        f"{stage}_q99_{name_y[1]}": q99_2,
        f"{stage}_ic": (ic1 + ic2) / 2,
        f"{stage}_q90": (q9_1 + q9_2) / 2,
        f"{stage}_q99": (q99_1 + q99_2) / 2,
    }
    return res


def metric_callback_hd(input, target, stage="test"):
    y1, y2 = target[:, 0].squeeze(), target[:, 1].squeeze()
    y1_hat, y2_hat = input[:, 0], input[:, 1]
    test_hd90 = calc_topk(y1_hat, y2 + y1, topk=10)
    test_hd99 = calc_topk(y1_hat, y2 + y1, topk=1)
    test_dh90 = calc_topk(y2_hat, y1 + y2, topk=10)
    test_dh99 = calc_topk(y2_hat, y1 + y2, topk=1)
    return {
        f"{stage}_hd90": test_hd90,
        f"{stage}_hd99": test_hd99,
        f"{stage}_dh90": test_dh90,
        f"{stage}_dh99": test_dh99,
    }


def single_metric_callback(input: Tensor, target: Tensor, stage: str = "train") -> Dict:
    y = target[:, 0].squeeze()
    yhat = input.squeeze()
    ic = pearsonr(yhat, y, batch_first=False)
    q90 = calc_topk(yhat, y, topk=10)
    q99 = calc_topk(yhat, y, topk=1)
    res = {
        f"{stage}_ic": ic,
        f"{stage}_q90": q90,
        f"{stage}_q99": q99,
    }
    return res


def valid_batch(batch: Tensor) -> Tuple[Tensor, Tensor]:
    x, y = batch
    idx = ~th.isnan(y).any(dim=1).squeeze()
    ret_x, ret_y = x[idx, ...], y[idx, ...].squeeze().nan_to_num(0, 0, 0)
    # print("valid rate:{:.2f}%".format(100 * len(ret_y) / len(y)))
    return ret_x, ret_y


def valid_ts_batch(batch: Tensor) -> Tuple[Tensor, Tensor]:
    x, y = batch
    x, y = x.squeeze(), y.squeeze()
    idx = ~th.isnan(y).any(dim=1).squeeze()
    ret_x, ret_y = x[:, idx, :], y[idx, ...]
    ret_x = ret_x.permute(1, 0, 2)
    return ret_x, ret_y


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
        pearson_r = pearsonr_(input, target)
        # Calculate the Concordance Correlation Coefficient (CCC)
        ccc = (2 * pearson_r * th.sqrt(var_input) * th.sqrt(var_target)) / (
            var_input + var_target + (mean_input - mean_target) ** 2
        )
        # Return 1 minus the CCC as the loss to be minimized
        return 1 - ccc


class PearsonCorrelation(nn.Module):
    def forward(self, x, y):
        mean_x = th.nanmean(x)
        mean_y = th.nanmean(y)
        xm = x - mean_x
        ym = y - mean_y
        r_num = th.sum(xm * ym)
        r_den = th.sqrt(th.sum(xm**2)) * th.sqrt(th.sum(ym**2))
        r_val = r_num / r_den
        return r_val.float()


class WeightedPearsonLoss(nn.Module):
    def __init__(self, alpha=1, beta=100):
        super(WeightedPearsonLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pearson_corr = PearsonCorrelation()

    def forward(self, yhat, y):
        yh_pred = yhat[:, 0]
        yd_pred = yhat[:, 1]
        yh_true = y[:, 0]
        yd_true = y[:, 1]

        corr_yh = self.pearson_corr(yh_pred, yh_true)
        corr_yd = self.pearson_corr(yd_pred, yd_true)
        weights = th.sigmoid(self.alpha * th.abs(yh_pred) * self.beta)
        weighted_corr_yd = corr_yd * (weights) * 10
        loss = -corr_yh - th.mean(weighted_corr_yd) - corr_yd
        return loss


class TestLoss(nn.Module):
    def __init__(self, alpha=1, beta=100):
        super(TestLoss, self).__init__()
        self.pearson_corr = PearsonCorrelation()
        self.alpha = alpha
        self.beta = beta

    def forward(self, yhat, y):
        yh_pred = yhat[:, 0]
        yd_pred = yhat[:, 1]
        yh_true = y[:, 0]
        yd_true = y[:, 1]

        corr_yh = self.pearson_corr(yh_pred, yh_true)
        corr_yd = self.pearson_corr(yd_pred, yd_true)
        corr_inter = self.pearson_corr(yh_pred, yd_true)

        return -(corr_yh**2) - corr_yd**2 - corr_inter


class SumLoss(nn.Module):
    def __init__(self, alpha=1, beta=100, gamma=0.9):
        super(SumLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # weight decay factor for moving average
        self.corr = (
            PearsonCorrelation()
        )  # Assuming PearsonCorrelation is implemented elsewhere
        self.mse = nn.MSELoss()

        # Initialize moving averages for variance
        self.var_corr_yh = 1.0
        self.var_corr_inter = 1.0
        self.var_error = 1.0
        self.var_corr_yd = 1.0

    def update_moving_average(self, current_var, new_value):
        return self.gamma * current_var + (1 - self.gamma) * new_value

    def forward(self, yhat, y):
        yhat_h, yhat_d = yhat[:, 0], yhat[:, 1]
        y_h, y_d = y[:, 0], y[:, 1]
        yhat_sum, y_sum = yhat.sum(1), y.sum(1)

        corr_yh = self.corr(yhat_h, y_h)
        corr_yd = self.corr(yhat_d, y_d)
        corr_inter = self.corr(yhat_h, y_sum)
        # error = self.mse(yhat_sum.float(), y_sum.float())

        # Update moving averages
        self.var_corr_yh = self.update_moving_average(
            self.var_corr_yh, corr_yh.detach()
        )
        self.var_corr_yd = self.update_moving_average(
            self.var_corr_yd, corr_yd.detach()
        )
        self.var_corr_inter = self.update_moving_average(
            self.var_corr_inter, corr_inter.detach()
        )
        total_loss = (
            -1 / self.var_corr_yh * corr_yh
            - 1 / self.var_corr_yd * corr_yd / 5
            - 1 / self.var_corr_inter * corr_inter
            + th.log(self.var_corr_yh)
            + th.log(self.var_corr_yd)
            + th.log(self.var_corr_inter)
        )
        return total_loss


def reg_loss(input, target):
    return F.mse_loss(input.float(), target.float())


# def rank_loss(input, target):
#     all_ones = th.ones_like(input, device=input.device,dtype=input.dtype)
#     target =target.to(input.dtype)
#     part_input = th.sub(input @ all_ones.t(), all_ones @ input.t())
#     # part_target = target - target.t()
#     part_target = th.sub(all_ones @ target.t(), target @ all_ones.t())
#     loss = th.mean(F.relu(part_input * part_target))
#     print(f"rankloss:{loss}")
#     return loss
def rank_loss(input, target):
    # 将目标转换为与输入相同的数据类型
    target = target.to(input.dtype)

    # 计算成对差异
    part_input = input.unsqueeze(1) - input.unsqueeze(0)
    part_target = target.unsqueeze(1) - target.unsqueeze(0)

    # 排名损失：对于目标得分高的样本，预测得分也应更高
    loss = th.mean(F.relu(-part_input * part_target))

    return loss


class TRSRLoss(nn.Module):
    def __init__(self, alpha=100):
        super(TRSRLoss, self).__init__()
        self.alpha = alpha
        self.corr = PearsonCorrelation()

    def forward(self, yhat, y):
        loss = 0
        for i in range(2):
            loss += -self.corr(yhat[:, i], y[:, i]) + self.alpha * rank_loss(
                yhat[:, i], y[:, i]
            )
        return loss


class Squeeze(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        out = x.squeeze()
        return out


def weight_init(model: pl.LightningModule):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            print(f"init {type(m)}")
            nn.init.kaiming_normal_(
                m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
            )
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            print(f"init {type(m)}")
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            print(f"init {type(m)}")
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
        else:
            pass


def configure_optimizers(model, lr, weight_decay):
    bn_params = []
    other_params = []
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_params.extend(list(module.parameters()))
        elif any(isinstance(module, cls) for cls in [nn.Linear, nn.LSTM]):
            other_params.extend(list(module.parameters()))
    print("bn_params:", len(bn_params))
    optimizer = optim.AdamW(
        [
            {"params": bn_params, "weight_decay": 0.0},  # BN层不使用权重衰减
            {
                "params": other_params,
                "weight_decay": weight_decay,
            },  # 其他层使用权重衰减
        ],
        lr=lr,
    )
    return optimizer
