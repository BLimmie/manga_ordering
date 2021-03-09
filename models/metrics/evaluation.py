import torch
from scipy import stats
import numpy as np

def kendall_tau(y: torch.Tensor, target: torch.Tensor) -> float:
    y = y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    y = stats.rankdata(y)
    target = stats.rankdata(target)
    tau, _ = stats.kendalltau(target, y, variant='b')
    if np.isnan([tau])[0]:
        tau = 0.0
    return tau


def pmr(y: torch.Tensor, target: torch.Tensor) -> float:
    y = y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    y = stats.rankdata(y)
    target = stats.rankdata(target)
    return 1.0 if all([i == j for i, j in zip(y, target)]) else 0.0
