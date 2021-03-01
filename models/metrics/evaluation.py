import torch
from scipy import stats


def kendall_tau(y: torch.Tensor, target: torch.Tensor) -> float:
    y = y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    y = stats.rankdata(y)
    target = stats.rankdata(target)
    tau, _ = stats.kendalltau(target, y, variant='c')
    return tau


def pmr(y: torch.Tensor, target: torch.Tensor) -> float:
    y = y.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    y = stats.rankdata(y)
    target = stats.rankdata(target)
    return 1.0 if all([i == j for i, j in zip(y, target)]) else 0.0
