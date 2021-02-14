import torch


def pointwise_ranking_loss(y, target) -> torch.Tensor:
    mse = torch.square(y - target)
    return torch.mean(mse)


def listwise_ranking_loss(y, target) -> torch.Tensor:
    P_n = torch.softmax(y, 0)
    P_n_hat = torch.softmax(target, 0)
    loss = - torch.sum(torch.log(P_n_hat) * P_n)
    return loss
