import torch


def pointwise_ranking_loss(y, target) -> torch.Tensor:
    mse = torch.square(y - target)
    return torch.mean(mse)


def listwise_ranking_loss(y, target) -> torch.Tensor:
    P_n = torch.softmax(y, 0)
    P_n_hat = torch.softmax(target, 0)
    loss = - torch.sum(torch.log(P_n_hat) * P_n)
    return loss


def listMLE(y, target, eps=1e-10):
    y =y.view(1,-1)
    target = target.view(1,-1)
    random_indices = torch.randperm(y.shape[-1])
    y_pred_shuffled = y[:, random_indices]
    y_true_shuffled = target[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)


    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))
