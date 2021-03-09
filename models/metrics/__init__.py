from typing import Dict

import torch

from .evaluation import pmr, kendall_tau


def calculate_metrics(y: torch.Tensor, target: torch.Tensor) -> Dict[str,float]:
    return {"Perfect Match Ratio": pmr(y, target), "Kendall's Tau": kendall_tau(y, target)}
