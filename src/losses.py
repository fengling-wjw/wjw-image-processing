from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def mask_to_edge(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    pad = width // 2
    dilated = F.max_pool2d(mask, width, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, width, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def compute_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, variant: str):
    mask_logits = outputs["mask"]
    if variant == "baseline":
        loss = F.binary_cross_entropy_with_logits(mask_logits, target)
        return loss, {"bce": float(loss.detach())}
    bce = F.binary_cross_entropy_with_logits(mask_logits, target)
    dice = soft_dice_loss(mask_logits, target)
    edge_target = mask_to_edge(target)
    edge = F.binary_cross_entropy_with_logits(outputs["edge"], edge_target)
    total = 0.35 * bce + 0.55 * dice + 0.10 * edge
    return total, {"bce": float(bce.detach()), "dice": float(dice.detach()), "edge": float(edge.detach())}

