from __future__ import annotations

import torch
from torch.nn import functional as F


def _edge(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, 3, 1, 1)
    eroded = -F.max_pool2d(-mask, 3, 1, 1)
    return (dilated - eroded) > 0.5


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    pred_f, truth_f = pred.float(), truth.float()
    dims = (1, 2, 3)
    tp = (pred_f * truth_f).sum(dims)
    fp = (pred_f * (1 - truth_f)).sum(dims)
    fn = ((1 - pred_f) * truth_f).sum(dims)
    tn = ((1 - pred_f) * (1 - truth_f)).sum(dims)
    eps = 1e-7
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    pe, te = _edge(pred_f), _edge(truth_f)
    pe_d = F.max_pool2d(pe.float(), 5, 1, 2) > 0
    te_d = F.max_pool2d(te.float(), 5, 1, 2) > 0
    edge_precision = ((pe & te_d).sum(dims).float() + eps) / (pe.sum(dims).float() + eps)
    edge_recall = ((te & pe_d).sum(dims).float() + eps) / (te.sum(dims).float() + eps)
    boundary_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall + eps)
    return {
        "iou": float(iou.mean()),
        "dice": float(dice.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "pixel_accuracy": float(accuracy.mean()),
        "boundary_f1": float(boundary_f1.mean()),
    }

