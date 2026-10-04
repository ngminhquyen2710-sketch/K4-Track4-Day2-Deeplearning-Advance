"""losses.py - các hàm loss (CE, Label Smoothing, Focal Loss, Class Weights, Mixup/CutMix)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-Entropy có làm mịn nhãn (Label Smoothing)."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        
        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)
            
        loss_ce = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        loss = (1.0 - self.smoothing) * loss_ce + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss cho bài toán mất cân bằng lớp. gamma=0 tương đương CE."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(name: str = "ce", num_classes: int = 9, label_smoothing: float = 0.0,
               gamma: float = 0.0, weight: torch.Tensor | None = None) -> nn.Module:
    """Tạo hàm mất mát theo cấu hình."""
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    elif name == "label_smoothing" or label_smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=weight)
    else:
        return nn.CrossEntropyLoss(weight=weight)


def class_weights_from_df(train_df: pd.DataFrame, num_classes: int = 9,
                          mode: str = "inverse") -> torch.Tensor:
    """Tính trọng số nghịch đảo tần suất lớp để bù trừ mất cân bằng."""
    counts = train_df["Label"].value_counts().sort_index().values
    if mode == "inverse":
        weights = 1.0 / (counts.astype(np.float32) + 1e-6)
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts.astype(np.float32) + 1e-6)
    else:
        weights = np.ones(num_classes, dtype=np.float32)
        
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def apply_mixup_cutmix(x: torch.Tensor, y: torch.Tensor, alpha_mixup: float = 0.8,
                       alpha_cutmix: float = 1.0, prob: float = 0.5):
    """Áp dụng Mixup hoặc CutMix ngẫu nhiên trên batch."""
    if np.random.rand() > prob:
        return x, y, y, 1.0

    lam = np.random.beta(alpha_mixup, alpha_mixup)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Mixup
    x_mixed = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam


def mixup_loss(criterion: nn.Module, preds: torch.Tensor, y_a: torch.Tensor,
               y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Tính loss có trọng số giữa 2 nhãn của Mixup."""
    return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)
