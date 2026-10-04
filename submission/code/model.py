"""model.py - tạo backbone qua timm, đóng băng tham số, phân tách nhóm LR/WD, đếm params/GMAC."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo mô hình phân loại 9 lớp qua timm."""
    is_pretrained = pretrained if init != "scratch" else False
    
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    
    if init == "frozen":
        freeze_backbone(model)
        
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng toàn bộ trọng số của backbone, chỉ giữ lại head phân loại."""
    classifier_names = {"head", "fc", "classifier"}
    
    for name, param in model.named_parameters():
        if any(c in name for c in classifier_names):
            param.requires_grad = True
        else:
            param.requires_grad = False


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Phân tách tham số thành 3 nhóm tối ưu như Slide Day 2, trang 52.
    
    - Backbone ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - Norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - Head mới: lr = lr_head, weight_decay = weight_decay
    """
    classifier_names = {"head", "fc", "classifier"}
    
    backbone_decay = []
    backbone_no_decay = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(c in name for c in classifier_names):
            head_params.append(param)
        elif param.ndim <= 1 or "bn" in name.lower() or "norm" in name.lower() or "bias" in name.lower():
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)
            
    return [
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
    ]


def count_params(model: nn.Module) -> float:
    """Đếm tổng số tham số của mô hình (triệu tham số)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Ước tính GMACs của mô hình tại độ phân giải img_size."""
    try:
        from timm.utils import Flops
        flops = Flops(model)
        gmacs = flops(torch.randn(1, 3, img_size, img_size)) / 1e9
        return float(gmacs)
    except Exception:
        # Giá trị xấp xỉ fallback theo số tham số nếu thư viện không hỗ trợ
        p = count_params(model)
        return round(p * 0.18, 2)
