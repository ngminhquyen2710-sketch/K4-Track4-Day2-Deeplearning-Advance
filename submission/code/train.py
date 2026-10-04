"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F)."""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import eval as ev
from dataset import load_split, build_transforms, make_loader
from model import build_model, param_groups
from losses import build_loss, apply_mixup_cutmix, mixup_loss


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> AdamW:
    """AdamW với 3 nhóm tham số."""
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số (Exponential Moving Average)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    total_loss = 0.0
    
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixup / Cutmix
        if cfg.mix:
            images, targets_a, targets_b, lam = apply_mixup_cutmix(images, targets)
            with torch.amp.autocast(device_type=device, enabled=cfg.amp):
                preds = model(images)
                loss = mixup_loss(criterion, preds, targets_a, targets_b, lam)
        else:
            with torch.amp.autocast(device_type=device, enabled=cfg.amp):
                preds = model(images)
                loss = criterion(preds, targets)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema:
            ema.update(model)
            
        total_loss += loss.item() * len(targets)
        
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_filenames, all_logits = [], [], [], []
    
    for images, targets, filenames in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        with torch.amp.autocast(device_type=device, enabled=cfg.amp):
            logits = model(images)
            loss = criterion(logits, targets)
            
        total_loss += loss.item() * len(targets)
        probs = torch.softmax(logits, dim=-1)
        all_logits.append(logits.float().cpu().numpy())
        all_preds.append(probs.argmax(-1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_filenames.extend(filenames)
        
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    logits_arr = np.concatenate(all_logits)
    exp_l = np.exp(logits_arr - logits_arr.max(axis=-1, keepdims=True))
    probs_arr = exp_l / exp_l.sum(axis=-1, keepdims=True)
    
    metrics = ev.compute_metrics(y_true, y_pred, probs_arr)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, all_filenames, y_true, logits_arr, probs_arr


def run(cfg: Config) -> dict:
    """Một hàm run() duy nhất cho mọi cấu hình."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    
    train_transform = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    eval_transform = build_transforms(train=False, img_size=cfg.img_size)
    
    train_loader = make_loader(train_df, cfg.images_dir, train_transform, cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, eval_transform, cfg.batch_size, train=False, num_workers=cfg.num_workers)
    
    model = build_model(cfg.backbone, pretrained=True, num_classes=ev.NUM_CLASSES, drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)
    
    criterion = build_loss(cfg.loss, label_smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device_type=device, enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    
    history = []
    best_f1 = -1.0
    best_logits = None
    
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        sec_epoch = time.time() - t0
        
        val_metrics, val_names, val_y, val_l, val_p = evaluate(model, val_loader, criterion, cfg, device)
        
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top1": val_metrics["top1"],
            "sec_epoch": sec_epoch,
            "lr_backbone": optimizer.param_groups[0]["lr"]
        }
        history.append(row)
        
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_logits = val_l
            torch.save(model.state_dict(), rdir / "best.pt")
            np.save(rdir / "val_logits.npy", best_logits)
            
    # Lưu history
    pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
    
    # Save test predictions nếu bật
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, eval_transform, cfg.batch_size, train=False, num_workers=cfg.num_workers)
        test_metrics, test_names, test_y, test_l, test_p = evaluate(model, test_loader, criterion, cfg, device)
        ev.save_predictions(pred_path(cfg, "test"), test_names, test_y, test_p)
        
    return {"best_f1": best_f1, "history": history}
