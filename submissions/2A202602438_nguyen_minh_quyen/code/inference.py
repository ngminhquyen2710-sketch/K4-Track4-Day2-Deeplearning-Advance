"""inference.py - suy luận, TTA, temperature scaling, ensemble và gộp kết quả."""
from __future__ import annotations

import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F


def predict_loader(model: torch.nn.Module, loader, device: str = "cuda", amp: bool = True):
    """Suy luận toàn bộ loader trả về filenames, y_true, logits."""
    model.eval()
    model.to(device)
    
    all_logits = []
    all_targets = []
    all_filenames = []
    
    with torch.no_grad():
        for images, targets, filenames in loader:
            images = images.to(device)
            with torch.amp.autocast(device_type=device, enabled=amp):
                logits = model(images)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(targets.numpy())
            all_filenames.extend(filenames)
            
    return all_filenames, np.concatenate(all_targets), np.concatenate(all_logits)


def tta_hflip(model: torch.nn.Module, loader, device: str = "cuda", amp: bool = True, mode: str = "logit"):
    """TTA lật ngang (K=2 views: ảnh gốc + lật ngang)."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_filenames = []
    
    with torch.no_grad():
        for images, targets, filenames in loader:
            images = images.to(device)
            images_flip = torch.flip(images, dims=[-1])
            
            with torch.amp.autocast(device_type=device, enabled=amp):
                l_orig = model(images)
                l_flip = model(images_flip)
                
            if mode == "logit":
                l_mean = (l_orig + l_flip) / 2.0
                p_mean = F.softmax(l_mean, dim=-1)
            else:  # prob
                p_orig = F.softmax(l_orig, dim=-1)
                p_flip = F.softmax(l_flip, dim=-1)
                p_mean = (p_orig + p_flip) / 2.0
                
            all_preds.append(p_mean.float().cpu().numpy())
            all_targets.append(targets.numpy())
            all_filenames.extend(filenames)
            
    return all_filenames, np.concatenate(all_targets), np.concatenate(all_preds)


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray) -> float:
    """Khớp một tham số nhiệt độ T > 0 duy nhất trên tập validation bằng tối ưu NLL."""
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_y, dtype=torch.int64)
    
    def loss_fun(t_arr):
        t = float(t_arr[0])
        scaled_logits = logits_t / t
        return F.cross_entropy(scaled_logits, labels_t).item()
        
    res = scipy.optimize.minimize(loss_fun, x0=[1.0], bounds=[(0.05, 10.0)], method='L-BFGS-B')
    return float(res.x[0])


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Chia logit cho T rồi áp dụng softmax đúng 1 lần."""
    scaled = logits / T
    scaled_shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp_s = np.exp(scaled_shifted)
    return exp_s / exp_s.sum(axis=-1, keepdims=True)


def ensemble_logits(logits_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Gộp logit có trọng số từ nhiều mô hình rồi softmax."""
    if weights is None:
        weights = [1.0 / len(logits_list)] * len(logits_list)
    combined = sum(w * l for w, l in zip(weights, logits_list))
    shifted = combined - combined.max(axis=-1, keepdims=True)
    exp_c = np.exp(shifted)
    return exp_c / exp_c.sum(axis=-1, keepdims=True)


def ensemble_probs(probs_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Trung bình cộng xác suất từ nhiều mô hình."""
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    return sum(w * p for w, p in zip(weights, probs_list))
