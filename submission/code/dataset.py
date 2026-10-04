"""dataset.py - nạp dữ liệu DeepWeeds, kiểm tra chia fold, transforms và DataLoader."""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv."""
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None) -> dict:
    """Kiểm tra tính toàn vẹn phân vùng dữ liệu theo nguyên tắc S1-S4."""
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total = n_train + n_val + n_test

    assert total == 17509, f"Tổng số ảnh ({total}) khác kỳ vọng 17.509"

    # Kiểm tra giao rỗng
    s_train = set(train_df["Filename"])
    s_val = set(val_df["Filename"])
    s_test = set(test_df["Filename"])

    assert len(s_train & s_val) == 0, "Giao train và val không rỗng"
    assert len(s_train & s_test) == 0, "Giao train và test không rỗng"
    assert len(s_val & s_test) == 0, "Giao val và test không rỗng"

    if images_dir and os.path.exists(images_dir):
        for name in list(s_train | s_val | s_test)[:50]:
            assert (Path(images_dir) / name).exists(), f"Ảnh {name} không tồn tại trong {images_dir}"

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "total": total,
        "train_per_class": dict(train_df["Label"].value_counts().sort_index()),
        "val_per_class": dict(val_df["Label"].value_counts().sort_index()),
        "test_per_class": dict(test_df["Label"].value_counts().sort_index()),
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transforms theo chế độ train/eval và mức độ augmentation."""
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    if not train:
        # Đánh giá: Resize 256 rồi CenterCrop hoặc resize thẳng img_size
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize,
        ])

    if aug == "color":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            normalize,
        ])
    elif aug == "randaug":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            normalize,
        ])
    else:  # basic
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh DeepWeeds trả về (tensor, label_int, filename_str)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].values
        self.labels = self.df["Label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        fn = self.filenames[idx]
        label = self.labels[idx]
        img_path = self.images_dir / fn
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, fn


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader PyTorch."""
    ds = DeepWeedsDataset(df, images_dir, transform=transform)
    
    sampler_obj = None
    shuffle = train
    if train and sampler == "balanced":
        class_counts = df["Label"].value_counts().sort_index().values
        weights = 1.0 / class_counts
        sample_weights = [weights[l] for l in df["Label"]]
        sampler_obj = WeightedRandomSampler(weights=sample_weights, num_samples=len(df), replacement=True)
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train
    )
