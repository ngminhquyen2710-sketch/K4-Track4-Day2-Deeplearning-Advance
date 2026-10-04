"""benchmark.py - đo độ trễ suy luận p50/p95/p99 đúng chuẩn (warm-up, CUDA sync, nhiều lần lặp)."""
from __future__ import annotations

import time
import numpy as np
import torch


def measure_latency(model: torch.nn.Module, input_size: tuple[int, int, int] = (3, 224, 224),
                    batch_size: int = 1, device: str = "cuda", dtype: torch.dtype = torch.float16,
                    warmup: int = 10, repeats: int = 50) -> dict:
    """Đo độ trễ p50, p95, p99 và thông lượng theo GUIDE.md mục 4.1.
    
    Quy tắc:
    - Bắt buộc gọi torch.cuda.synchronize() trước và sau mỗi lần đo
    - Warm-up bỏ qua các lần chạy đầu
    - Chạy tối thiểu >= 50 lần đo
    """
    model.eval()
    model.to(device)
    
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dummy_input = torch.randn(batch_size, *input_size, device=device, dtype=dtype)
    
    # 1. Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()

    # 2. Benchmark repeats
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            _ = model(dummy_input)
            
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
            
    times = np.array(times)
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))
    mean = float(np.mean(times))
    
    # Thông lượng: số ảnh / giây
    throughput = (batch_size / (mean / 1000.0))
    
    return {
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(mean, 3),
        "throughput_img_s": round(throughput, 2),
        "batch_size": batch_size,
        "dtype": str(dtype),
        "device": device
    }
