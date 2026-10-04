# HƯỚNG DẪN CHẠY LẠI VÀ THÔNG TIN BÀI NỘP LAB DAY 2

## 1. Thông tin sinh viên & Môi trường thực thi
- **Sinh viên:** Nguyễn Minh Quyền
- **Mã số sinh viên (MSSV):** 2A202602438
- **Bài lab:** Phân loại cỏ dại DeepWeeds (Backbone, Công thức huấn luyện, Suy luận).
- **Link Notebook Google Colab chính thức:** [Google Colab Notebook](https://colab.research.google.com/drive/1WlYAPip7qQPRhHI0QsJYmC5TDSe-DGAW)
- **Môi trường phần cứng:** GPU NVIDIA Tesla T4 (16GB VRAM), Google Colab / Kaggle.
- **Phiên bản thư viện chính:**
  - Python: 3.10+
  - PyTorch: 2.x
  - timm: 1.0.x
  - pandas: 2.x / 3.x
  - numpy: 1.25+ / 2.x
  - openpyxl: 3.1+
  - scikit-learn: 1.4+
- **Cố định hạt giống (Seeds):** `seed = 0, 1, 2` (nguyên tắc N4, ddof=1).

---

## 2. Cấu trúc thư mục bài nộp

```
submission/
├── README.md              # File hướng dẫn chạy lại này
├── results.xlsx           # Bảng tổng hợp 7 sheet (Backbones, Training, Inference, Final, PerClass, Latency, Summary)
├── report.md              # Báo cáo khoa học 9 phần chuẩn IEEE/Markdown (250+ dòng, 4300+ từ)
├── curves/                # Biểu đồ training loss & metric của từng thí nghiệm (B01-B05, T00-T04, F01)
├── predictions/           # File dự đoán test/val của chung kết và mốc cho tất cả các seed
└── code/                  # Toàn bộ mã nguồn hoàn thiện, không còn stub
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    └── lab_day2.ipynb
```

---

## 3. Thứ tự và Lệnh chạy lại thực nghiệm

### Bước 1: Tải dữ liệu và cài đặt môi trường
```bash
pip install -q timm openpyxl scikit-learn pandas numpy matplotlib
```

### Bước 2: Huấn luyện và tái lập kết quả
Mọi cấu hình đều được gọi qua một hàm duy nhất trong `train.py`:
```bash
# Huấn luyện mô hình mốc T00 (3 seed)
python code/train.py --set exp_id=T00 backbone=deit_small_patch16_224 seed=0 save_test_predictions=True
python code/train.py --set exp_id=T00 backbone=deit_small_patch16_224 seed=1 save_test_predictions=True
python code/train.py --set exp_id=T00 backbone=deit_small_patch16_224 seed=2 save_test_predictions=True

# Huấn luyện mô hình chung kết F01 (3 seed)
python code/train.py --set exp_id=F01 backbone=deit_small_patch16_224 seed=0 save_test_predictions=True
python code/train.py --set exp_id=F01 backbone=deit_small_patch16_224 seed=1 save_test_predictions=True
python code/train.py --set exp_id=F01 backbone=deit_small_patch16_224 seed=2 save_test_predictions=True
```

### Bước 3: Đánh giá chỉ số và Chấm điểm tự động bằng `eval.py`
```bash
# 1. Tính chỉ số độc lập cho cấu hình chung kết F01
python eval.py score --pred "submission/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

# 2. Tự chấm điểm mục I theo tiêu chí RUBRIC
python eval.py grade --final "submission/predictions/F01_seed*_test.csv" \
    --baseline "submission/predictions/T00_seed*_test.csv" \
    --uncal "submission/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submission/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 13.44 \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```
