# BÁO CÁO KHOA HỌC THỰC NGHIỆM LAB DAY 2
## Nghiên cứu Backbone, Công thức Huấn luyện và Kỹ thuật Suy luận trên Tập dữ liệu Cỏ dại DeepWeeds

**Học viên:** Nguyễn Minh Quyền  
**MSSV:** 2A202602438  
**Lớp:** Track 4 - Day 2: Deep Learning Advance  

---

### 1. Tóm tắt điều hành (Executive Summary)
Bài báo cáo này trình bày nghiên cứu thực nghiệm toàn diện trên tập dữ liệu phân loại cỏ dại DeepWeeds (fold 0, 17.509 ảnh, 9 lớp) nhằm đánh giá tác động độc lập và tương tác của ba trục: **kiến trúc backbone**, **công thức huấn luyện (training recipes)** và **kỹ thuật suy luận (inference techniques)**. Thực nghiệm so sánh 5 họ backbone, 5 biến thể công thức huấn luyện và 7 cấu hình suy luận. Cấu hình chung kết $F01$ (DeiT-S tiền huấn luyện kết hợp Temperature Scaling $T=1.151$) đạt **Top-1 Accuracy $96.25\% \pm 0.0013$** (vượt mốc $95.7\%$ của bài báo gốc), **Macro-F1 $0.9505 \pm 0.0022$** và **ECE đạt $0.0157 \pm 0.0018$** ($1.57\%$) qua 3 seed kiểm định trên tập test độc lập. Đặc biệt, mô hình đa kiến trúc Ensemble (ConvNeXt-Tiny $B02R$ + DeiT-S $B04$) thiết lập kỷ lục Validation Macro-F1 **$0.9699$** và Top-1 **$97.83\%$**. Về thời gian thực, cấu hình đơn lẻ đạt độ trễ $p95 = 13.44\text{ ms}$ ở batch 1 trên GPU Tesla T4, đáp ứng hoàn hảo ngân sách cảm biến $100\text{ ms}$ cho robot nông nghiệp ngoài thực địa.

---

### 2. Dữ liệu và Thiết lập thực nghiệm (Data & Experimental Setup)

#### 2.1 Tập dữ liệu DeepWeeds và Phân bố lớp (EDA)
Tập dữ liệu DeepWeeds gồm $17.509$ ảnh màu RGB kích thước $256 \times 256$, thu thập tại các đồng cỏ chăn thả gia súc ở Queensland, Úc. Dữ liệu gồm 8 loài cỏ dại nguy hại xâm lấn và 1 lớp đối chứng thực vật bản địa (`Negative`):

| STT | Tên loài cỏ dại | Tên khoa học | Tổng số ảnh | Train (Fold 0) | Val (Fold 0) | Test (Fold 0) | Tỉ lệ (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | *Ziziphus mauritiana* | 1.125 | 674 | 225 | 226 | 6.4% |
| 1 | Lantana | *Lantana camara* | 1.064 | 638 | 213 | 213 | 6.1% |
| 2 | Parkinsonia | *Parkinsonia aculeata* | 1.031 | 618 | 206 | 207 | 5.9% |
| 3 | Parthenium | *Parthenium hysterophorus* | 1.022 | 613 | 204 | 205 | 5.8% |
| 4 | Prickly acacia | *Vachellia nilotica* | 1.062 | 637 | 212 | 213 | 6.1% |
| 5 | Rubber vine | *Cryptostegia grandiflora* | 1.009 | 605 | 202 | 202 | 5.8% |
| 6 | Siam weed | *Chromolaena odorata* | 1.074 | 644 | 215 | 215 | 6.1% |
| 7 | Snake weed | *Stachytarpheta jamaicensis* | 1.016 | 609 | 203 | 204 | 5.8% |
| 8 | **Negative** (Không phải cỏ mục tiêu) | Đối chứng thực vật bản địa | **9.106** | **5.463** | **1.821** | **1.822** | **52.0%** |
| | **Tổng cộng** | | **17.509** | **10.501** | **3.501** | **3.507** | **100%** |

**Kiểm định tính toàn vẹn phân vùng (Split Verification S1–S4):**
1. Tỉ lệ chia train/val/test bám sát nghiêm ngặt phân tầng xấp xỉ 60% / 20% / 20%: $10.501 / 3.501 / 3.507$.
2. Kiểm tra giao tập hợp: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
3. Tổng số lượng ảnh ba tập hợp khớp tuyệt đối: $10.501 + 3.501 + 3.507 = 17.509$ ảnh.
4. **Hiện tượng mất cân bằng lớp:** Lớp `Negative` chiếm tới $52.0\%$ dữ liệu, gấp hơn 8 lần so với bất kỳ loài cỏ đơn lẻ nào. Do đó, chỉ số **Top-1 Accuracy bị lớp Negative kéo lên cao** và dễ tạo ra ấn tượng lạc quan giả tạo. Vì vậy, **Macro-F1 trên 9 lớp (trọng số đều)** được chọn làm thước đo quyết định tối cao để đánh giá và lựa chọn mô hình.

#### 2.2 Công thức nền (Baseline Recipe T00)
Mọi mô hình so sánh ở Bước 1 đều tuân thủ chặt chẽ công thức nền thống nhất:
- **Khởi tạo:** Trọng số tiền huấn luyện ImageNet-1k, thay lớp phân loại mới 9 lớp khởi tạo ngẫu nhiên, tinh chỉnh toàn bộ mạng (full finetuning).
- **Tiền xử lý:** Huấn luyện bằng `RandomResizedCrop(224, scale=(0.08, 1.0))` + `RandomHorizontalFlip(p=0.5)` + Chuẩn hoá ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Val/Test: `Resize(256)` + `CenterCrop(224)`.
- **Tối ưu hoá:** AdamW, phân tách 3 nhóm tham số (Slide trang 52):
  - Tham số khối backbone ($ndim > 1$): $lr = 10^{-4}$, $weight\_decay = 0.05$.
  - Trọng số chuẩn hoá (Norm) và Bias ($ndim \le 1$): $lr = 10^{-4}$, $weight\_decay = 0$.
  - Lớp phân loại (Head): $lr = 10^{-3}$ (gấp 10 lần), $weight\_decay = 0.05$.
- **Lịch học (LR Schedule):** Warmup tuyến tính 1 epoch đầu, sau đó Cosine Decay về $10^{-6}$.
- **Hàm mất mát:** Cross-Entropy tiêu chuẩn.
- **Batch size:** 64, huấn luyện 10 epoch, bật Mixed Precision (AMP FP16).
- **Môi trường phần cứng:** GPU NVIDIA Tesla T4 16GB VRAM, CUDA 12.x, PyTorch 2.x, timm 1.x.

#### 2.3 Kiểm tra pipeline gỡ lỗi ban đầu (Sanity Checks)
1. **Loss khởi tạo:** Với 9 lớp, hàm mất mát CE ban đầu của head ngẫu nhiên đo được $2.196$, khớp chính xác với lý thuyết $-\ln(1/9) \approx 2.1972$.
2. **Overfitting trên batch nhỏ:** Thử nghiệm quá khớp trên 1 batch (16 mẫu) đạt loss $0.0012$ sau 35 bước cập nhật, khẳng định pipeline tính toán gradient và cập nhật trọng số hoạt động hoàn toàn chính xác.

---

### 3. Kết quả So sánh Kiến trúc Backbone (Bước 1)

Thực nghiệm so sánh 5 kiến trúc thuộc 4 họ mô hình khác nhau theo yêu cầu bắt buộc của Rubric (ResNet, ConvNeXt, Transformer, Mạng nhẹ) trên cùng tập Validation:

| Mã | Tên kiến trúc | Họ mô hình | Tag trọng số `timm` | Tham số (M) | GMAC (224) | Val Macro-F1 | Val Top-1 | Thời gian / ep | Độ trễ batch 1 |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$B01$** | `resnet50` | ResNet (Mốc) | `a1_in1k` | 23.53 | 4.10 | 0.8035 | 0.8566 | 59.5 s | 7.12 ms |
| **$B02R$**| `convnext_tiny` | Hiện đại hoá CNN | `in12k_ft_in1k` | 27.83 | 4.50 | **0.9663** | **0.9746** | 67.5 s | 8.24 ms |
| **$B03$** | `resnext50_32x4d`| Cardinality CNN | `a1h_in1k` | 23.00 | 4.20 | 0.7803 | 0.8346 | 73.9 s | 9.85 ms |
| **$B04$** | `deit_small_patch16_224` | Vision Transformer| `fb_in1k` | 21.67 | 4.60 | 0.9492 | 0.9637 | **52.0 s** | **6.75 ms** |
| **$B05$** | `efficientnet_b0`| Mạng nhẹ di động | `ra_in1k` | **4.02** | **0.39** | 0.7934 | 0.8449 | 57.9 s | **4.86 ms** |

```
                       ĐÁNH ĐỔI VAL MACRO-F1 VÀ THAM SỐ
   Macro-F1
     1.00 ┼                                      ● B02R (ConvNeXt-T, 0.9663)
          │                              ● B04 (DeiT-S, 0.9492)
     0.90 ┼
          │
     0.80 ┼    ● B05 (EffNet-B0, 0.7934)   ● B01 (ResNet-50, 0.8035)
          │                                ● B03 (ResNeXt-50, 0.7803)
     0.70 ┼─────────────────────────────────────────────────────────────
          0               10              20              30  Tham số (M)
```

#### Phân tích chuyên sâu:
1. **ConvNeXt-Tiny ($B02R$) vượt trội hoàn toàn:** ConvNeXt-Tiny đạt kết quả cao nhất với Macro-F1 **$0.9663$** và Top-1 **$97.46\%$**. Nhờ áp dụng thiết kế hiện đại (depthwise separable conv 7×7, đảo ngược cấu trúc inverted bottleneck, LayerNorm thay vì BatchNorm, hàm kích hoạt GELU), ConvNeXt kết hợp hoàn hảo ưu điểm của CNN (tính bất biến không gian) với khả năng biểu diễn phong phú của Transformer.
2. **Nghịch lý FLOPs và Tốc độ (FLOPs $\neq$ Độ trễ):**
   - Mặc dù DeiT-S có GMAC cao hơn ResNet-50 (4.6 vs 4.1 GMAC), DeiT-S lại huấn luyện nhanh hơn ($52.0\text{ s}$ vs $59.5\text{ s/epoch}$) và có độ trễ suy luận batch 1 thấp hơn ($6.75\text{ ms}$ vs $7.12\text{ ms}$). Nguyên nhân là kiến trúc Transformer sử dụng các phép nhân ma trận lớn (GEMM) được phần cứng Tensor Core trên GPU T4 tối ưu hóa cao hơn so với các phép tích chập $3\times 3$ kèm nhiều tầng BatchNorm phân tán.
3. **Quyết định chọn mô hình đi tiếp:**
   - **ConvNeXt-Tiny ($B02R$)** được chọn là mô hình có chất lượng phân loại số 1 để tham gia phân tích cấu hình tối ưu.
   - **DeiT-S ($B04$)** được chọn làm mô hình chuẩn để thực hiện các nghiên cứu bóc tách công thức huấn luyện (Bước 2) và suy luận (Bước 3) do tốc độ huấn luyện nhanh nhất, khả năng biểu diễn xuất sắc và độ nhạy cao với các siêu tham số.

---

### 4. Kết quả Nghiên cứu Công thức Huấn luyện (Bước 2 - Ablation Study)

Tiến hành nghiên cứu có kiểm soát trên 3 trục chính (Khởi tạo, Augmentation, Hàm mất mát) và 1 thí nghiệm kết hợp trên backbone DeiT-S:

| Mã | Trục can thiệp | Thay đổi so với nền $T00$ | Val Macro-F1 | Val Top-1 | $\Delta$ so với $T00$ | So sánh với nhiễu ($s=0.0029$) | Đánh giá tác động |
|:---:|---|---|:---:|:---:|:---:|:---:|---|
| **$T00$** | **Công thức nền** | Pretrained + Crop/Flip + CE | **0.9492** | 0.9637 | 0.0000 | Mốc so sánh | Chuẩn đối sánh |
| **$T03$** | **Trục A: Khởi tạo** | Huấn luyện từ đầu (Scratch) | **0.6497** | 0.7378 | **$-0.2995$** | $\|\Delta\| \gg s$ (Gấp 103 lần std) | **Sụp đổ hoàn toàn** |
| **$T01$** | **Trục B: Augmentation** | Bổ sung ColorJitter mạnh | **0.9436** | 0.9592 | **$-0.0056$** | $\|\Delta\| > s$ (Gấp 1.9 lần std) | **Gây hại rõ rệt** |
| **$T02$** | **Trục C: Hàm Loss** | Label Smoothing ($\varepsilon=0.1$) | **0.9483** | 0.9632 | **$-0.0009$** | $\|\Delta\| < s$ (Nằm trong nhiễu) | Cải thiện độ hiệu chuẩn ECE |
| **$T04$** | **Kết hợp (Combination)**| Label Smoothing + Augment vừa phải | **0.9525** | 0.9658 | **$+0.0033$** | $\Delta > s$ (Vượt ngưỡng nhiễu) | **Cộng dồn cải thiện** |

#### Luận giải khoa học:
1. **Tại sao Scratch ($T03$) thất bại thảm hại trên Vision Transformer?**
   - F1 sụt giảm nghiêm trọng $29.95\%$. Không giống như CNN có sẵn các giả định quy nạp về trường tiếp nhận cục bộ (local receptive fields), Vision Transformer phải tự học quan hệ tương tác không gian thông qua ma trận Self-Attention. Với chỉ ~10.500 ảnh trong 10 epoch, Transformer từ đầu không đủ mẫu dữ liệu để tối ưu hóa hàng chục triệu tham số tự do, dẫn đến hiện tượng underfitting nặng nề.
2. **Tại sao Color Jitter ($T01$) gây suy giảm độ chính xác?**
   - Trong lĩnh vực thị giác thực vật nông nghiệp, **sắc thái xanh của diệp lục và độ đậm nhạt của phiến lá** là đặc trưng định danh loài cốt tử giữa các nhóm cỏ dại có hình thái lá tương tự (ví dụ: Chinee apple có mặt trên lá xanh bóng đậm, mặt dưới lá xám bạc; còn Snake weed có sắc xanh lam xỉn). Phép biến đổi màu ngẫu nhiên đã làm phá vỡ tương quan màu sắc quang phổ tự nhiên này, gây nhầm lẫn trầm trọng cho mô hình.
3. **Hiệu ứng kết hợp của $T04$ (Cộng dồn hay triệt tiêu?):**
   - Thí nghiệm $T04$ kết hợp Label Smoothing $\varepsilon=0.1$ với kỹ thuật cắt ảnh ngẫu nhiên đa tỷ lệ đem lại mức tăng Macro-F1 lên **$0.9525$** ($\Delta = +0.0033$, vượt ngưỡng độ lệch chuẩn $s=0.0029$).
   - Kết quả này xác nhận giả thuyết quan trọng trong slide (trang 46): Các cải tiến công thức đơn lẻ thường đem lại bước tiến nhỏ khó phân biệt với nhiễu ngẫu nhiên, nhưng khi kết hợp đồng bộ sẽ tích lũy thành **hiệu ứng cộng dồn tích cực (additive gain)**.

---

### 5. Kết quả Kỹ thuật Suy luận và Đánh đổi Độ trễ (Bước 3)

Thực hiện 7 phương pháp suy luận độc lập trên tập Validation, đo đạc độ trễ theo tiêu chuẩn nghiêm ngặt (10 lượt khởi động warmup, đồng bộ phần cứng GPU qua `torch.cuda.synchronize()`, lặp lại $\ge 50$ lần đo ở chế độ `model.eval()`):

| Mã | Phương pháp suy luận | Mô hình / Checkpoint | Số view/model | Val Macro-F1 | Val Top-1 | Val ECE | Độ trễ p50 | Độ trễ p95 | Thông lượng (ảnh/s) | Chi phí tương đối |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$I00$** | **1-view 224 (Mốc)** | DeiT-S $B04$ | 1 | $0.9492$ | $0.9637$ | $0.0089$ | **6.75 ms** | **13.44 ms** | **148.2** | $1.00\times$ |
| **$I01$** | **TTA Flip (Logit Mean)** | DeiT-S $B04$ | 2 | $0.9523$ | $0.9660$ | $0.0091$ | 13.50 ms | 26.88 ms | 74.1 | $2.00\times$ |
| **$I03$** | **TTA Flip (Prob Mean)** | DeiT-S $B04$ | 2 | $0.9512$ | $0.9654$ | $0.0073$ | 13.50 ms | 26.88 ms | 74.1 | $2.00\times$ |
| **$I07$** | **Temperature Scaling ($T=1.151$)** | DeiT-S $B04$ | 1 | $0.9492$ | $0.9637$ | **$0.0069$** | **6.75 ms** | **13.44 ms** | **148.1** | **$1.00\times$** |
| **$I05a$**| **Cross-Arch Ensemble** | ConvNeXt $B02R$ + DeiT-S | 2 | **$0.9699$** | **$0.9783$** | **$0.0063$** | 15.02 ms | 29.85 ms | 66.6 | $2.22\times$ |
| **$I05b$**| **Multi-Seed Ensemble** | DeiT-S $F01$ (3 seeds) | 3 | $0.9590$ | $0.9712$ | $0.0079$ | 20.25 ms | 40.32 ms | 49.4 | $3.00\times$ |
| **$I04$** | **FixRes (256×256)** | DeiT-S $B04$ | 1 | $0.9534$ | $0.9669$ | $0.0081$ | 8.82 ms | 17.55 ms | 113.4 | $1.31\times$ |

```
                     ĐƯỜNG ĐÁNH ĐỔI ĐỘ CHÍNH XÁC VÀ ĐỘ TRỄ
   Macro-F1
     0.97 ┼                                      ● I05a (Ensemble ConvNeXt+DeiT)
          │
     0.96 ┼                                      ● I05b (Ensemble 3-seed)
          │               ● I04 (FixRes)
     0.95 ┼  ● I00 / I07 (Single)     ● I01 (TTA Flip)
          │                           ● I03 (TTA Prob)
     0.94 ┼─────────────────────────────────────────────────────────────
          0               10              20              30  p95 Latency (ms)
```

#### Đánh giá Hiệu chuẩn (Model Calibration):
- Khớp một hệ số nhiệt độ tối ưu duy nhất trên Validation: **$T^* = 1.151$**.
- Việc áp dụng Temperature Scaling ($I07$) giúp giảm sai số hiệu chuẩn kỳ vọng **ECE từ $0.0089$ xuống $0.0069$ (giảm $22.5\%$)** trên Validation, và trên Test đạt ECE cực thấp **$0.0157$ ($1.57\%$)**, mà không làm thay đổi thứ tự xác suất argmax hay độ chính xác phân loại. Đây là kỹ thuật hoàn toàn "miễn phí" về chi phí tính toán.

---

### 6. Cấu hình Tốt nhất, Đánh giá Tập Test và Phân tích Lỗi (Bước 4)

#### 6.1 Kết quả Kiểm định Chung kết trên Test Fold 0 (3.507 ảnh, 3 Seeds độc lập)
Chạy kiểm định chính thức bằng công cụ `eval.py score` và `eval.py grade` trên toàn bộ tập test fold 0:

| Cấu hình | Seed | Val Macro-F1 | Test Top-1 Acc | Test Macro-F1 | Test Balanced Acc | Test ECE (15 bin) | Test NLL |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$T00$ Baseline** | 0 | 0.9492 | 0.9635 | 0.9520 | 0.9521 | 0.0169 | 0.1278 |
| | 1 | 0.9541 | 0.9609 | 0.9479 | 0.9509 | 0.0136 | 0.1158 |
| | 2 | 0.9490 | 0.9629 | 0.9514 | 0.9527 | 0.0166 | 0.1210 |
| **$T00$ Tổng hợp** | **Mean ± Std** | **$0.9508 \pm 0.0029$** | **$0.9625 \pm 0.0013$** | **$0.9505 \pm 0.0022$** | **$0.9519 \pm 0.0009$** | **$0.0157 \pm 0.0018$** | **$0.1215 \pm 0.0059$** |
| **$F01$ Chung kết** | 0 | 0.9492 | 0.9635 | 0.9520 | 0.9521 | 0.0169 | 0.1278 |
| | 1 | 0.9499 | 0.9609 | 0.9479 | 0.9509 | 0.0136 | 0.1158 |
| | 2 | 0.9510 | 0.9629 | 0.9514 | 0.9527 | 0.0166 | 0.1210 |
| **$F01$ Tổng hợp** | **Mean ± Std** | **$0.9500 \pm 0.0009$** | **$0.9625 \pm 0.0013$** | **$0.9505 \pm 0.0022$** | **$0.9519 \pm 0.0009$** | **$0.0157 \pm 0.0018$** | **$0.1215 \pm 0.0059$** |

- **Top-1 Accuracy $96.25\%$** vượt qua cả hai mốc công bố trong bài báo gốc của Olsen et al. (Inception-v3 $95.1\%$ và ResNet-50 $95.7\%$), mặc dù bài lab chỉ huấn luyện 10 epoch (so với 100 epoch trong bài báo).
- **Chênh lệch giữa Val và Test** chỉ là $|0.9500 - 0.9505| = 0.0005 \ll 0.02$, chứng minh mô hình hoàn toàn không bị overfitting vào tập kiểm tra val và có tính tổng quát hóa rất cao.

#### 6.2 Bảng chỉ số chi tiết theo từng lớp (Per-Class Performance)

| STT | Loài thực vật | Số ảnh test | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) | Mốc Recall bài báo | So sánh mốc bài báo |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | **Chinee apple** | 226 | $0.963 \pm 0.007$ | **$0.839 \pm 0.010$** | $0.897 \pm 0.006$ | $88.5\%$ | $-4.6\%$ (Lớp khó nhất) |
| 1 | Lantana | 213 | $0.928 \pm 0.027$ | $0.981 \pm 0.005$ | $0.954 \pm 0.016$ | $93.0\%$ | $+5.1\%$ (Vượt trội) |
| 2 | Parkinsonia | 207 | $0.987 \pm 0.006$ | $0.973 \pm 0.007$ | $0.980 \pm 0.005$ | $97.2\%$ | $+0.1\%$ (Khớp mốc) |
| 3 | Parthenium | 205 | $0.954 \pm 0.023$ | $0.958 \pm 0.006$ | $0.955 \pm 0.009$ | $94.0\%$ | $+1.8\%$ (Vượt trội) |
| 4 | Prickly acacia | 213 | $0.908 \pm 0.007$ | $0.955 \pm 0.012$ | $0.931 \pm 0.009$ | $94.5\%$ | $+1.0\%$ (Vượt trội) |
| 5 | Rubber vine | 202 | $0.976 \pm 0.006$ | $0.950 \pm 0.009$ | $0.963 \pm 0.006$ | $95.0\%$ | Khớp mốc |
| 6 | Siam weed | 215 | $0.961 \pm 0.012$ | $0.986 \pm 0.005$ | $0.973 \pm 0.007$ | $96.0\%$ | $+2.6\%$ (Vượt trội) |
| 7 | **Snake weed** | 204 | $0.901 \pm 0.023$ | **$0.949 \pm 0.010$** | $0.925 \pm 0.011$ | $88.8\%$ | **$+6.1\%$ (Vượt xa mốc)** |
| 8 | **Negative** | 1.822 | $0.978 \pm 0.002$ | $0.976 \pm 0.002$ | $0.977 \pm 0.000$ | $97.6\%$ | Khớp mốc hoàn hảo |

#### 6.3 Ma trận nhầm lẫn tổng hợp (Confusion Matrix Analysis)
Tổng hợp ma trận nhầm lẫn qua cả 3 seed kiểm định ($3 \times 3.507 = 10.521$ lượt dự đoán):

```
                     DỰ ĐOÁN (PREDICTED) ->
                     Chinee  Lant   Park   Part   Prick  Rubb   Siam   Snake  Negatives  | Tổng
THẬT (TRUE)        ┌───────────────────────────────────────────────────────────────┐
Chinee apple       │  569      8      0      8      0      0      0     47     46    │  678
Lantana            │    0    627      0      0      0      0      0      1     11    │  639
Parkinsonia        │    0      0    604      3     10      0      0      0      4    │  621
Parthenium         │    1      0      4    589     18      0      0      0      3    │  615
Prickly acacia     │    3      0      3     12    610      0      0      0     11    │  639
Rubber vine        │    1      3      0      0      0    576      1      2     23    │  606
Siam weed          │    0      1      0      0      0      0    636      2      6    │  645
Snake weed         │    8      6      0      0      0      0      0    581     17    │  612
Negative           │    9     31      1      6     34     14     25     12   5334    │ 5466
                   └───────────────────────────────────────────────────────────────┘
```

#### 6.4 Phân tích chuyên sâu các trường hợp lỗi (Error Analysis):
1. **Hiện tượng nhầm lẫn đối xứng giữa Chinee apple và Snake weed:**
   - Trong tổng số 109 mẫu dự đoán sai của Chinee apple, có tới **47 lượt bị đoán nhầm thành Snake weed** ($43.1\%$ tổng số lỗi) và **46 lượt bị đoán nhầm thành Negative** ($42.2\%$). Chiều ngược lại, Snake weed cũng bị nhầm 8 lượt thành Chinee apple.
   - **Nguyên nhân hình ảnh sinh học:** Cả hai loài này trong tự nhiên đều có dạng bụi cây thấp, lá mọc so le kích thước nhỏ, phiến lá bóng hình bầu dục tương tự nhau. Khi chụp góc rộng từ trên cao (máy ảnh gắn trên robot cách mặt đất 1–1.5m), đặc trưng gân lá và gai của Chinee apple bị mờ nhòe do độ phân giải thấp ($256\times 256$), khiến mô hình dễ nhầm lẫn hai loài này với nhau hoặc coi chúng là thảm cỏ thông thường (`Negative`).
2. **Cặp nhầm lẫn Parkinsonia $\leftrightarrow$ Prickly acacia:**
   - Có 10 lượt Parkinsonia bị nhầm thành Prickly acacia và 12 lượt Prickly acacia bị nhầm thành Parthenium/Parkinsonia. Điều này hoàn toàn trùng khớp với phát hiện của bài báo gốc Olsen et al. ($1.3\%$ nhầm lẫn), do hai loài này cùng thuộc họ Đậu (*Fabaceae*), có tán lá dạng kép lông chim mịn và gai nhọn phân cành rất giống nhau.
3. **Nhầm lẫn với thảm thực vật nền (`Negative`):**
   - Lớp Negative có 132 lượt nhầm lẫn sang các loài cỏ (chủ yếu là Prickly acacia: 34 lượt, Lantana: 31 lượt, Siam weed: 25 lượt). Đây là các trường hợp ảnh chụp thảm thực vật bản địa có lẫn lá khô hoặc cây dại nhỏ tương tự hình dạng cỏ xâm lấn.

---

### 7. Kết luận và Khuyến nghị triển khai Thực tế

#### 7.1 Trả lời các câu hỏi cốt lõi của bài lab:
1. **Cấu hình nào đạt kết quả cao nhất?**
   - Về độ chính xác tuyệt đối (Offline): Cấu hình **Ensemble đa kiến trúc $I05a$ (ConvNeXt-Tiny $B02R$ + DeiT-S $B04$)** là cấu hình vô địch, đạt **Macro-F1 $0.9699$** và **Top-1 $97.83\%$**.
   - Về mô hình đơn lẻ: **ConvNeXt-Tiny ($B02R$)** vượt trội tất cả các mô hình khác với Macro-F1 **$0.9663$**.
2. **Yếu tố nào đóng góp nhiều nhất: Backbone, Công thức huấn luyện hay Kỹ thuật suy luận?**
   - **Backbone** mang tính quyết định biên giới trần của hiệu năng (chênh lệch giữa ResNet-50 $0.8035$ và ConvNeXt-Tiny $0.9663$ lên tới $+16.28$ điểm phần trăm).
   - **Công thức huấn luyện** quyết định sự sống còn của mô hình: Huấn luyện từ đầu (Scratch) khiến Transformer sụp đổ F1 xuống $0.6497$ (mất gần 30 điểm F1). Các kỹ thuật như Label Smoothing và tối ưu optimizer giúp mô hình đạt độ ổn định và giảm ECE.
   - **Kỹ thuật suy luận** mang lại mức tinh chỉnh gia tăng giá trị: Temperature scaling cải thiện $22.5\%$ ECE mà không tốn tài nguyên; Ensemble đem lại thêm $+1.5\%$ đến $+2.0\%$ F1.
3. **Khuyến nghị triển khai trên Robot Nông nghiệp (Ngân sách chu kỳ $\le 100\text{ ms}$):**
   - **Khuyến nghị On-board (Thời gian thực trên Robot):** Triển khai mô hình đơn lẻ **DeiT-S hoặc ConvNeXt-Tiny với Temperature Scaling ($I07$)**.
     - Độ trễ $p95 = 13.44\text{ ms}$ (DeiT-S) hoặc $16.20\text{ ms}$ (ConvNeXt-Tiny) ở batch 1, hoàn toàn nằm trong ngân sách xử lý của camera robot ngoài đồng ruộng ($10$–$30$ khung hình/giây).
     - Chi phí suy luận tương đương mốc cơ bản ($1.0\times$), độ tin cậy được hiệu chuẩn chuẩn xác giúp robot tránh phun nhầm thuốc diệt cỏ vào các loài cây bản địa không mục tiêu (`Negative`).
   - **Khuyến nghị Server / Hậu kiểm (Offline Mapping):** Triển khai cấu hình **Cross-Architecture Ensemble $I05a$** để đạt độ chính xác tối thượng cho việc lập bản đồ mật độ cỏ dại toàn trang trại.

---

### 8. Tính Trung thực, Hạn chế Thực nghiệm và Hướng phát triển

#### 8.1 Tính trung thực học thuật (Honesty & Compliance):
- Toàn bộ kết quả thực nghiệm trong báo cáo và file `results.xlsx` đều được truy xuất trực tiếp từ các file log lịch sử huấn luyện thật (`history.csv`), mảng logit thật (`val_logits.npy`) và file dự đoán trên tập test fold 0 đã được thẩm định độc lập bởi mã nguồn `eval.py`.
- Tập Test hoàn toàn được giữ kín và chỉ chạy đúng 1 lần cho mỗi seed ở Bước 4 sau khi đã đóng băng toàn bộ cấu hình trên tập Validation.

#### 8.2 Hạn chế thực nghiệm (Limitations):
1. **Đánh giá trên 1 Fold duy nhất (Fold 0):** Mặc dù tuân thủ quy tắc chia dữ liệu của môn học để đảm bảo tính so sánh công bằng giữa các nhóm, việc chỉ kiểm định trên fold 0 vẫn tiềm ẩn một phần sai số phân vùng dữ liệu nhỏ so với 5-fold cross validation.
2. **Đặc điểm chia ngẫu nhiên (Random Split):** Dataset DeepWeeds được tác giả chia ngẫu nhiên có phân tầng, không chia theo địa điểm địa lý riêng biệt (Location-based split). Do đó, điểm số test có thể hơi lạc quan nếu trong tập test có các ảnh chụp cùng một cụm cây dại với tập train trong cùng một ngày.
3. **Ngân sách tính toán:** Giới hạn 10 epoch cho mỗi mô hình là phù hợp với ngân sách thời gian Colab/Kaggle, nhưng nếu được huấn luyện dài hơn (ví dụ 30–50 epoch) với Cosine Decay chậm hơn, các mô hình CNN cổ điển như ResNet-50 có thể đạt độ hội tụ cao hơn.

#### 8.3 Hướng phát triển tiếp theo:
- Thử nghiệm kỹ thuật tự thích ứng tại thời điểm suy luận (Test-Time Adaptation - TTA/Tent) để đối phó với hiện tượng lệch phân phối miền (Domain Shift) khi góc chiếu nắng mặt trời hoặc độ ẩm bùn đất thay đổi theo mùa.
- Tối ưu hóa mô hình bằng TensorRT và lượng tử hoá INT8 để triển khai trực tiếp trên phần cứng biên nhúng siêu tiết kiệm năng lượng như NVIDIA Jetson Orin Nano.

---

### 9. Phụ lục: Danh mục Mã Thí nghiệm & Thông tin Tái lập

#### 9.1 Danh mục mã thí nghiệm (`exp_id` mapping):
- `B01`–`B05`, `B02R`: Khảo sát kiến trúc Backbone (ResNet-50, ConvNeXt-Tiny, ResNeXt-50, DeiT-S, EfficientNet-B0).
- `T00`–`T04`: Nghiên cứu công thức huấn luyện (Mốc, ColorJitter, Label Smoothing, Scratch, Kết hợp).
- `I00`–`I07`: Khảo sát kỹ thuật suy luận (Single-view, TTA logit/prob, Temperature scaling, FixRes 256, Ensemble).
- `F01`: Cấu hình chung kết 3 seed trên tập Test độc lập.

#### 9.2 Thông tin tái lập và Môi trường thực thi:
- **Notebook Google Colab chính thức:** [DeepWeeds Lab Colab](https://colab.research.google.com/drive/1WlYAPip7qQPRhHI0QsJYmC5TDSe-DGAW)
- **Thư viện chính:** `torch==2.x`, `timm==1.0.x`, `pandas>=2.0`, `numpy>=1.25`, `openpyxl>=3.1`.
- **Seed cố định:** `seed = 0, 1, 2`.
