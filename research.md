# Nghiên cứu nhận dạng CAPTCHA — bộ dữ liệu `data/`

Tài liệu này tổng hợp phân tích bộ ảnh CAPTCHA trong thư mục `data/` (thống kê metadata + quan sát trực quan + phân tích định lượng pixel) và đề xuất hướng huấn luyện dựa trên các dự án mã nguồn mở (GitHub) cũng như các mô hình có sẵn trên Hugging Face phù hợp với đặc điểm thực tế của bộ dữ liệu.

---

## 1. Tổng quan bộ dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Số lượng mẫu | **500** ảnh PNG |
| File nhãn | `data/metadata.csv` (cột `filename`, `text`) |
| Kích thước ảnh | **128 × 128** pixel |
| Định dạng | PNG, mode `RGBA` (alpha luôn = 255 → không có pixel trong suốt) |
| Độ dài chuỗi nhãn | **5 ký tự** ở 100% mẫu |
| Nhãn unique | **500 / 500** (không trùng) |
| Ảnh trùng pixel | **0** (md5 hash 500/500 unique) |
| Phân bố màu | Đa kênh, gradient/texture nền (mỗi kênh dùng ~40+ mức) → **không phải nhị phân** |

### 1.1 Phân tích bộ ký tự (alphabet)

Tổng số ký hiệu xuất hiện trong nhãn: **24 ký tự**.

```
3 4 7 9 A C D E F H J K L M N P Q R T U V W X Y
```

**Trải đề bao gồm:**

- **Chữ số** (4 ký tự): `3 4 7 9` — chỉ giữ 4 chữ số dễ phân biệt.
- **Chữ cái in hoa** (20 ký tự): `A C D E F H J K L M N P Q R T U V W X Y`.
- **Cố tình loại bỏ** các ký tự dễ nhầm về thị giác:
  - Chữ số bị bỏ: `0 1 2 5 6 8`
  - Chữ cái bị bỏ: `B G I O S Z` (vì lẫn với `8 6 1 0 5 2`)

> Đây là kiểu CAPTCHA "human-friendly alphabet" rất phổ biến (Cloudflare, Botdetect HiSafe, một số khung Java/.NET tự sinh). Đặc điểm này **giảm độ khó** so với CAPTCHA dùng full `0–9 + A–Z` (62 lớp), giúp mô hình hội tụ nhanh với chỉ 500 mẫu.

### 1.2 Phân bố tần suất ký tự

Dải tần suất chạy từ **87** (`Q`) đến **121** (`3`), tỉ lệ max/min ≈ **1.39** — dữ liệu khá cân bằng, không có lớp hiếm gây bias.

Top 10 ký tự xuất hiện nhiều nhất:

```
3:121  N:118  H:118  E:118  L:116  K:115  W:115  9:114  4:110  C:110
```

### 1.3 Phân tích định lượng ảnh (script `analyze_images.py`)

Trích xuất foreground glyph bằng mặt nạ HSV `S > 80` rồi đo bounding box của text trên cả 500 ảnh:

| Thống kê | x_min | y_min | x_max | y_max | width | height |
|---|---|---|---|---|---|---|
| min | 0 | 0 | 102 | 73 | 83 | 27 |
| mean | 3.8 | 8.8 | 123.1 | 118.1 | 119.3 | 109.3 |
| max | 24 | 54 | 127 | 127 | 127 | 127 |
| p5 | 0 | 0 | 109 | 80 | 95 | 36 |
| p50 | 1 | 1 | 127 | 126 | 125 | 124 |
| p95 | 16 | 46 | 127 | 127 | 127 | 127 |

**Điểm rút ra:**
- Text **trải gần như toàn ảnh**: trung bình chiếm 119×109 trên khung 128×128 (~93% chiều ngang, ~85% chiều dọc).
- Không có khoảng trống lớn ở mép → **không cần auto-crop trước khi train**, đưa thẳng 128×128 vào model là hợp lý.
- Vẫn có một số ảnh text co lại (height đến 27 px) → cần augmentation scale để model robust.

**Edge density (Canny 50/150)**: mean = 0.0492, std = 0.0204 → mức nhiễu trung bình, có line/đốm đan xen nhưng không quá đậm đặc (thường là 4–7% pixel là edge).

**Foreground HSV (sample 100 ảnh)**: H mean=48 std=49, S mean=107 std=39, V mean=182 std=45 → các glyph có **độ bão hòa cao** (saturated), trải rộng toàn dải hue → đúng với quan sát "mỗi ký tự một màu khác nhau".

### 1.4 Đặc trưng phong cách (qua 16 ảnh được quan sát trực tiếp)

| Đặc điểm | Mô tả |
|---|---|
| Bố cục text | Dải giữa ngang, kéo dài gần hết chiều rộng |
| **Màu ký tự** | **Mỗi ký tự một màu rực ngẫu nhiên** (đỏ, navy, xanh lá, tím, vàng, nâu, teal…) |
| Nền | Pastel có texture giống map/địa hình (lý do filename là `map_*`), pha màu xanh lá, hồng đào, xám lavender |
| Nhiễu nền | Đường line trắng/cong cong, vệt nguệch ngoạc, đốm vàng/cam nhỏ rải rác |
| Biến dạng glyph | Xoay (-15° đến +15°), nghiêng, scale không đều |
| **Chồng lấn** | Các ký tự **đè lên nhau** ở mép nét → khó tách ký tự bằng heuristic |
| Outline | Một số glyph có viền cyan/teal nhẹ |
| Render | Pixelated/aliased — gợi ý ảnh được upscale (có lẽ từ 32×32 lên 128×128) |
| Font | Stroke dày, phong cách doodle/handwritten variant của sans-serif |

> **Nhận xét quan trọng**: ký tự chồng lên nhau khiến phương pháp phân vùng cổ điển (segment-then-recognize) thất bại. Bắt buộc dùng end-to-end deep learning. Mặt khác, do **mỗi ký tự một màu khác nhau**, color hue có thể là tín hiệu phân ký tự rất mạnh cho CNN → nên giữ ảnh **RGB**, không convert sang grayscale.

---

## 2. Bài toán & cách phát biểu

- **Đầu vào**: ảnh 128×128 RGB (bỏ kênh alpha rỗng).
- **Đầu ra**: chuỗi 5 ký tự thuộc bảng chữ 24 ký hiệu (24⁵ ≈ 7.96 triệu tổ hợp).
- **Đặc thù**: độ dài cố định ⇒ có thể giải bằng cả 2 hướng:
  1. **Multi-head classification** (đơn giản, mạnh khi length cố định): 5 đầu softmax × 24 lớp.
  2. **Sequence-to-sequence với CTC loss** (CRNN): tổng quát hơn, dễ mở rộng sang chuỗi độ dài thay đổi sau này.
- Vì bbox text trải gần kín ảnh, **không cần** crop chặt; vì chữ chồng lấn, **không nên** thử segment trước.

---

## 3. Đánh giá các dự án mã nguồn mở phù hợp

### 3.1 Hạng A — Khuyến nghị áp dụng trực tiếp

#### ⭐ Keras OCR for CAPTCHA (Keras Examples)
- **URL**: https://keras.io/examples/vision/captcha_ocr/
- **Model có sẵn**: https://huggingface.co/keras-io/ocr-for-captcha
- **Kiến trúc**: CNN (2 block Conv-Pool) → Reshape → 2× Bi-LSTM → Dense → **CTC loss**.
- **Vì sao phù hợp**: dataset gốc của ví dụ này (1040 ảnh CAPTCHA 5 ký tự) cấu trúc rất giống bộ của ta; code chỉ cần đổi `img_width=128, img_height=128` và alphabet là chạy được. Có sẵn checkpoint trên HF.
- **Lưu ý**: kiến trúc CRNN+CTC thiết kế cho ảnh băng dài. Với ảnh vuông 128×128, cần điều chỉnh kích thước feature map ngang/đứng (giảm pooling theo trục H, giữ nhiều time steps theo trục W) hoặc resize ảnh sang 200×50.

#### ⭐ ZiYang-xie/PyCAPTCHA (PyTorch)
- **URL**: https://github.com/ZiYang-xie/PyCAPTCHA
- **Tuyên bố độ chính xác**: > 98% trên CAPTCHA 5 ký tự.
- **Kiến trúc**: CNN backbone + multi-head classifier (5 đầu × N lớp).
- **Vì sao phù hợp**: đúng paradigm fixed-length 5-char, khớp 100% với bộ này. Triển khai gọn nhẹ, dễ tùy biến số lớp, tốt cho ảnh vuông 128×128.

#### ⭐ TrOCR fine-tuned cho CAPTCHA (Hugging Face)
- **anuashok/ocr-captcha-v3**: https://huggingface.co/anuashok/ocr-captcha-v3 — fine-tune từ `microsoft/trocr-base-printed`. Có sẵn code mẫu xử lý RGBA → composite trên nền trắng → đưa vào processor (khớp đúng tình huống RGBA của bộ ta).
- **tomofi/trocr-captcha**: https://huggingface.co/tomofi/trocr-captcha — CER báo cáo: 0.0019 (~99.8%).
- **Vì sao phù hợp**: model đã pretrain trên hàng triệu ảnh chữ in, fine-tune được với 500 mẫu.
- **Hạn chế**: nặng (≥330M params), inference chậm; có thể dư thừa cho bài toán 5 ký tự alphabet rút gọn. Encoder ViT có thể nhạy với pattern chồng lấn vì không có inductive bias 2D như CNN.

### 3.2 Hạng B — Tham khảo / lựa chọn dự phòng

| Dự án | URL | Ghi chú |
|---|---|---|
| airaria/CaptchaRecognition | https://github.com/airaria/CaptchaRecognition | CNN+RNN+Attention/CTC, hỗ trợ chuỗi độ dài thay đổi |
| ArivCR7/CaptchaRecognition | https://github.com/ArivCR7/CaptchaRecognition | CRNN + CTC bằng PyTorch, code ngắn |
| YenLinWu/CRNN_with_CTC_Loss | https://github.com/YenLinWu/CRNN_with_CTC_Loss | Notebook minh họa rõ |
| Graf-J/captcha-crnn-finetuned | https://huggingface.co/Graf-J/captcha-crnn-finetuned | CRNN cho alphanumeric (a-z, A-Z, 0-9), 1–8 ký tự |
| jameskokoska/CAPTCHA-Solver | https://github.com/jameskokoska/CAPTCHA-Solver | Bidirectional LSTM, code dễ đọc |
| DrMahdiRezaei/Deep-CAPTCHA | https://github.com/DrMahdiRezaei/Deep-CAPTCHA | CNN cho captcha alphanumeric |
| arunpatala/captcha.irctc | https://github.com/arunpatala/captcha.irctc | Đạt 98% trên IRCTC bằng deep learning |
| pyimagesearch tutorial | https://pyimagesearch.com/2021/07/14/breaking-captchas-with-deep-learning-keras-and-tensorflow/ | Hướng dẫn end-to-end |

---

## 4. Đề xuất chiến lược huấn luyện

### 4.1 Lộ trình ưu tiên (3 cấp độ)

**Cấp 1 — Baseline nhanh (BẮT ĐẦU TỪ ĐÂY)**

- Kiến trúc: **Multi-head CNN** kiểu PyCAPTCHA, input 128×128×3.
  - Backbone: ResNet-18 (pretrained ImageNet) hoặc CNN tự xây 5 block (Conv → BN → ReLU → MaxPool) đến feature 4×4×256 hoặc 8×8×256.
  - Global pool + Flatten → 5 head `Linear(feat_dim → 24)`.
  - Loss: tổng `CrossEntropy` qua 5 đầu (mỗi đầu trọng số 0.2).
- Đầu ra: predict 5 ký tự độc lập rồi ghép chuỗi.
- Ưu điểm: train cực nhanh (~5–10 phút trên GPU phổ thông), không cần CTC, debug dễ. Với 500 mẫu + augmentation hợp lý, mục tiêu **sequence accuracy ≥ 90%**.

**Cấp 2 — CRNN + CTC**

- Áp dụng pipeline `keras.io/examples/vision/captcha_ocr/`. Khi giữ ảnh vuông 128×128 cần thiết kế lại stride: `(2,2) → (2,2) → (2,1) → (2,1)` để feature map ra 8×32 (32 time steps, đủ cho 5 ký tự + blank).
- Lựa chọn này tổng quát hơn nếu sau này CAPTCHA có chuỗi độ dài thay đổi.

**Cấp 3 — Fine-tune TrOCR**

- Khi muốn đẩy độ chính xác sát 99%+ và chấp nhận model nặng.
- Dùng `anuashok/ocr-captcha-v3` làm checkpoint khởi tạo, fine-tune 5–10 epoch trên 500 mẫu.

### 4.2 Tiền xử lý đề xuất

1. `Image.open(path).convert("RGB")` — bỏ alpha (rỗng).
2. **Giữ màu RGB** (không convert grayscale) vì màu là tín hiệu phân biệt ký tự rất mạnh.
3. Resize tùy kiến trúc:
   - Multi-head CNN: giữ 128×128 hoặc resize 96×96 nếu cần tốc độ.
   - CRNN+CTC: có thể giữ 128×128 (chỉnh stride) hoặc resize sang 160×64.
   - TrOCR: dùng `TrOCRProcessor` (tự xử lý 384×384).
4. Chuẩn hóa: chia 255, hoặc dùng mean/std ImageNet `[0.485, 0.456, 0.406]/[0.229, 0.224, 0.225]` nếu dùng backbone pretrained.

### 4.3 Augmentation đề xuất (rất quan trọng vì chỉ có 500 mẫu)

Dựa trên quan sát thực:

- **Affine nhẹ**: `RandomRotation(±5°)`, `RandomAffine(translate=(0.05,0.05), scale=(0.9,1.1), shear=±3°)` — text ở giữa, không nên dịch quá xa.
- **Color jitter**: `brightness=0.15, contrast=0.15, saturation=0.2, hue=0.05` — vì mỗi ký tự một màu, hue jitter mạnh có thể làm mất tín hiệu màu phân ký tự, nên hue chỉ jitter nhẹ.
- **Gaussian noise** σ=0.01–0.03 và **random erasing** 1 patch nhỏ (16×16) để mô phỏng nhiễu line/đốm.
- **Coarse dropout** patch 4×4 ~5 vị trí — mô phỏng pixelated/aliasing.
- **KHÔNG** dùng horizontal/vertical flip (làm sai chữ).
- **KHÔNG** dùng `ColorJitter(hue=0.5)` mạnh — phá tín hiệu phân biệt ký tự.

### 4.4 Tách dữ liệu

- Train / Val / Test = **400 / 50 / 50** (do nhãn unique và pixel unique → tách ngẫu nhiên seed cố định là an toàn).
- Stratify theo ký tự đầu hoặc theo độ phủ ký tự nếu thấy mất cân bằng.
- Lưu lại `splits.json` với `seed=42` để reproducible.

### 4.5 Chỉ số đánh giá

- **Sequence accuracy** (đoán đúng cả 5 ký tự): chỉ số chính, kỳ vọng ≥ 90% với baseline.
- **Character accuracy** (đúng từng vị trí / 5): chỉ số phụ.
- **CER** (Character Error Rate, edit distance / chiều dài): so sánh khi mở rộng sang CTC.
- Báo cáo confusion matrix mức ký tự để phát hiện cặp dễ nhầm. Dự đoán ưu tiên kiểm tra: `M ↔ N`, `K ↔ X`, `H ↔ K`, `V ↔ Y`, `9 ↔ Q`, `4 ↔ A`, `D ↔ 0` (mặc dù `0` không có), `7 ↔ T`.

### 4.6 Gợi ý cấu hình train

- Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)`.
- Scheduler: `CosineAnnealingLR(T_max=epochs)` hoặc `OneCycleLR`.
- Batch size: 32–64.
- Epoch: 50–100 với early stopping theo val sequence accuracy (patience 10).
- Mixed precision (AMP) để tăng tốc.

---

## 5. Bảng ánh xạ ký tự ↔ chỉ số (label encoding)

```
idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
char: 3  4  7  9  A  C  D  E  F  H  J  K  L  M  N  P  Q  R  T  U  V  W  X  Y
```

Khi dùng CTC cần thêm 1 lớp blank token (idx 24) → tổng số class = **25**.

---

## 6. Kế hoạch các bước tiếp theo

1. **(Optional) Khám phá thêm**: chạy `analyze_images.py` để kiểm tra ảnh "bất thường" có height < 50 px (text co lại bất thường) — quyết định có cần augmentation scale mạnh hơn không.
2. **Triển khai baseline cấp 1** (multi-head CNN): viết file `train_baseline.py`, mục tiêu > 90% sequence accuracy trên test 50 mẫu.
3. Phân tích lỗi: in ra 20 ảnh sai dự đoán đầu tiên + ký tự sai để biết cặp nhầm chủ yếu.
4. Nếu cần đẩy thêm 5–8% nữa → chuyển sang CRNN+CTC hoặc fine-tune TrOCR.
5. Đóng gói: lưu mapping ký tự, ngưỡng confidence, script inference đơn ảnh.

---

## 7. Tóm tắt nhanh

- **Dữ liệu**: 500 ảnh 128×128 RGBA **sạch** (0 trùng pixel, 0 trùng nhãn), nhãn 5 ký tự cố định, bảng chữ 24 ký hiệu (`3 4 7 9 A C D E F H J K L M N P Q R T U V W X Y`), đã loại trừ ký tự dễ nhầm.
- **Phong cách CAPTCHA**: nền pastel kiểu bản đồ, mỗi ký tự một màu rực, ký tự **chồng lấn**, có line/đốm nhiễu nhẹ, glyph bị pixelated. Bbox text **chiếm ~93% × 85%** ảnh ⇒ không cần auto-crop.
- **Khuyến nghị #1**: bắt đầu với **multi-head CNN** (PyCAPTCHA-style), input 128×128×3, 5 đầu softmax × 24 lớp — đơn giản, mạnh cho fixed-length 5.
- **Khuyến nghị #2**: nếu muốn dùng đồ có sẵn → fork `keras-io/ocr-for-captcha` (Keras CRNN+CTC), chỉnh stride cho input vuông + đổi alphabet.
- **Khuyến nghị #3**: muốn precision tối đa → fine-tune `anuashok/ocr-captcha-v3` (TrOCR).

---

## Phụ lục: lệnh chạy phân tích định lượng

```powershell
cd c:\Users\Administrator\Desktop\TrainAI
python analyze_images.py
```

File `analyze_images.py` (đã tạo cùng workspace) tính: bounding box text, hash trùng pixel, foreground HSV, edge density.
