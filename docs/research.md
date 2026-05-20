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


---

## 8. Phân tích chi tiết 20 ảnh thật đầu tiên (`map_00000` → `map_00019`)

Mục đích section này: đối chiếu **quan sát thị giác** với **nhãn ground-truth** trong `metadata.csv` để (a) hiểu rõ kiểu nhiễu thực tế của bộ dữ liệu, (b) tìm các cặp ký tự hay nhầm, (c) rút ra yêu cầu cho synth generator và augmentation.

### 8.1 Bảng tổng hợp 20 ảnh đầu

| File | GT label | Nền | Màu/đặc trưng từng glyph (trái → phải) | Mức chồng lấn | Glyph có nguy cơ đọc sai | Ghi chú nhiễu |
|---|---|---|---|---|---|---|
| `map_00000.png` | `4KTN9` | Xám lavender + đốm vàng nhỏ | 4(teal) · K(đỏ) · T(xanh lá) · N(đỏ) · 9(navy) | **Cao** — `4` và `K` đè khít, `T-N` dính nét | `4` dễ trông như `A`/`H` do nét đè bởi `K`; `N` dễ thành `M`/`H` | Đường line trắng mảnh chéo nền |
| `map_00001.png` | `7UTUP` | Vàng pastel + mảng hồng đào | 7(xanh lá) · U(xanh dương) · T(olive) · U(nâu) · P(teal) | Trung bình — `T-U` chữ giữa dính | `T` ở giữa rất dễ thành `I`/`l` vì nét ngang bị `U` lân cận che | Đường nền cong nhạt |
| `map_00002.png` | `D37JF` | Hồng đào + xám | D(vàng nâu) · 3(đỏ tía) · 7(navy) · J(đỏ) · F(teal) | Thấp — các ký tự khá tách | `7` mảnh có thể thành `1`; `J` chân cong dễ thành `I` | Line trắng cắt qua nền |
| `map_00003.png` | `HTJA9` | Xám lavender + chấm cam | H(xanh lá) · T(teal) · J(navy) · A(teal) · 9(cyan) | Trung bình — `J-A` chân chữ chạm nhau | `H` thiếu nét ngang dễ thành `II`/`N`; `A` không có gạch ngang dễ thành `H` | Đường cong tím-cam mờ phía trên |
| `map_00004.png` | `JX7CL` | Hồng đào + line trắng | J(vàng) · X(xanh dương) · 7(đỏ) · C(nâu vàng) · L(navy) | **Cao** — `X` và `7` đè rất sâu (nét chéo X cắt 7) | `7` bị nuốt gần hoàn toàn bởi `X` → có thể bị bỏ sót; `J` đứng riêng có thể thành `I` | Line gãy chéo qua nền |
| `map_00005.png` | `JYRJX` | Xám lavender + đốm vàng | J(teal) · Y(đỏ) · R(navy) · J(xanh lá) · X(xanh dương) | **Cao** — `Y-R` đè, `R-J` dính | `J` đầu mảnh dễ thành `I`; `Y` có thể thành `V` khi che nét trên; cặp `R-J` cuối có thể đọc thành `RI` | Line cong xám |
| `map_00006.png` | `KK4EK` | Hồng đào + line xám | K(nâu) · K(đỏ tía) · 4(navy) · E(đỏ) · K(xanh dương) | **Rất cao** — 5 chữ ép sát, K-K-4 chồng nhiều | `4` ở giữa có thể thành `A`; cặp `KK` đầu dễ đọc thành `W`/`KX`; `E` không có nét trên rõ dễ thành `F` | Có cong arc xám lớn ở dưới |
| `map_00007.png` | `KWNVJ` | Xám + đường vàng cong | K(teal) · W(xanh lá) · N(navy) · V(tím) · J(olive) | **Cao** — toàn nét nhọn (W,N,V) đè vào nhau | `W` và `N` rất dễ hoán vị (đỉnh ngược nhau); `V-J` cuối dễ thành `Y` | Vài chấm vàng |
| `map_00008.png` | `PY3WU` | Xám sáng + đường tím lượn | P(vàng) · Y(nâu) · 3(olive) · W(teal) · U(đen) | Trung bình — `Y-3` đè | `3` đè bởi `Y` có thể thành `5`/`8` (dù 5,8 không có trong tập); `Y` mất 1 nhánh dễ thành `V` | Line tím-cyan rõ trên nền |
| `map_00009.png` | `TH9TQ` | Xám + chấm cam-xanh | T(navy nhạt) · H(tím) · 9(xanh lá) · T(vàng) · Q(nâu đỏ) | Trung bình | `T` thứ 1 có thể thành `1`/`I`; `Q` không lộ đuôi rõ dễ thành `O` (nhưng O không có trong tập) | Line cong trắng dưới |
| `map_00010.png` | `TJKN9` | Xanh lá pastel + tam giác xám | T(xanh lá đậm) · J(đỏ tía) · K(vàng) · N(xanh dương) · 9(cyan) | **Cao** — `J-K` đè, `K-N` đè, `N` đứng nghiêng | `J` mảnh thành `I`; `K` đè `J` có thể thành `M`; `9` cyan trên nền cyan-pastel có thể bị chìm | Đường line cam mảnh |
| `map_00011.png` | `WELDP` | Xanh lavender đậm + line hồng | W(teal) · E(đỏ tía) · L(xanh lá) · D(navy) · P(tím) | Trung bình — `W-E` đè đầu | Chữ `W` có thể đọc thành `V`/`U` vì nhánh trái mảnh; `D` đè `L` có thể thành `O`/`Q` | Cong hồng trên nền |
| `map_00012.png` | `UPVAP` | Xám lavender + line vàng | U(đỏ) · P(tím) · V(đen) · A(xanh lá) · P(teal) | **Cao** — `P-V-A` chen nhau giữa | `V` đen đè giữa `P-A` có thể thành `Y` hoặc lẫn vào nét `A`; `A` thiếu gạch ngang dễ thành `H` | 2 line vàng cong |
| `map_00013.png` | `FYEVU` | Xám-tím + nhiều chấm li ti | F(navy) · Y(xanh lá) · E(xanh lá đậm) · V(đen) · U(tím) | **Cao** — `E-V` đè rất sâu (V xuyên qua E) | `E` đè bởi `V` có thể đọc thành `B`/`F` (B không có); `Y` mất nhánh dễ thành `V` | Đường gân hồng dưới + đốm cam |
| `map_00014.png` | `Q9DHQ` | Xanh lavender đậm + line xám | Q(đỏ) · 9(vàng cam) · D(đen) · H(navy) · Q(navy) | Trung bình — `D-H` đè | `Q` đầu dễ thành `O`; `9` và `Q` cuối đối xứng có thể nhầm; `H` đè `D` có thể thành `B` | Đốm vàng + line cong |
| `map_00015.png` | `3WCE7` | Hồng đào + xám-xanh dưới | 3(navy) · W(nâu) · C(đỏ tía) · E(tím) · 7(xanh lá) | **Cao** — `3-W` đè, `W-C` đè | `3` đè bởi `W` có thể bị "ăn" thành `B`; `C` đè bởi `W` đọc thành `O`; `E` đè `7` thành `B`/`F` | Line trắng cong |
| `map_00016.png` | `LDWC7` | Xanh lavender + line trắng cong | L(tím) · D(olive) · W(teal) · C(nâu) · 7(cyan) | **Rất cao** — `L-D` đè, `D-W` đè (D bị W xé nét) | `D` đè bởi `W` rất giống `O`/`I`; `L` đè bởi `D` thành `U`; toàn dải này dễ đọc thành `LIWC7` | Line trắng mảnh xuyên ngang |
| `map_00017.png` | `QR939` | Hồng đào + line trắng | Q(đỏ tía đậm) · R(teal) · 9(cam) · 3(đen) · 9(navy) | **Cao** — `9-3` đè (3 đen ăn vào 9 cam) | `3` đen bị `9` cam ăn nét trái rất dễ thành `D`/`B`; `Q-R` đầu cũng dính khít có thể thành `OR` | Line trắng cong dài |
| `map_00018.png` | `R3AWX` | Xám-vàng + line cam | R(hồng cánh sen) · 3(teal) · A(xanh lá) · W(tím) · X(vàng nâu) | Trung bình — `A-W` đè ngọn | `3` teal có thể thành `B` khi mắt nhìn nhanh; `A-W` ghép có thể bị đọc nhầm `AVV` | Mảng vàng và line trắng |
| `map_00019.png` | `WTVRY` | Xám sáng + line vàng chéo | W(tím) · T(teal nhạt) · V(đen) · R(xanh lá) · Y(xanh lá) | Trung bình — `T-V` và `V-R` đè | `V` đen ở giữa có thể đọc thành `Y`; cặp `R-Y` cuối cùng tone xanh giống nhau dễ trộn ranh giới | Line vàng mảnh dài chéo |

> Quy ước "mức chồng lấn": **Thấp** = ranh giới rõ; **Trung bình** = đụng nét nhưng còn phân biệt được; **Cao** = nét đè xuyên qua nhau ≥ 30% chiều rộng glyph; **Rất cao** = >50% pixel của ít nhất 1 glyph bị che bởi glyph khác.

### 8.2 Quan sát rút ra qua 20 ảnh

#### A. Phân bố mức độ khó

| Mức chồng lấn | Số ảnh | Tỉ lệ |
|---|---|---|
| Thấp | 1 (`map_00002`) | 5% |
| Trung bình | 7 | 35% |
| Cao | 10 | 50% |
| Rất cao | 2 (`map_00006`, `map_00016`) | 10% |

**60% mẫu rơi vào nhóm "Cao" trở lên** ⇒ chồng lấn là đặc trưng chủ đạo, không phải edge case. Synth generator **bắt buộc** phải mô phỏng overlap mạnh; nếu chỉ render glyph rời nhau, model train trên synth sẽ collapse khi gặp real.

#### B. Cặp ký tự dễ nhầm — bằng chứng cụ thể

| Cặp dễ nhầm | Quan sát thực tế trên 20 ảnh | Nguy cơ |
|---|---|---|
| `4 ↔ A ↔ H` | `map_00000` (4 đè bởi K), `map_00006` (4 ép giữa K-E), `map_00012` (A không gạch ngang trông như H) | **Rất cao** |
| `M ↔ N ↔ W` | `map_00007` (W,N,V xếp sát), `map_00000` (N đè T), `map_00003` (H thiếu nét) | **Cao** |
| `J ↔ I ↔ 1` | `map_00002`, `map_00005`, `map_00010` (J mảnh, đứng riêng) | Cao |
| `7 ↔ T ↔ 1` | `map_00001` (T giữa hai U), `map_00009` (T đầu mảnh) | Cao |
| `Q ↔ O ↔ 9` | `map_00014` (Q-9-...-Q), `map_00017` (9-3 đè) | Cao (lưu ý `O` không có trong tập) |
| `D ↔ O ↔ 0` | `map_00016` (D bị W xé nét) | Trung bình (`O,0` không có trong tập, giảm rủi ro) |
| `V ↔ Y` | `map_00012`, `map_00007` (V xen giữa các nét), `map_00013` (Y mất nhánh) | Trung bình |
| `E ↔ F ↔ B` | `map_00006` (E mất nét trên), `map_00013` (E bị V xuyên), `map_00015` (E đè 7) | Cao (lưu ý `B` không có) |
| `3 ↔ B ↔ 8` | `map_00017` (3 đen bị 9 ăn), `map_00018` (3 teal trông như B) | Trung bình (`B,8` không có) |
| `K ↔ X` | `map_00000`, `map_00006`, `map_00010` (K đè/lồng vào hàng xóm) | Cao |

> Thiết kế bảng chữ đã loại bỏ `0,1,2,5,6,8,B,G,I,O,S,Z` ⇒ nhiều cặp ở trên về thực tế **không gây sai vì không có target**, nhưng model vẫn cần được "nhắc" rằng các glyph này nằm ngoài alphabet (mask logits ở 24 lớp). Đây là **lợi thế lớn** của bộ dữ liệu này.

#### C. Đặc trưng nền (background)

3 kiểu nền lặp lại trong 20 mẫu:

1. **Pastel gray-lavender + chấm vàng/cam li ti** (`00000, 00007, 00009, 00012, 00013, 00019` → 30%): nền đồng nhất nhất, dễ train.
2. **Hồng đào (peach) + đường line trắng/xám cong** (`00002, 00004, 00005, 00006, 00015, 00016, 00017, 00018` → 40%): xuất hiện nhiều nhất; line trắng có thể nhầm là phần stroke chữ.
3. **Vàng pastel hoặc xanh lá pastel + mảng nhị phân** (`00001, 00010, 00011` → 15%): tone nóng, hue gần với một số glyph, dễ giảm contrast.

⇒ Augmentation `RandomBackgroundCompose` (đè ảnh chữ trong suốt lên 3 kiểu texture trên) là biện pháp giàu giá trị. Nếu synth generator hiện tại chỉ dùng 1 loại nền, **cần bổ sung tối thiểu 3 preset texture**.

#### D. Đặc trưng glyph

- **Stroke dày**, gần như block-style — gợi ý font họ "Marker Felt", "Komika Axis", hoặc tự vẽ shape rồi dilate.
- **Pixelated/aliased** rõ → ảnh có vẻ được render ở 32×32 hoặc 64×64 rồi upscale 128×128 bằng nearest-neighbor. Synth nên render ở **độ phân giải nhỏ** rồi upscale để giống texture aliased.
- **Mỗi glyph một màu HSV bão hòa cao** (S > 100), V trải rộng → khi sinh synth, sample màu từ `HSV(H ~ Uniform[0,180], S ~ Uniform[120,220], V ~ Uniform[100,220])`.
- **Có outline mảnh** ở một số glyph (cyan/teal hoặc đen) → bật `RandomGlyphOutline(width=1, color in {cyan, teal, black})`.
- **Xoay glyph độc lập**: mỗi ký tự có affine riêng `rotate ∈ [-15°, +15°]`, `shear ∈ [-10°, +10°]`, `scale ∈ [0.85, 1.15]`.

#### E. Đặc trưng layout

- Bố cục dải ngang ở giữa, **không thẳng hàng tuyệt đối**: baseline có nhấp nhô ±5–8 px (xem `map_00010`, `map_00018`).
- Khoảng cách giữa các ký tự **âm** trong 60% trường hợp (chồng lấn). Khoảng dương trung bình ước tính: -8 px → +4 px.
- Toàn bộ cụm chữ **center-aligned** (lệch trái/phải tối đa ~10 px).

⇒ Synth nên dùng layout: `x_i = x_{i-1} + glyph_width_{i-1} + offset_i`, `offset_i ~ Uniform[-12, +6]`, sau đó dịch toàn cụm để center.

### 8.3 Bài học cho synth generator (cập nhật `src/synth.py`)

Checklist tối thiểu để synth giống real (xếp theo độ ưu tiên giảm dần):

1. ✅ **Bắt buộc** — overlap mạnh giữa các ký tự (offset âm), tỉ lệ overlap ≥ 60%.
2. ✅ **Bắt buộc** — render rồi upscale nearest-neighbor để có aliasing.
3. ✅ **Bắt buộc** — mỗi glyph 1 màu HSV bão hòa cao, độc lập.
4. ✅ **Bắt buộc** — affine riêng cho từng glyph (rotate/shear/scale nhẹ).
5. 🔶 **Rất nên** — ≥ 3 preset background (pastel-lavender, peach + lines, mint/yellow pastel).
6. 🔶 **Rất nên** — vẽ vài đường line trắng/xám/vàng cong (Bezier ngẫu nhiên) chạy xuyên ảnh.
7. 🔶 **Rất nên** — rải các chấm điểm 1–2 px màu vàng/cam/teal mật độ thấp (~0.3% pixel).
8. 🔷 **Nên** — outline ngẫu nhiên 0/1 px màu cyan/teal/đen ở một số glyph.
9. 🔷 **Nên** — baseline jitter ±5 px theo trục y giữa các ký tự liên tiếp.
10. 🔷 **Nên** — alphabet đúng 24 ký hiệu `3 4 7 9 A C D E F H J K L M N P Q R T U V W X Y` (không lẫn ký tự khác).

### 8.4 Bài học cho augmentation trên ảnh real (cập nhật pipeline train)

Vì dữ liệu real đã có nhiễu nội tại mạnh, augmentation chỉ cần **biến thể nhẹ** — augmentation quá nặng sẽ làm vỡ tín hiệu màu phân ký tự.

| Aug | Khuyến nghị | Lý do |
|---|---|---|
| `RandomRotation` | ±5° (nhẹ) | Glyph đã xoay sẵn, không cần xoay thêm nhiều |
| `RandomAffine(translate)` | (0.03, 0.03) | Text trải gần kín ảnh, dịch nhiều sẽ bị crop |
| `ColorJitter(hue)` | 0.03 (cực nhẹ) | **KHÔNG** dùng hue mạnh — phá tín hiệu màu phân ký tự |
| `ColorJitter(brightness, contrast, saturation)` | (0.15, 0.15, 0.15) | Vừa đủ |
| `RandomErasing` | p=0.3, scale (0.01, 0.04) | Mô phỏng dropout pixel |
| `GaussianNoise σ` | 0.01–0.02 | Đã có nhiễu chấm sẵn, chỉ thêm nhẹ |
| `RandomHorizontalFlip` | **CẤM** | Làm sai chữ |
| `RandomVerticalFlip` | **CẤM** | Làm sai chữ |
| `Cutout 16×16` | p=0.2 | Mô phỏng line/đốm che |

### 8.5 Tổng kết section 8

- **20 ảnh đầu khẳng định 4 đặc trưng then chốt**: chồng lấn mạnh (60% mẫu), nền có line/đốm, glyph mỗi màu một kiểu, render aliased.
- **Cặp ký tự dễ nhầm** chủ yếu nằm ở: `4-A-H`, `M-N-W`, `J-I-1`, `7-T-1`, `K-X`, `E-F` — cần theo dõi confusion matrix sau khi train baseline.
- **Synth generator cần bổ sung**: overlap âm, baseline jitter, ≥3 background preset, line/dot noise, outline glyph.
- **Augmentation real**: nhẹ tay với hue/affine, ưu tiên erasing và noise nhẹ.



---

## 9. Phân tích chi tiết 20 ảnh thật batch 2 (`map_00020` → `map_00039`)

### 9.1 Bảng tổng hợp

| File | GT label | Nền | Màu/đặc trưng từng glyph | Mức chồng lấn | Glyph có nguy cơ đọc sai | Ghi chú nhiễu |
|---|---|---|---|---|---|---|
| `map_00020.png` | `H7K3X` | **MỚI: peach + blue chia chéo (water+land map)**, đốm cam dày | H(navy) · 7(hồng cánh sen) · K(hồng) · 3(navy) · X(xanh lá đậm) | **Rất cao** — `H-7` đè khít, glyph hòa với block nền | `7` đè bởi `H` rất khó tách → có thể bỏ sót, đọc thành `HK3X`; mảng nước lớn ở giữa cắt qua nét | Nền có texture **mảng vuông màu** (water tile) — kiểu nền mới |
| `map_00021.png` | `CRUNX` | Sage green + line hồng cong + line trắng | C(xanh dương) · R(đỏ) · U(cyan, **stroke mảnh**) · N(xanh lá đậm) · X(đỏ tía) | Trung bình | `U` cyan stroke mỏng hơn các chữ khác đáng kể → có thể bị xoá khi blur/down-sample → đọc thành `CR_NX`; line hồng cong qua giữa | **Stroke-width disparity** (không phải height disparity như đoán ban đầu) |
| `map_00022.png` | `NULLH` | Sage green + đường hồng cong + chấm hồng dày | N(xanh lá) · U(đỏ tía) · L(tím) · L(đen) · H(tím) | **Rất cao** — `L-L-H` đè rất sâu, ranh giới biến mất | Cặp `L L` đè đến mức trông như chữ `I` đơn hoặc `H` méo; toàn cụm `LLH` dễ đọc thành `IH`/`UH` | **Ký tự lặp + đè sâu** — ca khó nhất batch |
| `map_00023.png` | `FMMAE` | Xám lavender + line xanh dương cong | F(hồng) · M(xanh lá) · M(đen) · A(teal) · E(xanh dương) | **Cao** — `M-M-A` cụm giữa đè nhau | Hai `M` khác màu nhưng cùng shape — dễ trộn thành 1 chữ `M` rộng; `M-A` kết thúc bằng nét xiên có thể đọc thành `W` | Line cong lớn xuyên qua text |
| `map_00024.png` | `XQJWL` | Xám sáng + line tím cong + mảng tile xám | X(đỏ tía) · Q(đen, **rất to**) · J(teal, **rất nhỏ**) · W(xanh lá) · L(xanh dương) | **Cao** — `X-Q` đè, `J` bị `Q-W` ép | `J` cyan nhỏ ép giữa Q-W gần như biến mất → đọc thành `XQWL`; `Q` to có thể đọc thành `O` | **Disparity về scale rất lớn** giữa Q và J |
| `map_00025.png` | `W3TRC` | Xám lavender + line cam mảnh | W(xanh lá) · 3(đỏ) · T(**lavender/màu sát nền**) · R(xanh lá đậm) · C(xanh lá) | Trung bình | `T` màu **gần trùng nền** → contrast rất thấp, model có thể bỏ sót → đọc thành `W3RC` (4 ký tự) | Cảnh báo: glyph color quá gần bg color |
| `map_00026.png` | `XFYVM` | Xám sáng + line cam + tile xám | X(xanh lá) · F(xanh dương) · Y(hồng) · V(đen + **outline cyan**) · M(tím) | **Cao** — `Y-V-M` chen giữa | `V` đen outline cyan ép giữa Y-M nên có thể đọc thành `XFYM` (4 chữ) hoặc `V` thành `K`/`X`; **chấm đen rời lơ lửng phía trên V** là noise cần phân biệt với dấu chữ | **Noise: blob đen tự do** không thuộc glyph |
| `map_00027.png` | `AV4LL` | Xám lavender phẳng | A(đen + **outline cyan rõ**) · V(xanh lá) · 4(teal) · L(đen) · L(navy) | Thấp-Trung bình | Cặp `LL` cuối đè nhau ít → vẫn tách được; `4` teal gọn | **Outline cyan rõ ràng** — xác nhận glyph có thể có outline ngoài |
| `map_00028.png` | `UWRUH` | Xám sáng + **lưới line vàng dày chéo** | U(nâu) · W(rám nắng) · R(xanh dương) · U(đen) · H(xanh lá) | Trung bình | `U-W` đầu cùng tone vàng-nâu dễ trộn; `U` đen ở vị trí 4 đè vào `H` xanh nên `UH` có thể đọc thành `OH`/`OB` | Lưới line vàng cắt qua text — kiểu nền có nhiễu nét đậm hơn batch 1 |
| `map_00029.png` | `Q9YJA` | Sage green + line hồng-trắng | Q(xanh lá) · 9(đen) · Y(cyan) · J(tím) · A(đỏ) | **Thấp** — các chữ tách rời rõ | (Ít rủi ro) — có thể `9` đen hơi giống `Q`/`O` nhưng tách biệt | Nền sạch, ảnh "dễ" trong batch |
| `map_00030.png` | `Y3Q93` | Sage green + nhiều line hồng đỏ | Y(đen) · 3(teal) · Q(xanh lá) · 9(đỏ) · 3(tím) | **Rất cao** — `3-Q-9-3` cụm cuối đè kinh khủng | `9-3` cuối ăn vào nhau tạo shape `B`/`8` (cả 2 không có trong tập); `Q` xanh lá bị `3` teal ăn nét dễ đọc thành `O` | Nhiều line hồng-đỏ song song qua text |
| `map_00031.png` | `YHTRT` | **MỚI: peach + blue chia chéo** + line tím cong dày | Y(teal) · H(olive) · T(teal nhạt) · R(xanh lá) · T(xanh dương) | **Cao** — `H-T-R-T` cụm dính | Hai `T` khác vị trí, `T` thứ 1 ở giữa rất khó tách; `Y-H` đầu cùng tone xanh dễ trộn | Giống `00020`: nền peach/blue split — pattern nền mới phổ biến hơn nghĩ |
| `map_00032.png` | `3JYER` | Xám sáng + line cyan-vàng cong | 3(xanh lá) · J(teal) · Y(đen) · E(**tím đậm**) · R(**hồng/magenta**) | Trung bình | `J` có nét xiên dễ thành `I`; `E` tím và `R` hồng **khác màu** nên ít risk merge hơn ước đoán; cặp cuối có ranh giới rõ | Đính chính: E và R không cùng tone hồng |
| `map_00033.png` | `ML4CT` | **MỚI: khaki/vàng đậm phẳng** + line xám đậm | M(xanh dương) · L(đỏ) · 4(xanh lá) · C(đen + **outline cyan**) · T(tím) | Trung bình | `M-L` đầu khít, `M` xanh có thể bao trùm `L` → đọc thành `Ml4CT` hoặc `MN4CT` | Nền vàng đậm khác hẳn — preset mới |
| `map_00034.png` | `NFWRE` | Xám sáng + **2 line vàng đậm chéo X lớn** | N(đen) · F(đỏ) · W(đen + **outline cyan rõ**) · R(tím) · E(xanh dương) | **Cao** — `F-W-R` cụm giữa đè | `W` cyan-outlined có nét nhọn xen `F-R` nên đọc thành `M`/`MR`; `N-F` đầu cùng tối có thể merge | Line vàng X lớn — chú ý có thể nhầm là nét chữ |
| `map_00035.png` | `QHXVH` | Xám lavender + line teal nhạt cong | Q(tím) · H(nâu) · X(xanh lá) · V(navy) · H(navy) | **Cao** — `X-V-H` cuối cùng tone navy ngấu | `V-H` cuối **cùng màu navy** + đè khít → V có thể bị bao bởi H đọc thành `QHXH` (4 chữ); cặp `X-V` cũng dễ thành `K`/`Y` | Cảnh báo: đồng màu liên tiếp xảy ra |
| `map_00036.png` | `QWMJL` | Xám lavender + line xanh lá cong + mảng peach dưới | Q(tím) · W(xanh lá) · M(đen) · J(vàng) · L(xanh dương) | **Cao** — `W-M-J` cụm giữa đè rất sâu | `W-M` đè đến mức trông như chữ `M` đôi lớn; `M-J` đè khiến `J` ngắn dễ thành `I` | Nền có 2 lớp (xám trên + peach dưới) |
| `map_00037.png` | `NJWND` | **MỚI: peach + sage chia chéo** | N(xanh lá) · J(vàng) · W(teal) · N(đỏ) · D(navy) | **Rất cao** — `J-W-N` cụm giữa đè cực sâu | `J` vàng + `W` teal merge thành shape không định danh; toàn cụm dễ đọc thành `NWND` (mất J) | Nền pha trộn 2 tone — preset mới |
| `map_00038.png` | `T9LFR` | **MỚI: vàng pastel phẳng** + line hồng-cyan mảnh | T(hồng + **outline cyan mảnh**) · 9(đen) · L(olive) · F(tím) · R(xanh lá) | **Thấp** — chữ tách rời gần đều | (Ít rủi ro) — `9` đen có thể nhầm `Q` nhưng đứng riêng; `T` có outline mỏng | Ảnh "dễ" thứ 2 batch — nền sạch + tách chữ; xác nhận outline cyan rất phổ biến |
| `map_00039.png` | `ETETY` | Xám sáng + line vàng cong + line xám | E(đen + **outline cyan**) · T(xanh dương) · E(cam-mustard) · T(tím) · Y(cam) | **Cao** — pattern lặp `E-T-E-T-Y`, baseline nhấp nhô | `T` cuối có thể bị mất giữa cụm; `T` xanh dương và `T` tím khác màu giúp tách nhưng pattern repeat khó cho model; `Y` cam đứng riêng OK | **Baseline nhấp nhô per-glyph** (không phải xoay đồng bộ) — đính chính: không phải string-tilt |

### 9.2 Quan sát mới so với batch 1

#### A. Phân bố mức độ khó (batch 2)

| Mức chồng lấn | Số ảnh | Tỉ lệ |
|---|---|---|
| Thấp | 2 (`00029`, `00038`) | 10% |
| Thấp-Trung bình | 1 (`00027`) | 5% |
| Trung bình | 7 | 35% |
| Cao | 7 | 35% |
| Rất cao | 3 (`00020`, `00022`, `00030`, `00037`) | 15%* |

(*đếm ra 4 ảnh "Rất cao" ⇒ batch 2 khó hơn nhẹ ở đầu phân bố)

**Cộng dồn batch 1+2 (40 ảnh)**: ~50% rơi vào Cao+Rất cao → khẳng định lại rằng overlap mạnh là đặc trưng phổ thông, không phải tail.

#### B. Phát hiện nền MỚI (chưa thấy ở batch 1)

| Mã preset | Mô tả | Ví dụ batch 2 | Ước tỉ lệ trong tập |
|---|---|---|---|
| **BG-D** | Peach + blue chia chéo (mô phỏng water+land map tile, có ô vuông xanh) | `00020`, `00031`, một phần `00037` | ~10% |
| **BG-E** | Khaki/vàng đậm phẳng (solid yellow-olive) | `00033` | ~3-5% |
| **BG-F** | Vàng pastel phẳng + line mảnh (giống bg vàng nhạt batch 1 nhưng đậm hơn) | `00038`, một phần `00033` | ~5% |

⇒ Tổng hiện tại đã đếm được **6 preset nền** (3 từ batch 1 + 3 mới batch 2). Synth generator nên có ít nhất **6 background presets** để cover phân bố.

#### C. Hiện tượng MỚI cần ghi nhận

1. **Outline cyan rõ ràng** xuất hiện ở nhiều glyph trong batch 2: `A` (`00027`), `C` (`00033`), `W` (`00034`), `E` (`00039`). Tần suất ước tính 15-25% ảnh có ít nhất 1 glyph viền cyan/teal. Đây không còn là "hiếm" như suy đoán batch 1 → cần bật `RandomGlyphOutline(p=0.2)` cho synth.

2. **Letter-scale disparity rất lớn**: 
   - `00021`: U cao chỉ ~50% các glyph khác.
   - `00024`: Q to gấp ~1.6× J.
   - Ngược lại, các ảnh khác glyph có scale tương đối đồng đều.
   → Synth nên có 2 chế độ: "uniform scale" (80% mẫu) và "random scale per-glyph với σ lớn" (20% mẫu).

3. **Glyph color sát màu nền** (`00025` T lavender trên nền lavender): đây là **failure mode** đáng sợ nhất vì model không thể phân biệt bằng pixel. Nếu tập sinh synth, **cần có constraint** `|color_glyph - color_bg|_HSV > threshold` để tránh case patological — hoặc giữ nguyên để robust hơn (cần test).

4. **Đồng màu giữa các glyph liên tiếp** (`00035` V-H đều navy, `00031` Y-T đều teal nhạt): trộn ranh giới ở chỗ glyph đè nhau → mất khả năng dùng "color cluster" để segment. Synth nên cố ý cho phép 2 glyph kế tiếp cùng hue (xác suất ~5-10%) để model không lệ thuộc vào tín hiệu màu.

5. **Rotation toàn cụm chữ** (`00039` cả string ETETY nghiêng ~+15° đi lên): batch 1 mỗi glyph xoay độc lập, batch 2 có ca toàn-string xoay. Synth cần thêm tham số `string_tilt ∈ Uniform[-12°, +12°]` áp lên cả layout chứ không chỉ từng glyph.

6. **Ký tự lặp đè sâu**: `NULLH` (LL ăn vào H), `Y3Q93` (3...3), `ETETY` (T-E-T-E), `FMMAE` (MM), `AV4LL` (LL). Khi 2 ký tự giống nhau đứng cạnh, generator hiện hữu có thể vô tình render thành 1 hình duy nhất. Synth phải đảm bảo:
   - Khi `c[i] == c[i+1]`, ép màu khác nhau **đáng kể** (∆hue ≥ 60°)
   - HOẶC ép offset dương (không overlap) nếu cùng màu
   → Multi-head model 5×24 không có vấn đề với ký tự lặp; **CTC** sẽ cần blank token tách → ưu tiên multi-head ở giai đoạn baseline.

#### D. Cập nhật cặp ký tự dễ nhầm (sau 40 ảnh)

| Cặp | Bằng chứng (batch 1+2) | Mức rủi ro cập nhật |
|---|---|---|
| `4 ↔ A ↔ H` | `00000, 00006, 00012, 00027` (A có outline cyan), `00033` | **Rất cao** (giữ nguyên) |
| `M ↔ N ↔ W` | `00007, 00023, 00036, 00037, 00034` (W có outline cyan) | **Rất cao** ↑ (tăng từ Cao) |
| `J ↔ I ↔ 1` | `00002, 00005, 00010, 00021` (U scale nhỏ giống J), `00024` (J ép giữa) | **Cao** (giữ) |
| `T ↔ 7 ↔ 1` | `00001, 00009, 00025` (T pale chìm) | **Cao** ↑ (lý do mới: contrast quá thấp) |
| `Q ↔ O ↔ 9 ↔ 0` | `00014, 00017, 00024, 00029, 00030` | **Cao** (O,0 không có giúp giảm) |
| `V ↔ Y ↔ K ↔ X` | `00007, 00012, 00026, 00035` (V-H cùng màu navy) | **Cao** ↑ |
| `E ↔ F ↔ B ↔ R` | `00006, 00013, 00015, 00032` (E-R cùng màu hồng), `00039` | **Cao** (R thêm vào nhóm) |
| `3 ↔ 8 ↔ B` | `00017, 00030` (3-Q-9-3 cụm), `00018` | Trung bình (8,B không có) |
| `K ↔ X` | `00000, 00006, 00010, 00020, 00035` | Cao |
| **MỚI**: `L ↔ I ↔ 1` (do scale jitter) | `00021` (U nhỏ giống L), `00022` (LL→I), `00027` (LL cuối) | **Cao** |
| **MỚI**: `D ↔ B ↔ O` (đè bị che) | `00016, 00037` (D đè bởi N) | Trung bình |

### 9.3 Đặc trưng tổng hợp sau 40 ảnh

| Đặc trưng | Tỉ lệ ước tính | Hệ quả cho synth |
|---|---|---|
| Overlap Cao+ | ~50% | Bắt buộc render với offset âm |
| Có line nền (cong/X/cong dài) | ~70% | 1-3 line ngẫu nhiên/ảnh |
| Có chấm noise màu | ~80% | Rải dot mật độ 0.3-1% pixel |
| Glyph có outline | ~20% | `RandomGlyphOutline(p=0.2)` |
| Ký tự đứng độc lập (overlap thấp) | ~15% | Cho phép offset dương 5-10% mẫu |
| Toàn cụm chữ nghiêng | ~5% | `string_tilt ~ Uniform[-12,+12]°` |
| Letter scale disparity mạnh | ~15% | `RandomScalePerGlyph(σ=0.15)` |
| Glyph color quá sát bg | ~5% | Optional: bỏ constraint để model robust |
| Ký tự liên tiếp cùng hue | ~5-8% | Cho phép có chủ ý |

### 9.4 Cập nhật checklist synth (mở rộng từ 8.3)

Bổ sung vào danh sách 10 mục ban đầu:

11. 🔶 **Rất nên** — thêm preset BG-D (peach+blue split với water tiles) — chiếm ~10% real.
12. 🔶 **Rất nên** — thêm preset BG-E/F (khaki/vàng đậm phẳng) — chiếm ~5-8%.
13. 🔷 **Nên** — `RandomGlyphOutline(p=0.2, color in {cyan, teal, black}, width=1-2px)` — tần suất outline trong real đã được xác nhận.
14. 🔷 **Nên** — `RandomStringTilt(p=0.05, angle_deg=[-12,+12])` — toàn cụm chữ cong nghiêng.
15. 🔷 **Nên** — `RandomPerGlyphScale(p=0.15, scale_range=[0.55, 1.4])` — disparity scale lớn.
16. 🔷 **Nên** — khi `label[i] == label[i+1]`: ép `|hue[i] - hue[i+1]| ≥ 60°` để model không học shortcut "ký tự lặp = 1 hình".
17. 🔶 **Rất nên** — line noise nền: lưới X chéo (như `00028`, `00034`), arc cong dài (như `00022`, `00032`) — không chỉ line đơn.
18. 🔷 **Nên** — option "cùng hue cho 2 glyph kế tiếp" với p=0.05 để chống lệ thuộc tín hiệu màu.
19. 🔷 **Nên** — option "glyph color sát bg" với p=0.03 để robust với case `00025`.

### 9.5 Tổng kết section 9

- **40 ảnh đầu** xác nhận ~50% mẫu rơi vào nhóm chồng lấn Cao+, không phụ thuộc vị trí trong tập.
- **6 preset background** đã được xác định: 3 từ batch 1 (lavender-dot, peach-line, mint/yellow-pastel) + 3 mới batch 2 (peach/blue-split, khaki-flat, yellow-flat).
- **Outline cyan/teal** không hiếm như nghĩ — xuất hiện ~20% mẫu → tăng `p` trong synth.
- **Letter scale disparity** và **string-level rotation** là 2 tính năng synth cần bổ sung mới.
- **Ký tự lặp** (`LL`, `MM`, `TT`, `33`, `99`...) xuất hiện đủ nhiều để chứng minh multi-head **ưu việt hơn CTC** ở phase 1 (CTC cần blank tách, dễ hỏng khi 2 ký tự lặp quá dính).
- **Failure mode đáng lo nhất**: `T9LFR` thì OK, nhưng `00025` (T màu sát nền) cảnh báo synth không nên ép contrast quá thấp một cách hệ thống.


## 10. Phân tích chi tiết 20 ảnh thật batch 3 (`map_00040` → `map_00059`)

> Phân tích trực tiếp từ pixel 20 ảnh batch 3, đối chiếu nhãn `metadata.csv`. Mục tiêu: xác nhận pattern phát hiện ở batch 1+2 có ổn định không, đồng thời tìm thêm hiện tượng mới (đặc biệt: ký tự lặp 3 lần, glyph hình lowercase, layout dồn-trái/dồn-phải).

### 10.1 Bảng tổng hợp 20 ảnh batch 3

| File | GT label | Nền | Màu/đặc trưng từng glyph | Mức chồng lấn | Glyph có nguy cơ đọc sai | Ghi chú nhiễu |
|---|---|---|---|---|---|---|
| `map_00040.png` | `3YDKH` | Sage green + chấm peach + line xám/peach mỏng | 3(nâu cam) · Y(hồng) · D(tím đậm) · K(xanh lá đậm) · H(xanh dương nhạt) | **Cao** — `3-Y-D` đè khít, `D-K` đè vừa | `D-K` đè khiến `D` mất nét cong phải → có thể đọc thành `O`/`P`; `Y` ép giữa 3-D dễ thành `T` | Line peach mảnh chéo + line xám cong dưới |
| `map_00041.png` | `NA7HR` | Xám sáng + arc cyan-trắng cong | N(tím đậm) · A(xanh dương rực) · 7(đỏ đậm) · H(xanh lá) · R(đen) | **Cao** — `A-7-H` cụm giữa đè sâu, `7` ăn vào `A` và `H` | `7` đỏ với nét xiên trùng nét xiên `A` → có thể đọc thành `NAHHR` (dropped 7) hoặc `7→T`; `H` xanh lá có vẻ thiếu nét ngang | Arc trắng cong cắt qua phần trên text |
| `map_00042.png` | `LAD4V` | **MỚI: peach đậm + arc lavender mảnh** | L(đỏ đậm) · A(olive/vàng-nâu) · D(tím) · 4(teal) · V(navy) | **Rất cao** — `A-D-4-V` cụm giữa đè 3 lớp | `A` ép giữa `L-D` mất nét trên → `LAD` có thể đọc thành `L4D` hoặc `LDD`; `4-V` đè khít (V navy + 4 teal cùng tone xanh-lạnh) → dễ thành `H` hoặc `A` | Arc lavender song song dưới text + arc xám trên |
| `map_00043.png` | `W4PDT` | **BG-D: peach+blue split (water/land tile)** + line tím cong | W(đen + **outline cyan rõ**) · 4(đỏ đậm) · P(teal) · D(xanh lá đậm) · T(xanh lá nhạt) | **Cao** — `W-4` đè, `P-D-T` cụm cuối đè | `4` đỏ ép giữa `W-P` có thể bị nuốt → đọc thành `WPDT` (4 chữ); `D-T` cuối đè khít (cùng họ xanh lá) dễ trộn ranh giới | Nền water/land tile xác nhận lại preset BG-D từ batch 2 |
| `map_00044.png` | `XWN97` | Sage green + mảng peach trên + line cyan-cam mảnh | X(đỏ đậm) · W(xanh lá) · N(tím đậm) · 9(xanh lá nhạt) · 7(teal) | **Rất cao** — `X-W-N-9-7` toàn cụm đè liên tục, ranh giới rất khó | `W-N` đè (cùng họ xanh) thành 1 khối M-shape rộng → có thể đọc `XMN97` hoặc `XWNQ7`; `9` xanh nhạt có thể nhầm `Q`/`O`; layout text **lệch sang trái 30%** ảnh | **Layout lệch trái** — đặc trưng layout mới |
| `map_00045.png` | `J3PPJ` | **MỚI: peach + lavender chia chéo (BG-D variant)** + line xám mảnh | J(xanh dương) · 3(xanh lá đậm) · P(teal) · P(xanh lá tươi) · J(đỏ đậm) | **Cao** — `3-P-P-J` cụm cuối đè, **2 cặp lặp** (`PP` ở giữa, `JJ` đầu+cuối) | Cặp `P-P` cùng shape, khác hue (teal vs xanh tươi) → khả năng đọc đúng OK nhưng dễ thành `3PJ` (gộp 2 P); `J` cuối đỏ đậm rõ nét, không bị mất | **Pattern palindrome-ish** `J_P_J` |
| `map_00046.png` | `KNVJV` | Xám sáng + line trắng cong + tile xám | K(tím lavender) · N(xanh lá) · V(đỏ đậm) · J(teal) · V(nâu/khaki) | **Cao** — `N-V-J-V` cụm giữa+cuối đè, **2 V khác màu** | `V-J-V` đè khiến `J` teal nhỏ ép giữa 2 V → có thể bị nuốt thành `KNVV`; `K` đầu rõ; `N-V` đầu đè trộn xanh lá + đỏ thành mảng tối | **2 ký tự lặp khác vị trí** (V ở 3 và 5) |
| `map_00047.png` | `WPM7L` | Xám sáng + arc trắng cong + line vàng dưới | W(cam) · P(đỏ tươi) · M(tím đậm) · 7(xanh lá tươi) · L(olive/vàng-nâu) | **Cao** — `W-P-M-7` cụm 4 chữ đè liên tục | Cảnh báo: `7` xanh tươi nét xiên có thể đọc thành `J` hoặc `1`; `M` tím + `7` xanh đè khít tạo shape giống chữ `H` mở; `W-P` đầu đè vừa | **Glyph có nét xước** (W cam có vệt nhỏ) |
| `map_00048.png` | `JJXXC` | **MỚI: peach + cyan tile + line cyan dày** | J(teal đậm) · J(tím lavender) · X(xanh lá đậm) · X(teal nhạt) · C(xanh lá tươi + **outline đen**) | **Cao** — `JJ` đầu đè khít, `XX` giữa đè khít, **toàn label là 3 cặp lặp** | `JJ` đôi đè → có thể đọc thành `JXXC` (4 chữ) hoặc `IXXC`; `XX` đôi 2 màu khác nhau giữ ranh giới OK; `C` xanh tươi outline đen nổi bật rõ | **Outline ĐEN** (mới) — không phải cyan như batch 2 |
| `map_00049.png` | `UNHLW` | **BG-D: peach+lavender split** + line xanh tím dày chéo X | U(đỏ đậm) · N(tím đậm) · H(xanh dương rực) · L(xanh lá tươi) · W(nâu cam) | **Rất cao** — `U-N-H-L-W` toàn cụm 5 chữ đè liên tục, baseline nhấp nhô | Cảnh báo: `L` xanh tươi nhỏ + `W` nâu cùng nền lavender → ranh giới khó; `N-H` đè trộn tím + xanh thành mảng tối; **layout lệch trái mạnh** | Xác nhận layout lệch trái + nền BG-D variant |
| `map_00050.png` | `A44DD` | Sage green + line peach mỏng + chấm peach | A(xanh lá tươi) · 4(teal) · 4(xanh lá đậm) · D(tím lavender) · D(navy) | **Cao** — `4-4-D-D` cụm giữa+cuối đè, **2 cặp lặp liên tiếp** | `4-4` đôi đè (teal + xanh đậm), nét xiên trùng dễ thành 1 chữ `M` hoặc `H`; `D-D` đôi đè (tím + navy cùng họ tối) → có thể đọc thành `O` hoặc `B` | **Pattern A-44-DD** với 2 cặp lặp gần nhau |
| `map_00051.png` | `DLTDY` | Sage green + line hồng cong dày trên | D(tím đỏ đậm) · L(cyan nhạt) · T(đen) · D(tím đậm) · Y(xanh lá đậm) | **Cao** — `D-L-T-D` cụm giữa đè | `L` cyan nét mỏng + `T` đen đè → có thể đọc thành `D-T-T-D-Y` (mất L); 2 `D` khác vị trí (1 và 4) cùng họ tím có thể nhầm thứ tự | Line hồng cong rất dày trên — có thể bị nhầm là nét chữ |
| `map_00052.png` | `9HF7J` | Xám lavender + line peach mỏng + line tím nhạt | 9(xanh dương đậm) · H(đỏ đậm) · F(đen + **outline cyan**) · 7(vàng cam) · J(xanh lá đậm) | **Cao** — `H-F-7` cụm giữa đè khít | Cảnh báo: `7` vàng cam ép giữa `F-J` có nét xiên dễ thành `1` hoặc bị `F` nuốt → đọc `9HFJ`; `F-7` đè trộn đen + vàng tạo shape giống chữ `E` rộng | Outline cyan rõ ở `F` — xác nhận lại |
| `map_00053.png` | `RLRWV` | **MỚI: khaki/olive đậm phẳng** + line xám mảnh | R(đen + **outline cyan**) · L(xanh lá đậm) · R(vàng tươi) · W(tím đậm) · V(xanh lá đậm) | **Cao** — `R-L-R-W-V` cụm 4 chữ cuối đè | 2 `R` khác hue (đen vs vàng tươi) → tách OK; `L-R` đè khít, `L` có thể bị `R` thứ 2 nuốt → đọc thành `RRWV` (4 chữ); `W-V` cuối đè trộn ranh giới | Xác nhận BG-E khaki từ batch 2 |
| `map_00054.png` | `L9UXH` | **MỚI: khaki vàng pastel phẳng** (sạch, gần như không có line) | L(tím lavender) · 9(xanh dương) · U(đen) · X(cam đỏ) · H(olive/vàng-nâu) | **Cao** — `9-U-X-H` cụm 4 chữ cuối đè, **glyph 9 hình lowercase `g`** (đáng chú ý) | Cảnh báo: `9` xanh có nét cong xuống dưới giống chữ `g` lowercase → model học từ `9` digit có nét thẳng chân có thể fail; `U` đen nét cong dày dễ thành `D`/`O` | **Glyph 9 dạng "two-storey/lowercase g"** — pattern chữ MỚI |
| `map_00055.png` | `7KAM4` | Xám lavender + line peach mỏng + arc xám | 7(navy) · K(xanh lá đậm) · A(tím lavender) · M(xanh lá đậm) · 4(vàng) | **Rất cao** — `7-K-A-M` cụm 4 chữ đè liên tục | `K-A-M` cụm giữa cùng tone xanh+tím trộn ranh giới rất khó; `K` xanh và `M` xanh **cùng màu** → có thể nhầm thứ tự hoặc đọc thành `7KAH4` (M→H); `4` vàng cuối tách rõ | **2 glyph cùng màu cách 1 ký tự** (K và M cùng xanh đậm) |
| `map_00056.png` | `NHCRL` | Xám lavender + arc cam-cyan mảnh | N(xanh lá) · H(teal đậm) · C(olive/vàng-nâu) · R(cyan/teal nhạt) · L(hồng đỏ đậm) | **Trung bình** — chữ tách nhau khá rõ | `R` cyan nét mỏng + `C` olive cùng vùng → có thể bị trộn ở chỗ `C-R`; `H-C` đè vừa, không nguy hiểm | Ảnh "dễ" tương đối trong batch — nền sạch + tách chữ |
| `map_00057.png` | `TU9KU` | **BG-D: peach+lavender split** + line cyan mảnh | T(hồng) · U(đen) · 9(nâu cam) · K(xanh lá đậm) · U(tím đậm) | **Rất cao** — `T-U-9-K-U` toàn cụm đè liên tục, **2 U khác vị trí (2 và 5)**, **glyph 9 dạng lowercase `g`** lần nữa | `U` đen + `9` nâu liền nhau, `9` lại có nét cong xuống → có thể đọc `TO9KU` (U→O); `K-U` cuối trộn xanh+tím tối; layout dồn về **bên phải** | **Layout lệch phải** + xác nhận `9` lowercase pattern |
| `map_00058.png` | `3349V` | Xám sáng + arc cyan tươi rất dày trên+dưới | 3(nâu) · 3(teal) · 4(nâu đỏ) · 9(hồng đỏ) · V(xanh lá tươi) | **Cao** — `3-3-4-9` cụm 4 chữ đầu đè, **3 đôi đè khít** | `3-3` đôi đè (nâu + teal khác hue) tách ranh giới OK; `4-9` đè khít, `9` có nét cong dễ nhầm `g` lần nữa; arc cyan rất dày trên text gần như chạm nét chữ | **Arc cyan đậm như nét chữ** — nhiễu mạnh nhất batch |
| `map_00059.png` | `CJRAN` | Sage green + line hồng đậm chéo cắt qua text + arc xám | C(xanh lá đậm) · J(hồng tươi) · R(tím đậm) · A(đỏ đậm) · N(navy) | **Rất cao** — `J-R-A-N` cụm 4 chữ cuối đè, **line hồng đậm cắt thẳng qua giữa text** | `J-R` đè khiến `J` hồng có thể bị `R` tím nuốt → đọc `CRAN` (4 chữ); `R-A` đè khít cùng họ đỏ-tím trộn ranh giới; `A-N` đè vừa | **Line nhiễu trùng màu glyph** (line hồng cùng tone với J hồng) |

### 10.2 Quan sát mới so với batch 1+2 (cộng dồn 60 ảnh)

#### A. Phân bố mức độ khó (batch 3)

| Mức chồng lấn | Số ảnh | Tỉ lệ batch 3 |
|---|---|---|
| Trung bình | 1 (`00056`) | 5% |
| Cao | 14 | 70% |
| Rất cao | 5 (`00042`, `00044`, `00049`, `00055`, `00057`, `00059`) | 25%* |

(*đếm thực 6 ảnh "Rất cao" → batch 3 **khó hơn rõ** batch 1+2)

**Cộng dồn 60 ảnh (batch 1+2+3)**:
- Trung bình+: ~10%
- Cao: ~50% 
- Rất cao: ~25-30%
- Cao+Rất cao gộp: **~75-80%**

⇒ Cập nhật từ section 9: **không phải 50%** mà gần **80%** ảnh có overlap Cao trở lên. Đây là phát hiện quan trọng — synth không thể coi overlap mạnh là edge case.

#### B. Hiện tượng MỚI (chưa thấy ở batch 1+2)

1. **Glyph `9` dạng "two-storey lowercase g"** (`map_00054`, `map_00057`, một phần `map_00058`):
   - `9` được render với nét cong xuống dưới như chữ `g` lowercase, không phải `9` thẳng chân.
   - Tần suất ~10-15% trong batch 3 → đáng kể.
   - **Hệ quả**: synth nếu chỉ render `9` 1 kiểu sẽ thiếu manifold này → model học từ synth có thể đọc nhầm `9` real thành `g`/`q`. Synth cần render **2 variant cho 9** (single-storey và two-storey).

2. **Outline ĐEN** (`map_00048` C xanh tươi outline đen):
   - Trước đó chỉ thấy outline cyan/teal (batch 2).
   - Outline đen làm glyph nổi bật hơn nhưng cũng có thể bị nhầm là nét chữ thiết kế.
   - Tần suất ước ~3-5% → bổ sung vào synth.

3. **Layout lệch (skew alignment)**:
   - **Lệch trái** (`map_00044` XWN97, `map_00049` UNHLW): toàn text ở 30-40% ảnh trái.
   - **Lệch phải** (`map_00057` TU9KU): toàn text ở 60-70% ảnh phải.
   - Tần suất ước ~10% (batch 1+2 chủ yếu căn giữa).
   - **Hệ quả synth**: cần `RandomLayoutShift(p=0.1, x_shift=±25%)` thay vì luôn căn giữa.

4. **Line nhiễu trùng màu glyph** (`map_00059` line hồng đậm trùng với J hồng):
   - Line nhiễu cắt qua text **cùng hue** với một glyph → tín hiệu màu mất tác dụng, model phải dựa vào shape.
   - **Hệ quả**: synth nên có ít nhất 5% mẫu mà line noise cùng tone với 1 glyph trong text → không cho model học shortcut "màu = tách glyph".

5. **Arc cyan đậm như nét chữ** (`map_00058` 3349V):
   - Arc background dày đến mức gần chạm nét chữ và có thể bị nhầm là 1 nét trong glyph.
   - Tần suất ~5%.
   - **Hệ quả**: line/arc nền không nên luôn mảnh — cần `line_thickness ~ U[1, 3]` với p=0.1 cho thickness=3.

6. **Nhiều cặp ký tự lặp / ký tự lặp lưu trú nhiều vị trí**:
   - `J3PPJ` (`00045`): pattern palindrome `J_P_J` (lặp ở 2 đầu).
   - `KNVJV` (`00046`): V ở vị trí 3 và 5 (cách nhau 1 ký tự).
   - `JJXXC` (`00048`): 2 cặp lặp liên tiếp (JJ + XX).
   - `A44DD` (`00050`): 2 cặp lặp liên tiếp khác (44 + DD).
   - `RLRWV` (`00053`): R ở vị trí 1 và 3.
   - `7KAM4` (`00055`): K và M cùng màu xanh đậm.
   - `TU9KU` (`00057`): U ở vị trí 2 và 5.
   - **Hệ quả**: ký tự lặp ở mọi vị trí (đầu/cuối/cách nhau) là **đặc trưng phổ thông**. Multi-head 5×24 vẫn xử lý tốt. **CTC** đặc biệt khó với `JJXXC`, `A44DD` (cần blank token tách 2 cụm cặp đôi).

7. **Glyph cùng màu cách 1 ký tự** (`map_00055` K và M cùng xanh đậm):
   - Mở rộng từ phát hiện batch 2 (cùng màu liên tiếp) → giờ có cùng màu cách 1 vị trí.
   - Synth nên **không** ràng buộc nghiêm "mọi glyph khác màu" — cho phép trùng màu với p~5-10%.

#### C. Cập nhật danh sách preset background (sau 60 ảnh)

| Preset | Mô tả | Ảnh tham chiếu | Ước tỉ lệ tổng |
|---|---|---|---|
| BG-A | Pastel-lavender + chấm vàng/cam | batch 1 | ~25% |
| BG-B | Peach + line trắng/xám | batch 1, `00042` | ~25% |
| BG-C | Mint/sage green + line hồng/peach mảnh | batch 1, `00040`, `00050`, `00051`, `00059` | ~20% |
| BG-D | Peach + blue/lavender split (water/land tile) | batch 2 (`00020`, `00031`), `00043`, `00045`, `00049`, `00057` | **~15%** ↑ (tăng từ 10%) |
| BG-E | Khaki/olive đậm phẳng | batch 2 (`00033`), `00053` | ~5% |
| BG-F | Khaki vàng pastel phẳng (rất sạch) | batch 2 (`00038`), `00054` | ~5% |
| BG-G | Peach + cyan tile dày (có chấm cyan) | `00048` | ~3-5% (mới) |

⇒ **Tổng 7 preset background**. Synth generator nên có ít nhất 7 preset này, đặc biệt BG-D phổ biến hơn nghĩ.

#### D. Cập nhật cặp ký tự dễ nhầm (sau 60 ảnh)

| Cặp | Bằng chứng mới (batch 3) | Mức rủi ro cập nhật |
|---|---|---|
| `9 ↔ g/q` (do glyph two-storey) | `00054`, `00057`, `00058` | **MỚI: Cao** — chỉ xuất hiện với `9` |
| `4 ↔ A ↔ H` | `00040`, `00042`, `00050` (44 đôi) | **Rất cao** (giữ) |
| `M ↔ N ↔ W ↔ H` | `00041` (NA cụm), `00044` (WN), `00046` (NV), `00055` (KAM cùng tone) | **Rất cao** (giữ) |
| `J ↔ I ↔ 1 ↔ 7` | `00041` (7 nét xiên), `00045` (JJ đôi), `00047` (7 xanh tươi nét xiên), `00048` (JJ), `00052` (7 ép giữa F-J) | **Rất cao** ↑ (thêm 7) |
| `T ↔ 7 ↔ 1` | `00040`, `00041`, `00057` | **Cao** (giữ) |
| `Q ↔ O ↔ 9 ↔ D` | `00044` (9 nhạt), `00050` (DD đôi tối), `00054` (9 g-shape) | **Cao** (giữ) |
| `V ↔ Y ↔ K ↔ X` | `00040` (3YDKH), `00044` (X đầu), `00046` (V đôi cùng vị trí 3,5), `00053` (RLRWV) | **Rất cao** ↑ (do V đôi cách 1 vị trí) |
| `E ↔ F ↔ B ↔ R ↔ P` | `00043` (PDT), `00045` (PP đôi), `00052` (F outline cyan), `00053` (RR cách 1) | **Cao** (giữ) |
| `D ↔ B ↔ O` | `00040` (D bị K nuốt), `00042` (D giữa cụm), `00050` (DD đôi), `00051` (D đôi vị trí 1,4) | **Cao** ↑ |
| `K ↔ X ↔ V` | `00040`, `00046`, `00053`, `00055`, `00057` | **Rất cao** (giữ) |
| `L ↔ I ↔ 1 ↔ J` | `00045` (JJ palindrome), `00046` (J ép VV), `00049` (L nhỏ), `00056` (L cuối ngon) | **Cao** (giữ) |
| `U ↔ N ↔ H` | `00041` (NA), `00049` (UN cụm), `00057` (U đôi vị trí 2,5) | **MỚI: Cao** — phát hiện ở batch 3 |

### 10.3 Đặc trưng tổng hợp sau 60 ảnh (cập nhật từ 9.3)

| Đặc trưng | Tỉ lệ cập nhật | So với 40 ảnh | Hệ quả synth |
|---|---|---|---|
| Overlap Cao+ | ~75-80% | ↑ từ 50% | Bắt buộc render với negative offset trong **đa số** mẫu |
| Có line nền | ~75% | ↑ nhẹ | 1-3 line/ảnh, đa dạng độ dày |
| Có chấm noise màu | ~80% | giữ | Rải dot mật độ 0.3-1% pixel |
| Glyph có outline | ~25% | ↑ từ 20% | `RandomGlyphOutline(p=0.25, color in {cyan, teal, đen})` |
| Layout lệch (không căn giữa) | ~10% | mới | `RandomLayoutShift(p=0.1, x_shift=±25%)` |
| Toàn cụm chữ nghiêng/baseline nhấp nhô | ~10% | ↑ từ 5% | `RandomBaselineWave(p=0.1)` |
| Letter scale/stroke disparity mạnh | ~15% | giữ | `RandomScalePerGlyph(σ=0.15)` + `RandomStrokeWidth(p=0.15)` |
| Glyph color sát bg | ~5% | giữ | Optional |
| Ký tự lặp (mọi vị trí) | **~30% mẫu có ít nhất 1 cặp lặp** | ↑ rõ | Đặc trưng quan trọng → multi-head ưu tiên |
| Ký tự cùng hue (liền/cách 1) | ~10% | ↑ từ 5-8% | Cho phép có chủ ý |
| Glyph `9` dạng `g` (two-storey) | ~10-15% | mới | Synth phải render 2 variant cho `9` |
| Outline ĐEN | ~3-5% | mới | Bổ sung vào outline color set |
| Line nhiễu cùng màu glyph | ~5% | mới | Cố ý có với p=0.05 |
| Line nhiễu rất dày (~3px) | ~5% | mới | `line_thickness ∈ U[1,3]` |

### 10.4 Cập nhật checklist synth (mở rộng từ 8.3 + 9.4)

Bổ sung vào danh sách (đã có 19 mục từ section 9):

20. 🔴 **BẮT BUỘC** — render `9` với **2 variant** (single-storey "9 thẳng" và two-storey "9 cong giống g"), p=0.5 mỗi loại. Đây là missing manifold rõ rệt nhất.
21. 🔶 **Rất nên** — bổ sung outline color set: `{cyan, teal, đen}` thay vì chỉ `{cyan, teal}`.
22. 🔶 **Rất nên** — `RandomLayoutShift(p=0.1, x_shift_pct=[-25, +25])` để cover layout lệch trái/phải (~10% real).
23. 🔷 **Nên** — `line_thickness ~ U[1, 3]px` với p=0.1 cho thickness=3, để cover trường hợp arc/line dày như `00058`.
24. 🔷 **Nên** — `LineColorSameAsGlyph(p=0.05)`: ép một line nhiễu cùng hue với 1 glyph trong text, để model không học shortcut "màu khác = chữ".
25. 🔷 **Nên** — preset BG-G (peach + cyan tile dày) — cộng vào danh sách preset.
26. 🔶 **Rất nên** — đảm bảo synth render được pattern **2 cặp ký tự lặp liền nhau** (`JJXX`, `44DD`) vì chiếm ~10% real có cấu trúc kiểu này. Multi-head model cần training data với pattern này để không bias.
27. 🔷 **Nên** — `BaselineWave(p=0.1, amplitude=±5px per-glyph)` cho hiện tượng baseline nhấp nhô (`ETETY`, `00049`).

### 10.5 Khuyến nghị model (cập nhật từ section 9)

Sau 60 ảnh, củng cố thêm các điểm sau:

1. **Multi-head 5×24 ưu tiên rõ ràng over CTC** ở phase baseline:
   - 30% mẫu có ký tự lặp ở mọi cấu trúc (`JJ`, `XX`, `44`, `DD`, `PP`, `J_P_J`, `V_V`, `R_R`, `U__U`).
   - CTC cần blank token tách → khi 2 ký tự cùng nhau dính sát (`00022 NULLH`, `00050 A44DD`), CTC dễ collapse.
   - Multi-head 5 head độc lập, không có ràng buộc tương tự.

2. **Mask logits 24-class**: vì alphabet đã loại `0,1,2,5,6,8,B,G,I,O,S,Z`, các pred index ngoài 24 lớp này nên bị mask. Sau quan sát:
   - `9 ↔ g` rủi ro cao do glyph two-storey, nhưng `g`/`G` không có trong tập → mask giúp ép model đoán `9`.
   - `D ↔ O`, `O ↔ Q`: O không có → mask giúp giảm rủi ro.
   - `B ↔ E ↔ F ↔ 8`: B, 8 không có → mask cũng giúp.

3. **Augmentation phải nhẹ tay với một số kênh**:
   - **Cấm flip** — labels có cấu trúc (đặc biệt với J/L/3/7).
   - **Hạn chế hue jitter** — vì màu glyph là tín hiệu mạnh, jitter mạnh → label drift trong nội bộ batch.
   - **Cho phép** small rotation, cutout, gaussian noise, crop nhẹ, scale nhỏ.

### 10.6 Tổng kết section 10

Sau 60 ảnh, chân dung dataset đã rõ ràng hơn nhiều:

- **Overlap Cao+ tăng lên ~75-80%** (không phải 50% như đếm sau 40 ảnh) — đây là số quan trọng nhất, ảnh hưởng trực tiếp tới việc thiết kế synth.
- **7 preset background** đã được xác định, BG-D (peach+blue split) phổ biến hơn nghĩ (~15%).
- **Phát hiện missing manifold quan trọng**: `9` two-storey ("lowercase g") chiếm 10-15% — synth chắc chắn cần fix.
- **Cấu trúc ký tự lặp rất đa dạng**: liền (LL, MM, 44, DD, PP, JJ, XX), cách 1 (V_V, R_R, K_M cùng màu), palindrome (J_P_J), 2 cặp liền nhau (JJ XX, 44 DD). Tỉ lệ ~30% mẫu.
- **Layout lệch** (trái/phải) là feature mới — synth cần cover.
- **Line nhiễu cùng tone glyph** + **outline đen** là 2 đặc trưng nhỏ nhưng cần thêm vào synth.
- **Multi-head model là lựa chọn chính xác** ở phase 1, không nên chuyển sang CTC trừ khi đã giải quyết được vấn đề ký tự lặp dính.

**Tổng số mục checklist synth sau 3 batch: 27 mục** (10 từ 8.3 + 9 từ 9.4 + 8 từ 10.4).

## 11. Phân tích chi tiết 21 ảnh thật batch 4 (`map_00225` → `map_00245`)

> Mục đích: skip vào "giữa" tập (~50% chiều dài) để verify các pattern phát hiện ở batch 1-3 có ổn định trên toàn dataset không, đồng thời tìm thêm hiện tượng mới.

### 11.1 Bảng tổng hợp 21 ảnh batch 4

| File | GT label | Nền | Màu/đặc trưng từng glyph | Mức chồng lấn | Glyph có nguy cơ đọc sai | Ghi chú nhiễu |
|---|---|---|---|---|---|---|
| `map_00225.png` | `CQ4K3` | **BG-D variant: sage + peach split tile** + chấm cam | C(vàng tươi) · Q(navy đậm) · 4(xanh lá đậm, **nhỏ và mảnh**) · K(xanh lá tươi) · 3(tím) | **Cao** — `Q-4-K` cụm giữa đè khít, `4` ép sâu | `4` nhỏ ép giữa Q-K dễ bị nuốt → đọc `CQK3` (4 chữ); `Q-4` cùng họ tối + xanh dễ trộn | Tile peach lớn ở góc dưới-phải, sage tile trên-phải |
| `map_00226.png` | `QD7FL` | Xám lavender + line teal cong dày + line xanh dương mảnh | Q(olive/khaki, **lowercase `q`-shape với nét cong xuống**) · D(tím đậm) · 7(teal) · F(xanh lá tươi) · L(teal nhạt) | **Cao** — `Q-D` đè khít, `7` ép giữa D-F | Cảnh báo: `Q` lowercase shape có thể đọc thành `9`/`g` (mirror của 9 two-storey); `7` teal nhạt nét xiên dễ thành `1` hoặc `T` | **Glyph Q dạng lowercase `q`** — pattern MỚI tương tự `9` two-storey |
| `map_00227.png` | `EE9TV` | **MỚI: peach pastel + tile xám block-shape** | E(tím đỏ đậm) · E(tím lavender) · 9(xanh lá tươi, **lowercase `g`**) · T(xanh dương) · V(teal) | **Cao** — `E-E` đè khít cùng họ tím, `T-V` đè cuối | 2 `E` cùng họ tím (đỏ-đậm vs lavender) → khả năng nhầm thành 1 chữ `B`/`R`; `9` g-shape; `T-V` cùng họ xanh dễ trộn ranh giới | **Pattern `EE` đôi khác hue trong cùng họ tím** — khó hơn `LL` đôi khác hue rõ |
| `map_00228.png` | `9XJWH` | **BG-F: khaki vàng pastel** + arc xám mảnh | 9(tím lavender, **lowercase `g`**) · X(đen) · J(xanh dương) · W(hồng) · H(đen) | **Cao** — `X-J-W-H` cụm 4 chữ đè liên tục | `9` g-shape; `X-J` đè (đen + xanh) trộn shape; `W-H` cuối đè cùng có nét đen của H → có thể đọc thành `9XJW` (4 chữ); **2 H/X đen cùng tone** | Glyph cao thấp khác nhau (X cao, J ngắn hơn) — **height disparity** |
| `map_00229.png` | `JHNEC` | Xám sáng + arc trắng cong nhiều + chấm cyan | J(xanh lá tươi, **rất nhỏ và mảnh, ~50% chiều cao**) · H(teal đậm) · N(đen) · E(hồng đỏ đậm) · C(olive) | **Rất cao** — `J` mini ép sâu vào H, `H-N` đè khít cùng tối | `J` mini gần như biến mất → đọc `HNEC` (4 chữ); `H-N` cùng tone tối trộn shape thành chữ `M` rộng; `N-E` đè vừa | **Stroke-mảnh-cao-bị-ép** xác nhận lại pattern (đã thấy ở `00021` U cyan, `00037` Y) |
| `map_00230.png` | `RFENM` | **BG-B: peach phẳng** + line lavender mảnh | R(xanh lá đậm) · F(xanh lá tươi) · E(teal/navy đậm) · N(xanh lá tươi/nhạt) · M(nâu) | **Rất cao** — toàn cụm cùng họ xanh lá+teal → ranh giới mất | Cảnh báo: `R-F-E-N` 4 chữ cùng family xanh lá → model **không thể dùng tín hiệu màu** để tách glyph; `F-E-N` đè khít trộn shape; chỉ `M` nâu đứng riêng | **Toàn cụm cùng family màu** — khó nhất batch về mặt tín hiệu màu |
| `map_00231.png` | `9KHMU` | **MỚI: peach pink pastel** + line cyan-vàng cong mảnh | 9(nâu đậm, **lowercase `g`**) · K(xanh dương) · H(olive/vàng-mustard) · M(đỏ đậm) · U(olive nâu) | **Rất cao** — `K-H-M-U` cụm 4 chữ đè liên tục | `9` g-shape lần nữa; `H` và `U` cùng tone olive nâu → cách nhau 2 vị trí có thể nhầm; `K-H-M` đè khít trộn ranh giới | Xác nhận `9` g-shape rất phổ biến (xuất hiện 4 lần trong batch 4) |
| `map_00232.png` | `E3QPD` | Sage green + line trắng-hồng cong | E(tím lavender) · 3(tím đậm) · Q(tím đậm) · P(**khaki/cream gần trùng nền sage** — contrast cực thấp) · D(xanh lá đậm) | **Rất cao** — `E-3-Q-P` cụm 4 chữ đè, `P` gần như biến mất | Cảnh báo: `P` cream sát nền → khả năng cao bỏ sót → đọc `E3QD` (4 chữ); `3-Q` cùng tím trộn ranh giới dễ thành `B`/`8`; `E-3` đè cùng họ tím | **Failure mode tái xuất**: glyph color sát bg (như `00025` T lavender) — xác nhận ~5% mẫu |
| `map_00233.png` | `W7HKT` | Xám sáng + chấm cam | W(hồng) · 7(xanh dương) · H(đen + **outline cyan rõ**) · K(teal đậm) · T(navy) | **Rất cao** — `W-7-H-K-T` cụm 5 chữ đè liên tục, **baseline nhấp nhô lên-xuống** | `7` xanh nét xiên ép giữa W-H dễ thành `1`/`T`; `H-K` đè trộn (đen + teal cùng tối); `K-T` cuối đè cùng họ navy/teal | **Baseline nhấp nhô per-glyph** rõ (từng chữ ở chiều cao khác nhau) |
| `map_00234.png` | `LCHFM` | Sage green + line hồng cong dày + line xám | L(teal đậm) · C(đen + **outline cyan**) · H(teal đậm) · F(olive) · M(tím đậm) | **Rất cao** — `L-C-H` cụm 3 chữ đè khít, **L và H cùng màu teal** | `L-C-H` đè + `L=H` cùng màu teal → khả năng đọc ngược thứ tự hoặc đọc thành `LCH→ICH/HCL` | **Glyph cùng màu cách 1 vị trí** (L1 và H3) — xác nhận pattern từ batch 3 (`7KAM4`) |
| `map_00235.png` | `KUTLU` | **BG-B variant: peach + tile xám** | K(tím đậm) · U(đỏ đậm) · T(teal) · L(xanh lá tươi + **outline ĐEN**) · U(tím lavender) | **Cao** — `U-T-L-U` cụm 4 chữ đè, **2 U khác vị trí (2 và 5)** | 2 `U` khác hue (đỏ vs lavender) tách ranh giới OK nhưng cấu trúc giống `00057 TU9KU` (U ở vị trí 2 và 5); `T-L` đè khít | Outline đen rõ ở `L` — xác nhận outline đen không phải 3-5% mà ~8-10% |
| `map_00236.png` | `434UM` | **BG-D variant: peach + sage tile chia chéo lớn** + line xám đan xen | 4(olive) · 3(xanh dương) · 4(vàng-olive) · U(xanh lá tươi + **outline ĐEN**) · M(tím đậm) | **Rất cao** — `4-3-4-U` cụm 4 chữ đè khít, **2 chữ 4 khác vị trí (1 và 3) cùng họ olive vàng** | 2 `4` cùng họ olive (sắc thái khác nhau nhỏ) → khả năng nhầm thứ tự; `3-4` cụm giữa đè khít trộn shape | **Pattern lặp `4_4` cách 1 vị trí cùng họ màu** — combo khó cho model |
| `map_00237.png` | `EYKL4` | Xám sáng + line lavender cong nhiều + arc lavender | E(olive) · Y(xanh lá tươi, **rất nhỏ và mảnh**) · K(xanh dương đậm) · L(teal) · 4(tím đậm) | **Cao** — `Y` mini ép giữa E-K, `K-L-4` cụm cuối đè | `Y` mini gần như biến mất giữa E-K → đọc `EKL4` (4 chữ); `K-L` đè khít cùng họ xanh-teal | Lần thứ 4 thấy "stroke-mảnh-cao-bị-ép" (sau `00021 U`, `00037 Y`, `00229 J`) |
| `map_00238.png` | `W7FCN` | Xám sáng + line cam mảnh + line lavender mảnh | W(xanh lá đậm) · 7(hồng) · F(đen + **outline cyan**) · C(olive vàng) · N(xanh dương, **với nét móc dưới giống `j`**) | **Cao** — `W-7-F-C` cụm 4 chữ đè, baseline nhấp nhô | Cảnh báo: `N` xanh dương có nét móc xuống dưới → có thể đọc thành `J`/`H` mất nét; `7` hồng ép giữa W-F nét xiên dễ thành `1`/`T` | **Glyph N dạng có nét móc/serif** — pattern MỚI |
| `map_00239.png` | `4F9DU` | **BG-D variant: lavender + xám split** + arc lavender cong | 4(navy) · F(xanh lá tươi) · 9(hồng, **lowercase `g`**) · D(navy đậm) · U(xanh lá đậm) | **Cao** — `F-9-D` cụm giữa đè khít, `D-U` cuối đè vừa | `9` g-shape lần nữa; `4-F` cùng họ tối khó tách; `D-U` đè khít cùng họ xanh đậm | `9` g-shape xuất hiện thứ 5 lần trong batch 4 → tỉ lệ thực sự cao hơn ước lượng batch 3 |
| `map_00240.png` | `XCXJQ` | **BG-B: peach phẳng** + arc lavender mảnh | X(teal đậm) · C(cam) · X(xanh lá đậm) · J(tím) · Q(teal) | **Cao** — `X-C-X` cụm đầu đè (X cũng ăn vào C), `J-Q` cuối đè vừa | **2 chữ X khác vị trí (1 và 3) cùng họ xanh-teal** → có thể nhầm thứ tự; `C` cam ép giữa 2 X đè trộn shape; `Q` teal cuối + `X` teal đầu cùng tone → có thể nhầm cuối thành đầu | Xác nhận lại pattern lặp khác vị trí cùng họ màu |
| `map_00241.png` | `XPP9K` | **BG-D: peach + lavender split** + arc trắng | X(tím đậm) · P(xanh lá tươi) · P(đỏ đậm) · 9(hồng, **lowercase `g`**) · K(nâu) | **Cao** — `X-P-P` cụm đầu đè khít, `9-K` đè vừa | 2 `P` liền nhau khác hue (xanh + đỏ) tách OK nhưng `X-P` đầu đè trộn ranh giới; `9` g-shape; `K-9` cuối đè vừa | `9` g-shape thứ 6 trong batch 4 |
| `map_00242.png` | `NDP7T` | **BG-D variant: peach + xám lavender split** | N(tím đậm) · D(magenta/hồng tươi) · P(teal đậm) · 7(xanh dương) · T(xanh dương sáng) | **Cao** — `N-D` cụm đầu đè, `P-7-T` cụm cuối đè khít | `D` magenta đầy đặn có thể đọc thành `O`/`B`; `7-T` cuối đè cùng họ xanh dương → có thể đọc thành `1`/dropped; **2 chữ T-shape (P là biến thể)** | Glyph T với 2 màu xanh dương khác nhau gần kề (7 và T) |
| `map_00243.png` | `NMRRC` | Sage green + line hồng đậm cong dày trên + dưới | N(đen + **outline cyan**) · M(xanh lá) · R(vàng cam) · R(navy) · C(magenta đậm) | **Rất cao** — `N-M-R-R-C` toàn cụm 5 chữ đè liên tục, **2 R liền nhau khác hue** | 2 `R` liền nhau (vàng cam vs navy) tách OK nhưng `M-R-R` cụm giữa đè trộn shape; `N-M` đầu cùng tối → có thể nhầm thứ tự; **layout dồn trái nhẹ** | Line hồng đậm dày trên+dưới có thể nhầm là nét chữ |
| `map_00244.png` | `MJXMH` | **BG-B: peach pastel phẳng** + arc lavender mảnh | M(xanh dương) · J(xanh lá đậm) · X(đỏ đậm) · M(navy) · H(xanh lá tươi + **outline ĐEN**) | **Trung bình-Cao** — `M-J-X` cụm đầu đè, **có gap nhỏ giữa X và M2** | 2 `M` khác vị trí (1 và 4) cùng họ xanh đậm → khả năng nhầm thứ tự; `J-X` đè trộn xanh+đỏ; `H` outline đen cuối tách rõ | **Layout có GAP giữa glyph** (X và M2 cách nhau ~5px) — pattern MỚI |
| `map_00245.png` | `P4HNY` | **MỚI: BG-H pink/magenta đậm + sage chia chéo (siêu sặc sỡ)** + nhiều line đan xen dày trên+dưới | P(teal đậm) · 4(xanh dương) · H(tím đỏ đậm) · N(tím nhạt) · Y(olive) | **Rất cao** — `P-4-H-N` cụm 4 chữ đè, **nền pink rất to và nhiều line cong đan xen mật độ cao** | Cảnh báo: nền pink+sage chia chéo bằng đường cong dày → **line nhiễu nhiều như 1 lớp glyph nữa**; `H-N` đè khít cùng họ tím trộn shape | **Preset BG-H mới** — sặc sỡ nhất từ trước tới nay |

### 11.2 Quan sát mới so với batch 1+2+3 (cộng dồn 81 ảnh)

#### A. Phân bố mức độ khó (batch 4)

| Mức chồng lấn | Số ảnh | Tỉ lệ batch 4 |
|---|---|---|
| Trung bình | 0 | 0% |
| Trung bình-Cao | 1 (`00244`) | 5% |
| Cao | 12 | 57% |
| Rất cao | 8 (`00229`, `00230`, `00231`, `00232`, `00233`, `00234`, `00236`, `00243`, `00245`) | 38%* |

(*đếm 9 thực tế, ~43% — batch 4 **khó nhất** trong 4 batch đã xem)

**Cộng dồn 81 ảnh (1+2+3+4)**:
- Trung bình+: ~8%
- Cao: ~52%
- Rất cao: ~30-32%
- Cao+Rất cao: **~80-85%**

⇒ Tăng nhẹ từ con số 75-80% sau 60 ảnh. Bây giờ có thể nói an toàn: **>80% mẫu real có overlap Cao trở lên**.

#### B. Hiện tượng MỚI (chưa thấy ở batch 1-3)

1. **Glyph `Q` dạng lowercase `q`** (`map_00226`): tương tự pattern `9` g-shape ở batch 3.
   - Có nét cong xuống dưới như chữ `q` lowercase, không phải `Q` thẳng đẹp.
   - **Hệ quả**: cộng với `9` two-storey, synth giờ cần render **2 variant cho cả `Q` và `9`**. Đây là 2 glyph "two-storey" chính trong dataset.
   - Tần suất `Q` lowercase ước ~5-10% (ít hơn `9`).

2. **Glyph `N` với nét móc/serif dưới** (`map_00238`):
   - `N` xanh dương có nét móc xuống dưới giống chữ `j` lowercase hoặc serif kéo dài.
   - **Hệ quả**: synth nên có font variant với serif chân (Roman) chứ không chỉ sans-serif. Tần suất ước ~5-8%.

3. **Glyph cùng family màu (cùng họ)** (`map_00230 RFENM` toàn xanh lá+teal, `map_00227 EE` cùng họ tím):
   - Khác với "cùng hue" (đã có ở batch 2-3), đây là cùng **family/họ màu** (xanh lá + teal, tím đỏ + tím lavender, navy + xanh dương).
   - Khoảng cách hue có nhưng nhỏ (∆hue ≤ 30°) → tín hiệu màu không đủ tách rõ.
   - **Hệ quả synth**: khi sample màu, không chỉ cho phép trùng hue (∆=0) mà cũng nên cho phép gần hue (∆hue ∈ [10°, 30°]) với p~10%.

4. **Layout có GAP giữa các glyph** (`map_00244 MJX_MH` X và M2 cách ~5px):
   - Trước đây hầu hết text là cụm liên tục. Đây là ca text có khoảng trống giữa glyph (như "tách thành 2 cụm").
   - Tần suất ước ~3-5%.
   - **Hệ quả synth**: thêm `GapBetweenGlyphs(p=0.05, gap_px=±5-10)`.

5. **Preset BG-H pink/magenta đậm + sage chia chéo** (`map_00245`):
   - Nền sặc sỡ nhất từ trước tới nay (pink dominant, mảng sage chéo, nhiều line cong đan xen mật độ cao).
   - Tần suất ước ~2-3% (hiếm nhưng phải cover).
   - **Hệ quả synth**: mở rộng palette nền với hue cảm xúc cao (pink, magenta, đỏ pastel).

6. **Height disparity per-glyph** (`map_00228 9XJWH`):
   - Khác với scale disparity (đã có ở batch 2), đây là chữ X cao đẹp, J ngắn hơn rõ rệt → mỗi glyph có font-size khác nhau, không chỉ stroke khác nhau.
   - **Hệ quả synth**: mở rộng `RandomScalePerGlyph` để vary cả height (chiều cao thật) chứ không chỉ overall scale.

#### C. Cập nhật xác nhận (pattern đã thấy, batch 4 củng cố)

1. **`9` lowercase `g`** xuất hiện **6 lần trong 21 ảnh batch 4** (`00227, 00228, 00231, 00239, 00241`, một phần `00243`). Tần suất thực tế ~25-30% mẫu có chứa `9` thì 9 đó là g-shape. Đây là **missing manifold quan trọng nhất** đã được xác nhận lần thứ 4.

2. **Stroke-mảnh-cao-bị-ép** (glyph mảnh cao bị ép giữa các glyph dày): 
   - `00229 J` (xanh lá tươi mảnh ép vào H), `00237 Y` (xanh lá tươi mảnh ép vào E-K).
   - Cộng với `00021 U`, `00037 Y` ở batch 1-2 → 4 ca trong 81 ảnh.
   - Tần suất ~5% — đáng cover trong synth.

3. **Outline đen** không hiếm: `00235 L`, `00236 U`, `00244 H` — thêm 3 ca → tổng ~7-10 ca trên 81 ảnh ⇒ **8-10%**, không phải 3-5% như ước batch 3.

4. **Glyph color sát bg** (failure mode):
   - `00025 T lavender on lavender bg` (batch 2) + `00232 P cream on sage bg` (batch 4).
   - 2 ca trên 81 ảnh = ~2.5% — vẫn hiếm nhưng đủ thường xuyên để gọi là "feature" chứ không phải accident.

5. **Ký tự lặp đa dạng**: batch 4 thêm:
   - `EE` đôi liền nhau cùng họ (`00227`).
   - `4_4` cách 1 cùng họ olive (`00236`).
   - `RR` đôi liền nhau khác hue (`00243`).
   - `LH` cùng tone teal cách 1 (`00234`).
   - `XX` cách 1 cùng họ xanh-teal (`00240`).
   - `UU` cách 2 (`00235`).
   - `MM` cách 2 (`00244`).
   ⇒ **9 ca trong 21 ảnh = ~43%** mẫu batch 4 có cấu trúc lặp/cùng họ. Tăng từ ước lượng 30% sau batch 3.

#### D. Cập nhật danh sách preset background (sau 81 ảnh)

| Preset | Mô tả | Ước tỉ lệ tổng |
|---|---|---|
| BG-A | Pastel-lavender + chấm vàng/cam | ~22% |
| BG-B | Peach phẳng + line trắng/xám/lavender | ~22% |
| BG-C | Mint/sage green + line hồng/peach mảnh | ~18% |
| BG-D | Peach + blue/lavender split (water/land tile) | **~18%** ↑ (tăng từ 15%) |
| BG-E | Khaki/olive đậm phẳng | ~5% |
| BG-F | Khaki vàng pastel phẳng | ~5% |
| BG-G | Peach + cyan tile dày | ~3-5% |
| **BG-H (MỚI)** | Pink/magenta đậm + sage chia chéo siêu sặc sỡ | ~2-3% |

⇒ **Tổng 8 preset background**. BG-D tiếp tục phổ biến hơn ước (gần 1/5 mẫu).

#### E. Cập nhật cặp ký tự dễ nhầm (sau 81 ảnh)

| Cặp | Bằng chứng mới (batch 4) | Mức rủi ro cập nhật |
|---|---|---|
| `9 ↔ g/q` (two-storey/lowercase) | `00227, 00228, 00231, 00239, 00241` | **Rất cao** ↑ (được xác nhận thứ 6 lần) |
| **MỚI: `Q ↔ q ↔ 9 ↔ g`** | `00226` (Q lowercase) + tất cả ca `9` g-shape | **Cao** (mới, do glyph Q cũng có variant) |
| `4 ↔ A ↔ H` | `00225` (4 nhỏ ép QK), `00236` (44 cùng họ olive) | **Rất cao** (giữ) |
| `M ↔ N ↔ W ↔ H` | `00229 (HN trộn)`, `00231 (KHM)`, `00238 (N có móc)`, `00243 (NM)`, `00244 (MM)` | **Rất cao** (giữ + thêm `N→J` do nét móc) |
| `J ↔ I ↔ 1 ↔ 7` | `00226 (7 nhạt)`, `00228 (J ngắn)`, `00229 (J mini)`, `00233 (7 ép)`, `00237 (Y mini cũng risk)`, `00238 (7 hồng)`, `00242 (7-T cùng họ xanh)` | **Rất cao** (giữ) |
| `T ↔ 7 ↔ 1` | `00226, 00233, 00238, 00242` | **Rất cao** ↑ (tăng từ Cao) |
| `Q ↔ O ↔ 9 ↔ D ↔ q` | `00225 (Q ép)`, `00226 (Q lowercase)`, `00240 (Q teal cùng họ X)`, `00242 (D đầy đặn)` | **Rất cao** ↑ (thêm `q`) |
| `V ↔ Y ↔ K ↔ X` | `00237 (Y mini)`, `00240 (XX cách 1)`, `00244 (X giữa MJ-MH)` | **Rất cao** (giữ) |
| `E ↔ F ↔ B ↔ R ↔ P` | `00227 (EE đôi cùng họ tím)`, `00230 (F-E cùng xanh lá)`, `00232 (P sát nền)`, `00238 (F outline cyan)`, `00243 (RR đôi)` | **Rất cao** ↑ (P thêm vào do failure mode sát nền) |
| `D ↔ B ↔ O` | `00226 (D giữa Q-7)`, `00232 (D đè P)`, `00239 (D-U)`, `00242 (D magenta đầy)` | **Cao** (giữ) |
| `K ↔ X ↔ V` | `00225 (K sau 4)`, `00233 (HK đè)`, `00240 (XX cách)` | **Rất cao** (giữ) |
| `L ↔ I ↔ 1 ↔ J` | `00229 (J mini như L mảnh)`, `00234 (LH cùng tone)`, `00235 (LU cuối)`, `00237 (Y mini)` | **Rất cao** (giữ) |
| `U ↔ N ↔ H ↔ M` | `00229 (HN trộn)`, `00231 (KHMU 4 chữ đè)`, `00235 (UU cách 2)`, `00244 (MM cách 2)` | **Rất cao** (giữ) |
| `C ↔ G ↔ O` (G,O không có) | `00229 (C đứng cuối)`, `00231 (không có C)`, `00234 (C có outline)`, `00240 (C cam ép XX)`, `00243 (C magenta cuối)` | **Trung bình** — `C` có nguy cơ thấp do alphabet đã loại G,O |

### 11.3 Đặc trưng tổng hợp sau 81 ảnh (cập nhật từ 10.3)

| Đặc trưng | Tỉ lệ cập nhật | So với 60 ảnh | Hệ quả synth |
|---|---|---|---|
| Overlap Cao+ | ~80-85% | ↑ nhẹ từ 75-80% | Bắt buộc render với negative offset trong **đại đa số** mẫu |
| Có line nền | ~80% | ↑ nhẹ | 1-3 line/ảnh, đa dạng độ dày |
| Có chấm noise màu | ~85% | ↑ nhẹ | Rải dot mật độ 0.3-1% pixel |
| Glyph có outline | ~25% | giữ | `RandomGlyphOutline(p=0.25)` |
| **Outline ĐEN** | **~8-10%** | ↑ từ 3-5% | Bổ sung mạnh vào outline color set |
| Layout lệch (không căn giữa) | ~12% | ↑ nhẹ | `RandomLayoutShift(p=0.12)` |
| **Layout có GAP giữa glyph** | **~3-5%** | mới | `GapBetweenGlyphs(p=0.05)` |
| Toàn cụm baseline nhấp nhô/per-glyph | ~12% | ↑ | `RandomBaselineWave(p=0.12)` |
| Letter scale/stroke disparity | ~18% | ↑ từ 15% | Rộng hơn |
| Glyph color sát bg | ~3% | giữ | Optional |
| Ký tự lặp/cùng họ (mọi vị trí) | **~40%** | ↑ rõ từ 30% | Đặc trưng PHỔ THÔNG → multi-head ưu tiên mạnh |
| Ký tự cùng hue/cùng family | ~15% | ↑ từ 10% | Cho phép có chủ ý |
| **Glyph `9` dạng `g`** | **~25-30%** mẫu có `9` | xác nhận | Synth phải render 2 variant cho `9` |
| **Glyph `Q` dạng `q`** | **~5-10%** mẫu có `Q` | mới | Synth phải render 2 variant cho `Q` |
| **Glyph `N` có móc/serif** | ~5-8% mẫu có `N` | mới | Bổ sung font variant |
| Line nhiễu cùng màu glyph | ~5% | giữ | p=0.05 |
| Line nhiễu rất dày (~3px) | ~6% | ↑ nhẹ | `line_thickness ∈ U[1,3]` |
| **Toàn cụm cùng family màu** | **~5%** | mới | Cho phép có chủ ý với p=0.05 |

### 11.4 Cập nhật checklist synth (mở rộng từ 8.3 + 9.4 + 10.4)

Bổ sung vào danh sách (đã có 27 mục từ section 10):

28. 🔴 **BẮT BUỘC** — render `Q` với **2 variant** (uppercase Q chuẩn và lowercase q-shape có nét cong xuống), p=0.85/0.15.
29. 🔶 **Rất nên** — bổ sung font variant **có serif** (Roman) cho ít nhất `N, M, R, T, F, E, H` với p=0.1 — cover ca `N` nét móc dưới.
30. 🔶 **Rất nên** — preset BG-H (pink/magenta + sage chia chéo) — thêm vào palette nền.
31. 🔷 **Nên** — `GapBetweenGlyphs(p=0.05, gap_px ∈ [3, 12])` — cover layout có khoảng trống giữa glyph.
32. 🔶 **Rất nên** — khi sample màu glyph, cho phép `∆hue ∈ [10°, 30°]` với p~10% (gần hue) thay vì chỉ phân thành "trùng hue" và "khác hue rõ".
33. 🔶 **Rất nên** — `SameFamilyMultiGlyph(p=0.05)`: với xác suất 5%, ép **3+ glyph trong text** cùng family màu (xanh lá+teal, tím+lavender, navy+xanh dương, đỏ+hồng) → cover ca `RFENM`.
34. 🔷 **Nên** — `RandomHeightPerGlyph(p=0.18, height_range=[0.7, 1.15])` — cover height disparity per-glyph (không chỉ scale).
35. 🔶 **Rất nên** — tăng `outline_color_weight` cho **đen**: thay vì 50/50 cyan-teal, dùng phân bố `cyan:0.5, teal:0.3, đen:0.15, trắng:0.05` để khớp tỉ lệ thực tế ~8-10% outline đen.

### 11.5 Khuyến nghị model (cập nhật từ 10.5)

Sau 81 ảnh, các điểm sau được củng cố mạnh hơn:

1. **Multi-head 5×24 vẫn là lựa chọn ĐÚNG** ở phase baseline:
   - Tỉ lệ ký tự lặp/cùng họ tăng lên ~40% sau 81 ảnh (từ 30% sau 60 ảnh).
   - Nhiều cấu trúc khó: `EE` cùng họ tím (`00227`), `4_4` cùng họ olive (`00236`), `RR` đôi (`00243`), `XX` cách 1 cùng họ teal (`00240`).
   - CTC vẫn cần blank token tách → khả năng fail trên các cấu trúc này.

2. **Mask logits 24-class** càng quan trọng hơn:
   - `Q ↔ q ↔ 9 ↔ g`: Q và 9 trong tập, q và g không → mask giúp ép model đoán Q/9.
   - `N ↔ J` (nét móc dưới): N và J đều trong tập nhưng J alphabet có shape rất khác → mask giảm risk.
   - `D ↔ B ↔ O`: D trong tập, B/O không → mask giúp.

3. **Augmentation cần giữ thận trọng** với:
   - **Color jitter mạnh**: bị cấm vì màu là tín hiệu phân biệt glyph.
   - **Spatial jitter mạnh**: cho phép small translate/rotate (≤8°) nhưng không erase glyph chính.
   - **Cutout có chủ ý**: nên có với p=0.1 để cover failure mode "glyph color sát bg" — ép model robust khi mất 1 phần glyph.

4. **Khả năng generalize từ synth → real**:
   - 8 preset background phải cover được, đặc biệt BG-D (peach+lavender split, ~18%) là phổ biến nhất.
   - 2 glyph variant `9 g-shape` và `Q q-shape` là missing manifold rõ ràng nhất.
   - 40% mẫu có cấu trúc lặp/cùng họ → synth phải sample label uniform mà không chế tạo "label easy" (5 chữ khác nhau khác hue).

### 11.6 Tổng kết section 11

Sau 81 ảnh đã phân tích trên cả tập, chân dung dataset bây giờ đã rất rõ:

- **Overlap Cao+ ổn định ở mức ~80-85%** (xác nhận lần thứ 4) — không phụ thuộc vị trí trong tập.
- **8 preset background** đã được xác định, BG-D phổ biến nhất ngoài lavender-flat.
- **2 missing manifold rõ rệt nhất cho synth**:
  1. `9` two-storey (g-shape) — ~25-30% mẫu có `9`.
  2. `Q` lowercase (q-shape) — ~5-10% mẫu có `Q`.
- **Cấu trúc ký tự lặp/cùng họ**: ~40% mẫu (vs 30% ước batch 3, 15% ước batch 1).
- **Layout không phải chỉ căn giữa**: lệch trái/phải ~12%, có gap giữa glyph ~5%.
- **Outline đen** phổ biến hơn nghĩ: ~8-10% (vs 3-5% ước batch 2).
- **Failure mode "glyph sát nền"** xác nhận: ~3% — robust hơn khi không né.
- **Toàn cụm cùng family màu** là feature khó nhất chưa cover được trong synth (`RFENM`).

**Tổng số mục checklist synth sau 4 batch: 35 mục** (10 từ 8.3 + 9 từ 9.4 + 8 từ 10.4 + 8 từ 11.4).

**Đánh giá tổng thể**: dataset có tính nhất quán cao về phong cách (cùng generator, cùng pipeline render) nhưng đa dạng về biểu hiện. Các đặc trưng phát hiện ổn định qua 4 batch (đầu, giữa, ngẫu nhiên giữa tập) → có thể tin dùng làm bản thiết kế synth chính thức.

## 12. Spot-check cuối + Tổng kết toàn bộ phân tích

### 12.1 Spot-check 2 ảnh đại diện range 300-319

| File | GT label | Nền | Màu/đặc trưng từng glyph | Mức chồng lấn | Glyph có nguy cơ đọc sai | Ghi chú nhiễu |
|---|---|---|---|---|---|---|
| `map_00300.png` | `YMH7J` | **BG-B variant: peach pastel + line lavender mảnh ĐAN XEN X-shape mật độ rất cao** | Y(teal đậm) · M(tím đỏ đậm/đen) · H(magenta/hồng) · 7(xanh lá tươi) · J(navy + **outline cyan rõ**) | **Rất cao** — `Y-M-H-7` cụm 4 chữ đè liên tục, ranh giới mất | `Y` đầu mất 1 nhánh do `M` đè → có thể đọc thành `V`/`U`; `7` xanh lá nét chéo dài → có thể đọc thành `1`/`L`; `J` có outline cyan tách rõ | **Mật độ line nền cao nhất từ trước tới nay** — line lavender đan xen X-shape phủ ~30% diện tích nền |
| `map_00319.png` | `VA7M7` | **BG-A variant: xám lavender + chấm peach + tile peach nhỏ rải rác** | V(đen + **outline cyan mảnh**) · A(hồng pink) · 7(xanh lá tươi) · M(vàng tươi) · 7(navy) | **Cao** — `V-A-7-M` cụm 4 chữ đè, **2 chữ `7` cách 1 vị trí khác hue** | `V-A` đầu đè trộn đen+hồng; `M` vàng + `7` xanh lá đè khít trộn shape; 2 `7` khác hue (xanh lá vs navy) tách OK; `V` outline cyan tách rõ | **Pattern lặp số `7_7` cách 1 vị trí** — mở rộng từ pattern `4_4`, `R_R` ở batch trước |

### 12.2 Hiện tượng mới (nhỏ) từ 2 ảnh spot-check

1. **Mật độ line nền cực cao** (`map_00300`): line lavender mảnh đan xen X-shape phủ ~30% diện tích → mật độ cao hơn các ca trước. Chứng tỏ tham số mật độ line nên có giá trị max cao hơn ước (~30-40% pixel coverage thay vì 10-15%).
   - **Hệ quả synth**: `line_density ∈ [0.05, 0.35]` thay vì cap ở 0.15.

2. **Pattern lặp số `7_7`** (`map_00319`): mở rộng cấu trúc lặp đã thấy:
   - `4_4` cùng họ olive (`map_00236`)
   - `R_R` khác hue (`map_00046, 00053, 00243`)
   - `7_7` khác hue (`map_00319`)
   - Tổng cộng giờ có pattern lặp ở **mọi loại ký tự** (chữ cái + số).

3. **Outline cyan vẫn ổn định**: `J` ở `00300` và `V` ở `00319` đều có outline cyan rõ → xác nhận tỉ lệ ~25% mẫu có outline (không thay đổi sau spot-check).

### 12.3 Tổng số liệu chính sau **83 ảnh** đã phân tích trên cả tập 500 ảnh

> **Lưu ý phương pháp**: 83 = 20 (batch 1, đầu tập) + 20 (batch 2) + 20 (batch 3) + 21 (batch 4, ~giữa tập) + 2 (spot-check 300, 319). Phân bố vị trí cover: **đầu, đầu+1, đầu+2, giữa, gần giữa-cuối**. Không cover được ~70 ảnh cuối (430-500), nhưng đã đủ để khẳng định tính nhất quán của dataset.

| Đặc trưng | Tỉ lệ ổn định cuối cùng | Ghi chú |
|---|---|---|
| **Overlap Cao+** (chồng lấn mạnh) | **~80-85%** | Xác nhận qua 4 batch + spot-check |
| Overlap Rất cao | ~30% | Stable |
| Có line nền | ~80% | 1-3 line/ảnh, density có thể tới 30% |
| Có chấm noise màu | ~85% | Mật độ 0.3-1% pixel |
| Glyph có outline | ~25% | cyan dominant + đen 8-10% + teal + trắng |
| **Outline đen** | ~8-10% | Cao hơn ước ban đầu |
| Layout lệch (không căn giữa) | ~12% | Trái/phải cân bằng |
| **Layout có GAP** giữa glyph | ~3-5% | Hiếm nhưng phải cover |
| Baseline nhấp nhô per-glyph | ~12% | Khác với rotation toàn-string |
| Stroke disparity (mảnh-cao bị ép) | ~5% | 4 ca đã xác nhận |
| Letter scale disparity | ~13% | Bao gồm height và width |
| **Glyph color sát bg** (failure mode) | ~3% | 2 ca trên 83 |
| **Ký tự lặp/cùng họ** (mọi vị trí) | **~40%** | Đặc trưng phổ thông |
| Ký tự cùng hue/family liền kề | ~15% | |
| **`9` dạng `g`** (lowercase two-storey) | ~25-30% mẫu có `9` | **Missing manifold #1** |
| **`Q` dạng `q`** (lowercase) | ~5-10% mẫu có `Q` | **Missing manifold #2** |
| **`N` có nét móc/serif** | ~5-8% mẫu có `N` | Mới phát hiện batch 4 |
| **Toàn cụm cùng family màu** | ~5% | Mới phát hiện batch 4 |
| Line nhiễu cùng màu glyph | ~5% | |
| Line dày (~3px) | ~6% | |
| **Mật độ line cao (~30%)** | ~3% | Mới phát hiện spot-check |

### 12.4 Danh sách 8 preset background (cuối cùng)

| Preset | Mô tả | Tỉ lệ ước | Đại diện |
|---|---|---|---|
| BG-A | Pastel-lavender + chấm vàng/cam | ~22% | `map_00000`, `map_00319` |
| BG-B | Peach phẳng + line trắng/xám/lavender | ~22% | `map_00230`, `map_00240`, `map_00300` |
| BG-C | Mint/sage green + line hồng/peach mảnh | ~18% | `map_00029`, `map_00040`, `map_00229` |
| BG-D | Peach + blue/lavender split (water/land tile) | ~18% | `map_00020`, `map_00043`, `map_00057`, `map_00225` |
| BG-E | Khaki/olive đậm phẳng | ~5% | `map_00033`, `map_00053` |
| BG-F | Khaki vàng pastel phẳng | ~5% | `map_00038`, `map_00054`, `map_00228` |
| BG-G | Peach + cyan tile dày | ~3-5% | `map_00048` |
| BG-H | Pink/magenta đậm + sage chia chéo siêu sặc sỡ | ~2-3% | `map_00245` |

### 12.5 Checklist synth cuối cùng (35 mục, được rút từ tổng hợp)

#### 🔴 BẮT BUỘC (impact lớn nhất, missing manifold)
1. Render `9` với 2 variant (single-storey + two-storey g-shape), p=0.7/0.3.
2. Render `Q` với 2 variant (uppercase Q + lowercase q-shape), p=0.85/0.15.

#### 🔶 RẤT NÊN (impact lớn)
3. Đủ 8 preset background (BG-A đến BG-H) với phân bố tỉ lệ ước trên.
4. Negative offset rendering: ép overlap Cao+ trong ~80% mẫu (không phải 50%).
5. Outline color set: cyan(0.5), teal(0.3), đen(0.15), trắng(0.05).
6. RandomLayoutShift(p=0.12, x_shift=±25%) cho layout lệch.
7. SameFamilyMultiGlyph(p=0.05): ép 3+ glyph cùng family màu trong 1 text.
8. Khi `c[i] == c[i+1]`: hue khác **hoặc** offset dương rõ; cấm cả 2 trùng hue + dính.
9. Bổ sung font variant có serif cho `N, M, R, T, F, E, H` với p=0.1.
10. Đa dạng line noise: line đơn, arc cong, lưới X chéo, cluster đan xen mật độ cao.

#### 🔷 NÊN (impact vừa)
11. RandomGlyphOutline(p=0.25, color=trên).
12. RandomBaselineWave(p=0.12, amplitude=±5px per-glyph).
13. RandomScalePerGlyph(p=0.18, scale_range=[0.55, 1.4]).
14. RandomHeightPerGlyph(p=0.18, height_range=[0.7, 1.15]).
15. RandomStrokeWidth(p=0.15, range=[0.7, 1.3]).
16. GapBetweenGlyphs(p=0.05, gap_px=[3, 12]).
17. line_thickness ∈ U[1, 3]px với p=0.1 cho thickness=3.
18. line_density ∈ [0.05, 0.35] (tăng max từ 0.15).
19. LineColorSameAsGlyph(p=0.05).
20. RandomGlyphClose-to-bg(p=0.03) — failure mode.
21. SameHueOnNearbyGlyphs(p=0.05, distance ∈ [1, 3]).
22. SameFamilyHueNearby(p=0.10, ∆hue ∈ [10°, 30°]).
23. ChấmDot density 0.3-1% pixel.
24. Ép 2 cặp lặp liền nhau (`JJXX`, `44DD`) có thể xuất hiện.
25. RandomStringTilt(p=0.05, angle=[-12°, +12°]).

#### 💡 BỔ SUNG (impact nhỏ nhưng đáng có)
26-35. Các mục nhỏ khác đã liệt kê ở các section trước.

### 12.6 Khuyến nghị model (final)

1. **Multi-head 5×24 là lựa chọn ĐÚNG** ở phase baseline. Lý do:
   - 40% mẫu có cấu trúc ký tự lặp/cùng họ ở mọi vị trí.
   - CTC sẽ khó với các pattern: `EE` cùng họ tím, `4_4` cùng olive, `LL` đè đến mức trộn.
   - Multi-head 5 head độc lập → không bị ràng buộc CTC blank token.

2. **Mask logits 24 lớp (loại 0,1,2,5,6,8,B,G,I,O,S,Z)**:
   - Giảm risk `9 → g`, `Q → q` (q,g không trong tập).
   - Giảm risk `D → O/B`, `E/F → B/8`.
   - Là biện pháp phòng vệ rẻ nhưng hiệu quả.

3. **Augmentation strategy**:
   - **Cấm hoàn toàn**: hue jitter mạnh (≥30°), horizontal/vertical flip.
   - **Hạn chế**: rotation ±8°, brightness ±15%.
   - **Cho phép**: small translate, gaussian noise, cutout (p=0.1, ép robustness với failure mode "glyph sát bg"), small crop.

4. **Training pipeline 3 phase đề xuất**:
   - Phase 1: Pretrain trên synth (10k+ ảnh sinh từ checklist 35 mục) — ~50 epoch.
   - Phase 2: Fine-tune trên synth + real (mix 70/30) — ~30 epoch với lr giảm 5×.
   - Phase 3: Fine-tune chỉ real (500 ảnh) với augmentation nhẹ — ~20 epoch với lr giảm thêm.

### 12.7 Kết luận tổng (final summary)

Sau 83 ảnh phân tích chi tiết qua 4+1 batch (đầu/giữa-tập/spot-check), chân dung dataset hiện tại đã **rõ ràng và ổn định**:

**Đặc trưng cốt lõi (top 7 cần synth cover):**
1. Overlap Cao+ chiếm ~80-85% — không phải edge case.
2. `9` dạng `g` (~30% mẫu có 9) — missing manifold #1.
3. `Q` dạng `q` (~10% mẫu có Q) — missing manifold #2.
4. Ký tự lặp/cùng họ ~40% — đặc trưng phổ thông.
5. 8 preset background đã định danh đầy đủ.
6. Outline glyph (cyan/teal/đen) ~25% mẫu.
7. Failure mode "glyph sát bg" ~3% — không né mà cần robust.

**Đặc trưng thứ cấp (cần cover, không phải critical):**
- Layout lệch trái/phải, có gap, baseline nhấp nhô (~12-15% mỗi loại).
- Stroke/scale/height disparity (~15-18%).
- Line noise đa dạng (đơn/cong/X chéo/đan xen mật độ cao).
- Cùng family màu trong text (~5%).
- Font có serif cho một số chữ (~5-8%).

**Sai lệch trong synth hiện tại (suy đoán, cần verify khi mở `src/synth.py`)**:
- Có thể đang render `9` đẹp single-storey only → thiếu manifold lớn.
- Có thể đang ép glyph khác hue rõ → thiếu cấu trúc cùng họ màu.
- Có thể chưa có BG-D (water/land split) và BG-H (pink magenta) — 2 preset chiếm ~20% real.
- Có thể đang căn giữa cứng → thiếu layout lệch ~12%.

**Câu trả lời cho câu hỏi cốt lõi "có nên xài synth không?"**:
Có. Với 500 ảnh real, không đủ để train từ đầu nhưng đủ để **fine-tune**. Synth đóng vai trò pretrain với 10-100× lượng data. Tuy nhiên, **synth phải được điều chỉnh kỹ** theo 35 mục checklist trên — nếu không, synth-only sẽ có domain gap rất lớn (đặc biệt qua 2 missing manifold `9 g-shape` và `Q q-shape`).

**Điểm dừng phân tích bằng mắt**:
83 ảnh đủ để xây dựng bản thiết kế synth. Nếu sau khi train baseline có confusion matrix cụ thể → có thể quay lại phân tích thêm các ca confusion để bổ sung checklist.

---

> **End of research.md** — bản nghiên cứu dữ liệu được hoàn thiện qua 5 lần đối chiếu pixel↔nhãn (batch 1, 2, 3, 4, spot-check). Tổng dài 12 section. Có thể dùng làm base cho việc viết lại `src/synth.py` và thiết kế training pipeline 3 phase.

## 13. Bổ sung nghiên cứu từ quá trình tinh chỉnh synth (ưu tiên CAO — ảnh hưởng trực tiếp đến độ giống real)

> Các phát hiện dưới đây được rút ra trong quá trình so sánh trực tiếp ảnh synth_preview với ảnh real. Đây là những yếu tố **ưu tiên cao** vì ảnh hưởng lớn đến domain gap giữa synth và real.

### 13.1 Kích thước glyph (font size) — QUAN TRỌNG NHẤT

**Vấn đề**: synth ban đầu render glyph quá to (base_size 44-56 → glyph height ~50-60px trên canvas 128px = ~40-47% canvas). Real chỉ ~25-32% canvas.

**Giải pháp cuối cùng**: `base_size = 28-34` (glyph height thực tế ~22-28px trên canvas 128px).

**Lý do ưu tiên cao**: nếu glyph quá to, tổng chiều rộng 5 ký tự vượt quá canvas → chữ bị cắt ở rìa, hoặc tràn ra ngoài. Đây là sai lệch dễ thấy nhất khi so synth vs real.

**Quy tắc**: tổng chiều rộng string (spacing × 4) nên nằm trong ~60-65% chiều rộng canvas (tức ~77-83px trên 128px). Spacing tương ứng: 12-16px (overlap Cao+), 17-20px (easy).

### 13.2 Font — 3 nhóm với tỉ lệ cụ thể

**Phát hiện từ phân tích visual 18 ảnh real (batch 4 + spot-check)**:

| Nhóm font | Đặc điểm nhận dạng | Tỉ lệ trong real | Font Windows tương ứng |
|---|---|---|---|
| **Bold sans** | Stroke đồng đều dày, không có chân | ~55% | Verdana Bold, Arial Bold, Bahnschrift, Trebuchet Bold |
| **Slab serif** | Chân ngang dày rõ rệt (serif vuông) | ~30% | Rockwell Bold, Cooper Black, Georgia Bold, Cambria Bold |
| **Display heavy** | Stroke siêu dày, condensed | ~15% | Impact, Arial Black |

**Quy tắc font per-image**:
- **~85% ảnh**: tất cả 5 glyph dùng CÙNG 1 font (chỉ đổi màu/size nhẹ).
- **~15% ảnh**: mỗi glyph có thể dùng font khác nhau (mixed-font mode).

**Lý do ưu tiên cao**: nếu synth chỉ dùng 1 font (ví dụ Arial Bold) → model chỉ học 1 style → fail trên 45% real dùng serif/display. Ngược lại nếu random font per-glyph 100% → trông "frankenstein" không giống real.

### 13.3 Layout căn giữa — 95% mẫu

**Phát hiện**: ban đầu synth set layout shift 12% với biên độ ±22% → quá nhiều ảnh bị lệch trái/phải rõ rệt. Real chủ yếu (~90-95%) căn giữa, chỉ ~5% mới lệch nhẹ.

**Giải pháp**: giảm `layout_shift_prob` từ 0.12 → **0.05**, biên độ từ ±22% → **±15%**.

**Lý do ưu tiên cao**: nếu synth lệch nhiều, model học rằng "text có thể ở bất kỳ đâu" → phân tán attention, giảm accuracy trên real (vốn luôn căn giữa).

### 13.4 Per-glyph scale disparity — chỉ 13% mẫu

**Phát hiện**: synth ban đầu cho mỗi glyph scale 0.78-1.28 → **mọi ảnh** đều có glyph to nhỏ khác nhau rõ rệt. Real thì ~87% ảnh có glyph đồng đều (chỉ ±6% variation), chỉ ~13% mới có disparity rõ.

**Giải pháp**: 
- Default mode (87%): `size_scale ∈ [0.94, 1.06]`, `sx/sy ∈ [0.94, 1.06]`
- Disparity mode (13%): `size_scale ∈ [0.78, 1.22]`, `sx/sy ∈ [0.85, 1.15]`

**Lý do ưu tiên cao**: nếu synth luôn có disparity → model học rằng "glyph to nhỏ khác nhau là bình thường" → có thể bỏ qua tín hiệu kích thước khi phân biệt ký tự trên real (vốn đồng đều).

### 13.5 Per-glyph rotation — nhẹ hơn nhiều so với ban đầu

**Phát hiện**: synth ban đầu xoay ±12° per-glyph → trông quá "wobble". Real hầu hết glyph gần thẳng đứng, chỉ xoay nhẹ ±5-7°.

**Giải pháp**: giảm rotation từ ±12° → **±7°**.

**Lý do ưu tiên cao**: rotation quá mạnh thay đổi shape glyph đáng kể (ví dụ `7` xoay 12° trông giống `1`/`L`) → model học sai shape.

### 13.6 Tóm tắt thứ tự ưu tiên khi tinh chỉnh synth

| # | Yếu tố | Impact | Trạng thái |
|---|---|---|---|
| 1 | **Kích thước glyph** (base_size) | Rất cao | ✅ Fixed: 28-34 |
| 2 | **Font categories + weights** | Rất cao | ✅ Fixed: 55/30/15 sans/serif/display |
| 3 | **Layout căn giữa** | Cao | ✅ Fixed: 95% centred |
| 4 | **Scale disparity per-image flag** | Cao | ✅ Fixed: 87% uniform, 13% disparity |
| 5 | **Rotation per-glyph** | Cao | ✅ Fixed: ±7° |
| 6 | **Spacing vs canvas ratio** | Cao | ✅ Fixed: total width ≤ 80px |
| 7 | **Font per-image vs per-glyph** | Cao | ✅ Fixed: 85% same, 15% mixed |

> **Bài học**: các yếu tố "tỉ lệ" (size, spacing, layout position) ảnh hưởng đến domain gap **nhiều hơn** các yếu tố "chi tiết" (outline, noise, colour mode). Khi tinh chỉnh synth, luôn fix tỉ lệ trước, chi tiết sau.

### 13.7 ĐÍNH CHÍNH QUAN TRỌNG: Không có chữ thường (lowercase) trong dataset

**Sai lầm trước đó**: Sections 10, 11, 12 đã nhận định rằng:
- `9` có variant "two-storey lowercase g" (~25-30% mẫu có 9)
- `Q` có variant "lowercase q" (~5-10% mẫu có Q)

**Thực tế**: Sau khi so sánh trực tiếp synth output (có render lowercase `g`/`q`) với ảnh real, xác nhận rằng **TOÀN BỘ ký tự trong dataset real đều là UPPERCASE**. Những gì trước đó bị nhầm là "lowercase g/q" thực ra chỉ là:
- Font serif/slab serif render `9` với đuôi cong dài hơn (tail curly) — vẫn là ký tự `9` uppercase.
- Font serif render `Q` với nét descender dài — vẫn là ký tự `Q` uppercase.
- Sự khác biệt hình dạng đến từ **font variety** (serif vs sans), KHÔNG phải từ lowercase substitution.

**Hệ quả cho synth**:
- ❌ **XOÁ** mục 1 (render `9` as `g`) và mục 2 (render `Q` as `q`) khỏi checklist BẮT BUỘC.
- ✅ **THAY THẾ** bằng: đảm bảo font pool có đủ serif/slab serif (Rockwell Bold, Georgia Bold, etc.) để tự nhiên tạo ra variant "9 đuôi cong" và "Q descender dài" mà KHÔNG cần lowercase substitution.
- `G_VARIANT_PROB = 0.0`, `Q_VARIANT_PROB = 0.0` trong code.

**Bài học**: khi phân tích ảnh 128×128 bằng mắt ở resolution thấp, dễ nhầm font variant (serif `9` vs sans `9`) thành lowercase substitution. Cần verify bằng cách render thử rồi so sánh trực tiếp.

**Checklist synth cập nhật**: tổng giảm từ 35 mục → **33 mục** (bỏ mục 1 và 2 cũ, thay bằng "font diversity đủ 3 category").

### 13.8 ĐÍNH CHÍNH: Không có mode "thưa" (easy spacing) trong real

**Sai lầm trước đó**: Synth có 20% mẫu dùng spacing rộng (17-20px) tạo ra ảnh mà các chữ cách nhau rõ rệt, không chồng lấn.

**Thực tế**: Trong toàn bộ 83 ảnh real đã phân tích, **KHÔNG CÓ** ảnh nào mà 5 ký tự đứng rời nhau kiểu "spaced out". Tất cả đều có overlap hoặc ít nhất near-touching. Ảnh "dễ" nhất trong real (ví dụ `map_00029 Q9YJA`, `map_00038 T9LFR`) vẫn có glyph gần sát nhau, chỉ là overlap ít hơn.

**Giải pháp**: bỏ hoàn toàn mode "easy spacing". Tất cả mẫu synth dùng spacing **12-17px** (luôn overlap hoặc near-touching).

**Lý do ưu tiên cao**: ảnh synth thưa trông hoàn toàn khác real → model học pattern "chữ rời" không tồn tại trong real → wasted capacity + potential confusion.
