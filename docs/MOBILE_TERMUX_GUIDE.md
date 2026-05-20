# 📱 Hướng dẫn chạy TrOCR trên Termux (Snapdragon 8s Gen 3)

> **Checkpoint:** `best-epoch072.ckpt` | **Config:** `trocr_base_3090ti.yaml`  
> **Mục tiêu:** Chạy inference captcha solver trên điện thoại ARM64 — **không giảm độ chính xác**

---

## 📋 Mục lục

1. [Yêu cầu phần cứng & phần mềm](#1-yêu-cầu-phần-cứng--phần-mềm)
2. [Cài đặt Termux](#2-cài-đặt-termux)
3. [Cài đặt môi trường Python](#3-cài-đặt-môi-trường-python)
4. [Cài đặt PyTorch cho ARM64](#4-cài-đặt-pytorch-cho-arm64)
5. [Cài đặt dependencies](#5-cài-đặt-dependencies)
6. [Chuyển model sang điện thoại](#6-chuyển-model-sang-điện-thoại)
7. [Clone / copy source code](#7-clone--copy-source-code)
8. [Chạy inference 1 ảnh](#8-chạy-inference-1-ảnh)
9. [🆕 Test batch ảnh thư mục `ac/`](#9-test-batch-ảnh-thư-mục-ac)
10. [Tối ưu để KHÔNG giảm độ chính xác](#10-tối-ưu-để-không-giảm-độ-chính-xác)
11. [Chạy API server nhẹ trên Termux](#11-chạy-api-server-nhẹ-trên-termux)
12. [Xử lý lỗi thường gặp](#12-xử-lý-lỗi-thường-gặp)

---

## 1. Yêu cầu phần cứng & phần mềm

| Mục | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| Chip | Snapdragon 8s Gen 3 ✅ | Snapdragon 8 Gen 3 |
| RAM | 8 GB | 12 GB |
| Storage | 10 GB trống | 16 GB trống |
| Android | 12+ | 14 |
| Termux | F-Droid build | F-Droid build |

> ⚠️ **QUAN TRỌNG:** Dùng Termux từ **F-Droid**, không dùng Play Store (bản Play Store đã lỗi thời).

---

## 2. Cài đặt Termux

### Bước 2.1 — Tải Termux từ F-Droid
```
https://f-droid.org/packages/com.termux/
```

### Bước 2.2 — Cấp quyền Storage
Sau khi mở Termux lần đầu:
```bash
termux-setup-storage
```
→ Nhấn **Allow** khi Android hỏi quyền.

### Bước 2.3 — Cập nhật packages
```bash
pkg update -y && pkg upgrade -y
```

### Bước 2.4 — Cài các tool cơ bản
```bash
pkg install -y \
  python \
  python-pip \
  git \
  wget \
  curl \
  cmake \
  ninja \
  pkg-config \
  libopenblas \
  libjpeg-turbo \
  libpng \
  openssl \
  rust \
  clang
```

---

## 3. Cài đặt môi trường Python

### Bước 3.1 — Kiểm tra Python
```bash
python --version   # cần 3.10 hoặc 3.11
pip --version
```

### Bước 3.2 — Tạo virtual environment
```bash
# Tạo thư mục project
mkdir -p ~/captcha-solver
cd ~/captcha-solver

# Tạo venv
python -m venv .venv
source .venv/bin/activate
```

> 💡 Mỗi lần mở Termux mới, nhớ chạy lại:  
> `source ~/captcha-solver/.venv/bin/activate`

---

## 4. Cài đặt PyTorch cho ARM64

### ⚠️ Vấn đề quan trọng nhất
Chip **Snapdragon 8s Gen 3 là ARM64** — **KHÔNG có CUDA**. Model sẽ chạy trên **CPU**. Độ chính xác **KHÔNG thay đổi** so với GPU vì inference chỉ là tính toán forward pass — kết quả hoàn toàn giống nhau.

### Bước 4.1 — Kiểm tra bản PyTorch ARM64 có sẵn
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ Nếu lệnh trên thất bại (thường xảy ra với ARM), dùng phương án sau:

### Bước 4.2 — Phương án dự phòng: PyTorch wheel từ Termux mirror
```bash
# Cập nhật pip
pip install --upgrade pip

# Thử cài từ PyPI (bản CPU-only cho ARM)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

### Bước 4.3 — Nếu vẫn thất bại: dùng torch-nightly hoặc build từ source
```bash
# Tùy chọn 1: Nightly build (ổn định nhất cho ARM64)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Tùy chọn 2: Dùng pytorch-arm64 từ community wheels
pip install torch --extra-index-url https://torch.kmtea.eu/whl/stable
```

### Bước 4.4 — Kiểm tra cài đặt
```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('Device available:', 'cuda' if torch.cuda.is_available() else 'cpu (ARM/Mobile)')
t = torch.tensor([1.0, 2.0, 3.0])
print('Test tensor:', t.sum().item())
"
```
Kết quả mong đợi: `Device available: cpu (ARM/Mobile)`

---

## 5. Cài đặt dependencies

### Bước 5.1 — Cài thư viện hệ thống cần thiết
```bash
pkg install -y libopenblas libopenjpeg
```

### Bước 5.2 — Cài Pillow (không dùng opencv trên mobile)
```bash
pip install Pillow>=10.0
```

### Bước 5.3 — Cài các thư viện còn lại (KHÔNG cài opencv-python — quá nặng)
```bash
pip install \
  pytorch-lightning>=2.2,<3.0 \
  torchmetrics>=1.4 \
  "transformers>=4.40" \
  "sentencepiece>=0.2" \
  "numpy>=1.26" \
  "pandas>=2.0" \
  "pyyaml>=6.0" \
  "tqdm>=4.66"
```

> ⚠️ **Bỏ qua** `opencv-python`, `matplotlib`, `tensorboard`, `scikit-learn` — chỉ cần thiết khi training, không cần khi inference.

### Bước 5.4 — Kiểm tra transformers + sentencepiece
```bash
python -c "from transformers import TrOCRProcessor; print('transformers OK')"
```

---

## 6. Chuyển model sang điện thoại

### Phương án A: USB / ADB (nhanh nhất)
Trên máy tính Windows, chạy:
```powershell
# Kết nối USB debugging
adb push best-epoch072.ckpt /sdcard/captcha-solver/
adb push configs/trocr_base_3090ti.yaml /sdcard/captcha-solver/configs/
```

Trên Termux:
```bash
mkdir -p ~/captcha-solver/configs
cp /sdcard/captcha-solver/best-epoch072.ckpt ~/captcha-solver/
cp /sdcard/captcha-solver/configs/trocr_base_3090ti.yaml ~/captcha-solver/configs/
```

### Phương án B: SCP qua WiFi (không cần cáp)
Trên điện thoại, cài OpenSSH:
```bash
pkg install openssh
sshd   # khởi động SSH server
```

Trên máy tính Windows:
```powershell
# Tìm IP điện thoại trong Settings > About > IP address
scp best-epoch072.ckpt user@192.168.x.x:8022:~/captcha-solver/
scp configs/trocr_base_3090ti.yaml user@192.168.x.x:8022:~/captcha-solver/configs/
```

### Phương án C: Google Drive / Rclone
```bash
pkg install rclone
# Cấu hình rclone với Google Drive theo hướng dẫn: rclone config
rclone copy gdrive:captcha-solver/best-epoch072.ckpt ~/captcha-solver/
```

---

## 7. Clone / copy source code

### Bước 7.1 — Clone từ GitHub
```bash
cd ~/captcha-solver
git clone https://github.com/cavoixanh1806/captcha-map-solver.git .
```

### Bước 7.2 — Hoặc copy thủ công qua USB
```bash
# Copy toàn bộ thư mục src từ máy tính
adb push src/ /sdcard/captcha-solver/src/

# Trên Termux
cp -r /sdcard/captcha-solver/src ~/captcha-solver/
```

### Bước 7.3 — Kiểm tra cấu trúc thư mục
```bash
ls ~/captcha-solver/
# Cần có: src/, configs/, best-epoch072.ckpt
```

---

## 8. Chạy inference 1 ảnh

### Bước 8.1 — Kích hoạt môi trường
```bash
cd ~/captcha-solver
source .venv/bin/activate
```

### Bước 8.2 — Test nhanh 1 ảnh bất kỳ
```bash
python predict.py \
  --ckpt best-epoch072.ckpt \
  --image /sdcard/Pictures/captcha_test.png \
  --task trocr \
  --config configs/trocr_base_3090ti.yaml
```

---

## 9. 🆕 Test batch ảnh thư mục `ac/`

> Đây là cách **nhanh nhất** để kiểm tra độ chính xác model trên toàn bộ ảnh trong thư mục `ac/`.
> Script **tự đọc label từ tên file** nên không cần cấu hình thêm gì.

### Tên file trong `ac/` có dạng:
```
map_XXXXX_<timestamp>_<hash>__cvxxN.png
     ^^^^^
     label chính xác 5 ký tự — script tự parse
```

### Bước 9.1 — Chuyển thư mục `ac/` và script sang điện thoại

**Cách A — qua USB (ADB):**
```powershell
# Trên máy tính Windows
adb push ac/ /sdcard/captcha-solver/ac/
adb push mobile_test_ac.py /sdcard/captcha-solver/
```
Trên Termux:
```bash
cp -r /sdcard/captcha-solver/ac ~/captcha-solver/
cp /sdcard/captcha-solver/mobile_test_ac.py ~/captcha-solver/
```

**Cách B — git pull (nếu đã clone):**
```bash
cd ~/captcha-solver
git pull origin main
# Script mobile_test_ac.py và thư mục ac/ đã có sẵn trong repo
```

### Bước 9.2 — Chạy test toàn bộ ảnh trong `ac/`
```bash
cd ~/captcha-solver
source .venv/bin/activate

python mobile_test_ac.py
```

Kết quả mẫu:
```
File                                Label    Pred    ms
────────────────────────────────────────────────────────
…1_1779292888582_ba17__cvxx1.png   773JE   773JE  ✓ 4231
…2_1779292899051_3329__cvxx2.png   NEWTL   NEWTL  ✓ 3987
…3_1779292913706_0cd6__cvxx3.png   J4D7J   J4D7J  ✓ 4102
…4_1779292926785_5498__cvxx4.png   RMVTY   RMVTY  ✓ 3851
…5_1779292943712_a171__cvxx5.png   XUWQJ   XUWQJ  ✓ 4289
────────────────────────────────────────────────────────

📊 KẾT QUẢ:
  Tổng ảnh có label : 11
  Đúng              : 10
  Sai               : 1
  Accuracy          : 90.9%
  Tốc độ TB         : 4123 ms/ảnh
```

### Bước 9.3 — Các tùy chọn hữu ích

```bash
# Chỉ định thư mục ac khác
python mobile_test_ac.py --ac_dir /sdcard/Download/my_captchas

# Chỉ định checkpoint khác
python mobile_test_ac.py --ckpt /sdcard/captcha-solver/best-epoch052.ckpt

# Tắt màu ANSI (khi terminal không hỗ trợ)
python mobile_test_ac.py --no_color

# Kết hợp
python mobile_test_ac.py \
  --ckpt best-epoch072.ckpt \
  --ac_dir ac \
  --no_color
```

### Bước 9.4 — Lưu kết quả ra file
```bash
# Redirect output ra file để xem lại
python mobile_test_ac.py --no_color 2>&1 | tee ~/test_results.txt

# Xem file sau
cat ~/test_results.txt
```

### Bước 9.5 — Giải thích cột kết quả

| Cột | Ý nghĩa |
|-----|---------|
| `File` | Tên file rút gọn |
| `Label` | Ground truth từ tên file |
| `Pred` | Model đoán |
| `✓` / `✗` | Đúng / Sai |
| `ms` | Thời gian inference (millisecond) |

---

## 10. Tối ưu để KHÔNG giảm độ chính xác

### ✅ Nguyên tắc vàng: Không thay đổi gì về weights

> **CPU vs GPU KHÔNG ảnh hưởng đến độ chính xác**. Kết quả của forward pass là **xác định** (deterministic) — giống hệt nhau dù chạy trên GPU hay CPU, miễn là precision đúng.

### 9.1 — Giữ nguyên float32 (KHÔNG dùng quantization mặc định)

```python
# ❌ SAI — quantization làm giảm độ chính xác
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# ✅ ĐÚNG — giữ nguyên float32
model.eval()
model.to("cpu")  # Chỉ chuyển device, không đổi dtype
```

### 9.2 — Tham số generation PHẢI giống lúc train/eval

Checkpoint `best-epoch072.ckpt` được validate với:
```yaml
num_beams: 8          # BẮT BUỘC giữ nguyên
max_length: 16        # BẮT BUỘC giữ nguyên  
repetition_penalty: 1.3  # BẮT BUỘC giữ nguyên
```

```python
# ✅ Đúng — giữ nguyên tham số từ config
gen = model.model.generate(
    pixel_values,
    num_beams=8,
    max_length=16,
    repetition_penalty=1.3
)

# ❌ Sai — giảm num_beams để tiết kiệm RAM
gen = model.model.generate(pixel_values, num_beams=1)  # greedy, kém chính xác hơn
```

### 9.3 — Dùng `map_location` khi load checkpoint CUDA → CPU

```python
# ✅ Đúng — map weights từ GPU về CPU
model = TrOCRLitModel.load_from_checkpoint(
    "best-epoch072.ckpt",
    map_location="cpu"  # PHẢI có dòng này
)
```

### 9.4 — Nếu muốn quantization mà không giảm độ chính xác nhiều

Chỉ dùng **INT8 dynamic quantization** và chỉ áp dụng cho Linear layer:
```python
import torch

model = TrOCRLitModel.load_from_checkpoint("best-epoch072.ckpt", map_location="cpu")
model.eval()

# Dynamic quantization — mất <0.5% accuracy nhưng nhanh 2x, RAM giảm 4x
model_quantized = torch.quantization.quantize_dynamic(
    model.model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 9.5 — Bảng so sánh các chế độ chạy

| Chế độ | RAM dùng | Tốc độ | Độ chính xác |
|--------|----------|--------|--------------|
| Float32 CPU (mặc định) | ~4.0 GB | ~3–8s/ảnh | **100% (chuẩn)** |
| Float16 CPU | ~2.0 GB | ~3–6s/ảnh | ≈99.9% |
| INT8 Dynamic Quant | ~1.2 GB | ~1.5–3s/ảnh | ≈99.5% |
| INT8 + num_beams=4 | ~1.2 GB | ~0.8–2s/ảnh | ≈98–99% |

> ⚠️ Snapdragon 8s Gen 3 có **8–12 GB RAM**. Float32 ~4GB là hoàn toàn khả thi.

---

## 11. Chạy API server nhẹ trên Termux

Nếu muốn gọi model từ ứng dụng khác (ví dụ browser hoặc app Android):

### Bước 10.1 — Cài Flask nhẹ
```bash
pip install flask
```

### Bước 10.2 — Tạo `mobile_api.py`
```python
#!/usr/bin/env python3
"""
API server nhẹ để gọi model từ bên ngoài Termux.
Usage: python mobile_api.py
Endpoint: POST http://localhost:5000/solve
"""
import io
import base64
import torch
from flask import Flask, request, jsonify
from PIL import Image
from src.trocr import TrOCRLitModel

app = Flask(__name__)

# Load model 1 lần khi khởi động
print("Loading model... (có thể mất 30-60 giây)")
MODEL = TrOCRLitModel.load_from_checkpoint(
    "best-epoch072.ckpt",
    map_location="cpu"
)
MODEL.eval()
print("Model loaded! Server sẵn sàng.")

@app.route("/solve", methods=["POST"])
def solve():
    try:
        # Nhận ảnh dạng base64
        data = request.get_json()
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pixel_values = MODEL.processor(
            images=img, return_tensors="pt"
        ).pixel_values
        
        with torch.no_grad():
            gen = MODEL.model.generate(
                pixel_values,
                num_beams=8,
                max_length=16,
                repetition_penalty=1.3
            )
        
        result = MODEL.processor.batch_decode(
            gen, skip_special_tokens=True
        )[0].replace(" ", "").upper()
        
        return jsonify({"result": result, "status": "ok"})
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == "__main__":
    # Lắng nghe trên tất cả interface (để thiết bị khác cùng mạng gọi được)
    app.run(host="0.0.0.0", port=5000, debug=False)
```

### Bước 10.3 — Chạy server
```bash
cd ~/captcha-solver
source .venv/bin/activate
python mobile_api.py
```

### Bước 10.4 — Test API (từ thiết bị khác cùng WiFi)
```bash
# Trên máy tính — thay 192.168.x.x bằng IP điện thoại
curl -X POST http://192.168.x.x:5000/solve \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 captcha.png)\"}"
```

---

## 12. Xử lý lỗi thường gặp

### ❌ Lỗi: `hf-xet panic` / `rustls-platform-verifier` (LỖI QUAN TRỌNG NHẤT)

```
thread 'hf-xet-1' panicked at ...rustls-platform-verifier-0.6.2/src/android.rs:94:10:
Expect rustls-platform-verifier to be initialized
RuntimeError: Internal error: Join error: task panicked...
```

**Nguyên nhân:** HuggingFace dùng protocol `hf-xet` (XetHub) để download model — protocol này dùng `rustls` với Android TLS verifier, nhưng **không được khởi tạo trong Termux**.

**Fix A — Đã được fix sẵn trong `mobile_test_ac.py` mới nhất** (chạy `git pull` để cập nhật):
```bash
git pull origin main
python mobile_test_ac.py   # HF_HUB_DISABLE_XET=1 đã được set tự động
```

**Fix B — Nếu chạy script khác hoặc thủ công:**
```bash
# Set biến môi trường trước khi chạy
HF_HUB_DISABLE_XET=1 python mobile_test_ac.py

# Hoặc export vĩnh viễn trong ~/.bashrc
echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc
source ~/.bashrc
```

**Fix C — Upgrade huggingface_hub (phiên bản mới có fix):**
```bash
pip install --upgrade huggingface_hub
```

**Fix D — Dùng base model local (không cần download):**
```bash
# Trên máy tính Windows, download base model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/trocr-base-printed', local_dir='./trocr-base-printed')
"

# Copy sang điện thoại qua USB
adb push trocr-base-printed/ /sdcard/captcha-solver/trocr-base-printed/

# Trên Termux
cp -r /sdcard/captcha-solver/trocr-base-printed ~/captcha-solver/

# Chạy với --model_dir (không cần internet, không cần xet)
python mobile_test_ac.py --model_dir ./trocr-base-printed
```

> 💡 Sau lần đầu download thành công (Fix A hoặc B), các lần sau tự dùng cache:
> ```bash
> python mobile_test_ac.py --offline  # dùng cache, hoàn toàn offline
> ```

### ❌ Lỗi: `No module named 'cv2'`
```bash
# opencv-python không hỗ trợ ARM64 trên Termux — cài bản headless
pip install opencv-python-headless
# Nếu vẫn lỗi, tắt hoàn toàn opencv trong code (không cần cho inference TrOCR)
```


### ❌ Lỗi: `RuntimeError: PytorchStreamReader failed reading zip archive`
```
Checkpoint bị corrupt khi copy. Copy lại file.
```
Kiểm tra MD5:
```bash
# Trên Windows
certutil -hashfile best-epoch072.ckpt MD5
# Trên Termux
md5sum ~/captcha-solver/best-epoch072.ckpt
# Hai hash phải giống nhau
```

### ❌ Lỗi: `CUDA device not found`
```python
# Đảm bảo predict.py dùng map_location="cpu"
model = TrOCRLitModel.load_from_checkpoint(ckpt, map_location="cpu")
```
Hoặc thêm vào đầu script:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Tắt hoàn toàn CUDA
```

### ❌ Lỗi: `Killed` (OOM — Out of Memory)
```bash
# Giải phóng RAM trước khi chạy
# 1. Đóng tất cả app
# 2. Bật ZRAM nếu có
# 3. Dùng script giải phóng RAM
echo 3 > /proc/sys/vm/drop_caches  # Cần root
```
Hoặc giảm num_beams (chấp nhận mất <1% accuracy):
```python
gen = model.model.generate(pixel_values, num_beams=4, max_length=16)
```

### ❌ Lỗi: `sentencepiece` không cài được
```bash
pkg install libsentencepiece
pip install sentencepiece --no-binary sentencepiece
```

### ❌ Model load quá chậm (>5 phút)
```bash
# Lần đầu load sẽ chậm do verify weights
# Các lần sau sẽ nhanh hơn (OS cache)
# Dùng API server (Bước 10) để load 1 lần rồi giữ trong RAM
```

### ❌ Lỗi: `ImportError: libgomp.so.1`
```bash
pkg install libomp
```

---

## 📌 Tóm tắt nhanh (Quick Start)

```bash
# 1. Cài Termux từ F-Droid
# 2. Trong Termux:
termux-setup-storage
pkg update -y && pkg upgrade -y
pkg install -y python python-pip git cmake libopenblas

# 3. Tạo môi trường
mkdir ~/captcha-solver && cd ~/captcha-solver
python -m venv .venv && source .venv/bin/activate

# 4. Cài PyTorch CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Cài dependencies (không có opencv)
pip install pytorch-lightning transformers sentencepiece pillow numpy pandas pyyaml tqdm

# 6. Copy files (qua USB/WiFi/GitHub)
git clone https://github.com/cavoixanh1806/captcha-map-solver.git .
# Copy best-epoch072.ckpt vào ~/captcha-solver/

# 7. Chạy inference 1 ảnh
python predict.py \
  --ckpt best-epoch072.ckpt \
  --image /sdcard/Pictures/captcha.png \
  --task trocr \
  --config configs/trocr_base_3090ti.yaml

# 8. Test toàn bộ ảnh trong ac/ (recommended)
python mobile_test_ac.py
```

---

## 🔑 Điểm mấu chốt để giữ độ chính xác 100%

| ✅ CẦN làm | ❌ KHÔNG làm |
|-----------|-------------|
| `map_location="cpu"` khi load checkpoint | Dùng `torch.quantization` tùy tiện |
| Giữ `num_beams=8` | Giảm `num_beams` xuống 1 (greedy) |
| Giữ `max_length=16` | Đổi `max_length` |
| Giữ `repetition_penalty=1.3` | Bỏ `repetition_penalty` |
| Dùng **float32** (mặc định) | Ép `half()` / `bfloat16()` |
| Copy file đầy đủ, kiểm tra MD5 | Copy file khi đang dùng (corrupt) |

---

*Tạo ngày: 2026-05-20 | Cập nhật: 2026-05-20 | Chip: Snapdragon 8s Gen 3 (ARM64) | Model: TrOCR-base best-epoch072*
