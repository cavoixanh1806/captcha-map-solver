# Hướng dẫn Setup và Chạy Train - CAPTCHA Solver (TrOCR)

Tài liệu này hướng dẫn nhanh các bước thiết lập môi trường (Setup) và bắt đầu chạy huấn luyện (Train) mô hình TrOCR trên máy local hoặc máy chủ **RTX 3090/3090 Ti** từ xa.

---

## 1. Hướng dẫn Setup Môi trường

Sau khi clone hoặc tải mã nguồn về máy, bạn thực hiện cài đặt môi trường theo hệ điều hành tương ứng:

### A. Trên hệ điều hành Windows:
git clone https://github.com/cvx1806/captcha-map-solver.git
cd captcha-map-solver
Bạn chỉ cần click đúp vào file **`setup.bat`** ở gốc dự án. Nó sẽ tự động:
* Khởi tạo môi trường ảo `.venv`
* Nâng cấp pip và cài đặt PyTorch CUDA optimized (hỗ trợ CUDA 12.1 đến 12.8+)
* Cài đặt toàn bộ thư viện cần thiết (`requirements.txt`)

### B. Trên hệ điều hành Linux (Ubuntu):
Mở terminal tại thư mục dự án và chạy tuần tự các lệnh sau:
```bash
# 1. Khởi tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Nâng cấp pip và cài đặt PyTorch hỗ trợ CUDA tối ưu cho RTX 3090
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 2. Các Lệnh Huấn luyện (Train)

Bạn chạy lệnh huấn luyện tương ứng bằng cách copy-paste trực tiếp dòng lệnh dưới đây (không cần kích hoạt môi trường ảo thủ công):

### Cách A: Huấn luyện mới từ đầu (From Scratch)
Bắt đầu huấn luyện từ mô hình pre-trained gốc của Hugging Face (`microsoft/trocr-base-printed`):
```bash
.venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml
```

### Cách B: Huấn luyện kết hợp Trộn ảnh Nhân tạo (Khuyên dùng)
Trộn thêm 2.000 ảnh CAPTCHA sinh ngẫu nhiên mỗi epoch để mô hình học thêm nét chữ và màu sắc phong phú, tăng độ chính xác vượt trội **90%+**:
```bash
.venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml --synth 2000
```

### Cách C: Huấn luyện tiếp tục từ Checkpoint cũ (`best-epoch052.ckpt`)
1. Đặt file checkpoint cũ (ví dụ: `best-epoch052.ckpt`) vào thư mục gốc của dự án.
2. Chạy lệnh:
   ```bash
   .venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml --synth 2000 --resume best-epoch032.ckpt
   ```
   *Mô hình sẽ tự động khôi phục optimizer, scheduler và chạy tiếp tục từ **Epoch 32**.*
