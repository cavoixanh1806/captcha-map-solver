# Hướng dẫn Chạy lệnh - CAPTCHA Solver (TrOCR)

Tài liệu này hướng dẫn chi tiết các bước quản lý dữ liệu, đồng bộ hóa mã nguồn qua Git, chuẩn bị cấu hình và chạy huấn luyện (train) model trên máy local hoặc máy chủ **RTX 3090/3090 Ti** từ xa.

---

## 1. Thêm Dữ liệu ảnh mới (Import Data)

Khi bạn thu thập được thêm các ảnh CAPTCHA mới cần gán nhãn và đưa vào tập train:

### Cách A: Nạp ảnh mới gán nhãn từ thư mục `ac/` (Mặc định)
1. Copy toàn bộ ảnh mới vào thư mục `ac/` ở gốc dự án.
   * *Định dạng tên file:* `map_[LABEL]_[TIMESTAMP]_[HASH].png` hoặc `map_[LABEL].png` (Ví dụ: `map_33JJP.png`).
2. Chạy lệnh import tự động:
   ```bash
   .venv\Scripts\python import_data.py
   ```

### Cách B: Nạp ảnh sửa sai từ thư mục `checker/` (Dành cho dán lại nhãn lỗi)
1. Chạy gán nhãn và kiểm tra ảnh sai trong thư mục `checker/`.
2. Chạy lệnh import chỉ định thư mục:
   ```bash
   .venv\Scripts\python import_data.py --dir checker
   ```

> [!NOTE]
> * Tool sẽ tự động chuẩn hóa tên ảnh dạng index nối tiếp (`map_01553.png`,...), copy vào thư mục `data/` và tự cập nhật nhãn vào `data/metadata.csv`.
> * Sau khi chạy thành công, hãy dọn dẹp sạch các file ảnh cũ trong thư mục nguồn (`ac/` hoặc `checker/`) để tránh bị trùng lặp ở lần sau.

---

## 2. Cấu hình Kích thước Dataset (BẮT BUỘC)

Vì dự án sử dụng cấu chế độ kiểm tra kích thước phân tách tập dữ liệu rất nghiêm ngặt, **mỗi khi bạn nạp thêm dữ liệu mới**, hãy thực hiện 2 bước sau:

1. **Xóa file splits cũ:**
   * **Windows:** Xóa file `configs\splits.json` hoặc chạy lệnh:
     ```cmd
     del configs\splits.json
     ```
   * **Linux:** Chạy lệnh:
     ```bash
     rm configs/splits.json
     ```
2. **Cập nhật kích thước train trong các file config YAML:**
   Mở file cấu hình bạn muốn chạy (ví dụ: `configs/trocr_base_3090ti.yaml`) và sửa thông số `train_size` theo công thức:
   $$\text{train\_size} = \text{Tổng số ảnh hiện có} - \text{val\_size} (50) - \text{test\_size} (50)$$
   *Ví dụ hiện tại với 1.580 ảnh:*
   ```yaml
   data:
     train_size: 1480  # = 1580 - 50 - 50
     val_size: 50
     test_size: 50
   ```

---

## 3. Đồng bộ hóa và Thiết lập trên máy 3090 từ đầu (Git-Centric Workflow)

Để đồng bộ toàn bộ dữ liệu mới nhất (gồm ảnh, metadata và cấu hình) từ máy gán nhãn hiện tại sang máy chủ RTX 3090:

### Bước 1: Đẩy dữ liệu lên Git (Chạy trên máy gán nhãn)
Chạy tổ hợp lệnh Git sau để lưu trữ dữ liệu lên cloud:
```bash
git add data/ configs/ setup.bat import_data.py
git commit -m "Import new labeled data, update dataset configs"
git push
```

### Bước 2: Thiết lập trên máy 3090 mới (Windows/Linux)
1. **Tải mã nguồn về:**
   ```bash
   git clone https://github.com/cavoixanh1806/captcha-map-solver.git
   cd captcha-map-solver
   ```
2. **Cài đặt môi trường cực kỳ nhanh:**
   * **Trên Windows:** Chỉ cần click đúp vào file **`setup.bat`**. Nó sẽ tự động chuẩn bị virtual environment, nâng cấp pip, cài đặt PyTorch CUDA optimized và toàn bộ thư viện dependencies.
   * **Trên Linux:** Chạy các lệnh:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     pip install -r requirements.txt
     ```

---

## 4. Các Lệnh Huấn luyện (Train) trên máy 3090

Bạn có thể chạy các lệnh huấn luyện trực tiếp bằng cách copy-paste dòng lệnh tương ứng (không cần kích hoạt môi trường ảo thủ công):

### Cách A: Huấn luyện từ đầu (From Scratch)
Huấn luyện tiếp tục fine-tuning từ mô hình nền gốc của Hugging Face (`microsoft/trocr-base-printed`):
```bash
.venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml
```

### Cách B: Huấn luyện nâng cao kết hợp Trộn ảnh Nhân tạo (Khuyên dùng)
Trộn thêm 2.000 ảnh CAPTCHA sinh ngẫu nhiên mỗi epoch để mô hình học các biến dạng nét và màu sắc phong phú, tăng độ chính xác vượt trội **90%+**:
```bash
.venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml --synth 2000
```

### Cách C: Huấn luyện tiếp tục từ Checkpoint cũ (`best-epoch052.ckpt`)
1. Đặt file `best-epoch052.ckpt` của bạn vào thư mục gốc của dự án.
2. Chạy lệnh:
   ```bash
   .venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml --resume best-epoch052.ckpt
   ```
   *Mô hình sẽ khôi phục optimizer, scheduler và tự động chạy tiếp tục từ **Epoch 53**.*
