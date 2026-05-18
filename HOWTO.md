# Hướng dẫn Chạy lệnh - CAPTCHA Solver (TrOCR)

Tài liệu này hướng dẫn chi tiết các bước quản lý dữ liệu, chuẩn bị cấu hình và chạy huấn luyện (train) model trên **Windows 10** (hoặc đồng bộ lên **Ubuntu Server**).

---

## 1. Thêm Dữ liệu ảnh mới (Import Data)

Khi bạn thu thập được thêm các ảnh CAPTCHA mới cần gán nhãn và đưa vào tập train:

1. Copy toàn bộ ảnh mới vào thư mục `ac/` ở gốc dự án.
   * *Định dạng tên file yêu cầu:* `map_[LABEL].png` (Ví dụ: `map_33JJP.png`).
2. Mở Terminal (CMD / PowerShell) tại thư mục dự án và chạy công cụ import tự động:
   ```bash
   python import_data.py
   ```
3. **Kết quả:** Tool sẽ tự động đổi tên ảnh thành dạng index nối tiếp (`map_00500.png`,...), copy vào thư mục `data/` và tự động cập nhật nhãn vào file `data/metadata.csv`.
4. **Lưu ý:** Sau khi tool chạy thành công, hãy **xóa toàn bộ ảnh cũ trong thư mục `ac/`** đi để tránh bị import trùng lặp ở lần tiếp theo.

---

## 2. Chuẩn bị trước khi chạy Train (BẮT BUỘC)

Mỗi khi **thay đổi số lượng ảnh** (sau khi import thêm ảnh mới), bạn bắt buộc phải xóa file split cũ để hệ thống chia lại tập Train/Val/Test cho chuẩn xác:

* **Trên Windows:** Vào thư mục `configs/` và xóa file `splits.json`.
* **Hoặc xóa bằng lệnh (CMD/PowerShell):**
  ```bash
  del configs\splits.json
  ```
* **Trên Linux (Ubuntu):**
  ```bash
  rm configs/splits.json
  ```

---

## 3. Lệnh chạy Huấn luyện (Train Model)

Mở Terminal (CMD/PowerShell đã active virtual environment `.venv`) và chọn một trong hai cách chạy dưới đây:

### Cách A: Train lại từ đầu (Fine-tuning từ gốc)
Sử dụng khi bạn muốn train lại từ đầu với trọng số gốc của Hugging Face:
```bash
python train.py --config configs/trocr_base_3090ti.yaml
```

### Cách B: Train tiếp tục từ Checkpoint tốt nhất (VIP nhất)
Sử dụng khi bạn muốn model học tiếp từ những gì đã tích lũy trước đó (rất khuyên dùng sau khi thêm data mới để tiết kiệm thời gian):

1. Truy cập thư mục `checkpoints/trocr-base-3090ti/` để tìm file checkpoint tốt nhất của bạn.
   * File tốt nhất thường có tên dạng: `best-epochXXX.ckpt` (Ví dụ: `best-epoch042.ckpt`).
2. Chạy lệnh train kèm tham số `--resume`:
   ```bash
   python train.py --config configs/trocr_base_3090ti.yaml --resume checkpoints/trocr-base-3090ti/best-epoch042.ckpt
   ```
   *(Thay thế `best-epoch042.ckpt` bằng tên file checkpoint xịn nhất thực tế của bạn).*

---

## 4. Đồng bộ dữ liệu lên Server Ubuntu (Nếu cần)

Nếu sau này bạn muốn chuyển qua train trên Server Ubuntu từ xa (IP: `127.0.0.1`, Port: `13389`):

### Bước 1: Đẩy dữ liệu từ Windows lên Ubuntu (Chạy trên Windows)
1. **Đẩy Config đã sửa:**
   ```powershell
   scp -P 13389 C:\Users\Administrator\Desktop\TrainAI\configs\trocr_base_3090ti.yaml ezycloudx-admin@127.0.0.1:/home/ezycloudx-admin/captcha-map-solver/configs/
   ```
2. **Đẩy Metadata mới:**
   ```powershell
   scp -P 13389 C:\Users\Administrator\Desktop\TrainAI\data\metadata.csv ezycloudx-admin@127.0.0.1:/home/ezycloudx-admin/captcha-map-solver/data/
   ```
3. **Đồng bộ thư mục ảnh `data/`:**
   ```powershell
   scp -P 13389 -r C:\Users\Administrator\Desktop\TrainAI\data ezycloudx-admin@127.0.0.1:/home/ezycloudx-admin/captcha-map-solver/
   ```

### Bước 2: Chạy lệnh trên Server (Chạy trực tiếp trên Ubuntu Terminal)
1. **Di chuyển vào thư mục dự án:**
   ```bash
   cd /home/ezycloudx-admin/captcha-map-solver
   ```
2. **Xóa file split cũ để nhận dữ liệu mới:**
   ```bash
   rm configs/splits.json
   ```
3. **Chạy train:**
   ```bash
   python train.py --config configs/trocr_base_3090ti.yaml
   ```
