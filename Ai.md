# 🚀 Hướng dẫn chạy TrOCR Captcha API trên Điện thoại (Snapdragon 8s Gen 3)

Tài liệu này hướng dẫn cách cài đặt và chạy Server giải mã CAPTCHA tối ưu nhất cho chip Snapdragon, sử dụng **Node.js** và **Transformers.js (ONNX)**.

---

## 🛠 Bước 1: Cài đặt môi trường (Termux)
Tải App **Termux** từ [F-Droid](https://f-droid.org/packages/com.termux/). Mở Termux và gõ:

```bash
# Cập nhật hệ thống
pkg update -y && pkg upgrade -y

# Cài đặt Node.js và Python
pkg install nodejs python -y

# Cài đặt thư viện tải model
pip install huggingface_hub
```

---

## 📂 Bước 2: Tải Code và Cấu hình (Quan trọng)
**Lưu ý:** Bạn phải đưa code vào bộ nhớ riêng của Termux để tránh lỗi quyền truy cập file (EACCES).

```bash
# Di chuyển vào thư mục Home của Termux
cd ~

# Clone dự án (nếu chưa có)
git clone https://github.com/cavoixanh1806/captcha-map-solver.git

# Vào thư mục API
cd captcha-map-solver/api

# Cài đặt thư viện (Bắt buộc dùng --force để chạy trên Android)
npm install --force
```

---

## 🧠 Bước 3: Tải Model AI từ Hugging Face
Sử dụng lệnh Python để tránh lỗi xác thực SSL/Rust trên điện thoại:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cavoixanh1806/captcha-trocr-onnx', local_dir='../onnx_model', local_dir_use_symlinks=False)"
```

---

## 🏃 Bước 4: Khởi chạy API và Dashboard
Sau khi tải xong model, bạn khởi động server:

```bash
npm start
```

### 📱 Cách sử dụng giao diện App:
1. Mở trình duyệt (Chrome) trên điện thoại.
2. Truy cập: `http://127.0.0.1:5000`
3. Chọn **"Thêm vào màn hình chính"** để sử dụng như một ứng dụng thực thụ.

---

## 📝 Thông số API (Dành cho Auto/Tool)
- **Base URL:** `http://127.0.0.1:5000`
- **Endpoints:**
    - `GET /health`: Kiểm tra trạng thái.
    - `POST /solve-file`: Gửi file ảnh (key là `file`).
    - `POST /solve-base64`: Gửi JSON chứa chuỗi `image_base64`.

---
*Phát triển bởi cavoixanh1806 - Tối ưu cho Snapdragon 8s Gen 3 via WebGPU/Wasm.*
