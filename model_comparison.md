# So sánh Hiệu suất Mô hình TrOCR: Epoch 32 vs Epoch 72

Tài liệu này lưu lại kết quả đánh giá và so sánh chi tiết hiệu suất nhận diện captcha giữa hai checkpoint:
* **Model 1**: `best-epoch032.ckpt` (Được lưu ở epoch thứ 32)
* **Model 2**: `best-epoch072.ckpt` (Được lưu ở epoch thứ 72)

Phương pháp đánh giá: Chạy suy luận (inference) trên **500 ảnh đầu tiên** từ thư mục `data/` có nhãn tương ứng trong `data/metadata.csv`.

---

## 📊 Kết quả Tổng hợp

| Chỉ số | Model 32 (`best-epoch032.ckpt`) | Model 72 (`best-epoch072.ckpt`) |
| :--- | :---: | :---: |
| **Độ chính xác (Accuracy)** | **98.40%** | **98.80%** (Tốt nhất) |
| **Số ảnh đoán đúng** | 492 / 500 | 494 / 500 |
| **Số ảnh đoán sai** | 8 | 6 |
| **Số lỗi khắc phục được** | - | 7 trường hợp (M32 sai, M72 đúng) |
| **Số lỗi phát sinh mới** | - | 5 trường hợp (M32 đúng, M72 sai) |

> [!NOTE]
> Tổng số ảnh bị đoán sai bởi ít nhất một trong hai mô hình là **13 ảnh** (chiếm tỉ lệ 2.6%).

---

## 🔍 Danh sách Chi tiết các Trường hợp Dự đoán Sai

Dưới đây là danh sách toàn bộ các trường hợp có ít nhất một mô hình dự đoán sai:

| STT | Tên File | Nhãn Thật | Model 32 | Model 72 | Trạng thái chi tiết |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `map_00013.png` | **FYEVU** | `EYEVU` | `FYEVU` | 🟢 Model 72 khắc phục (Sửa `E` thành `F`) |
| **2** | `map_00065.png` | **VHMLW** | `VHMYW` | `VHMLW` | 🟢 Model 72 khắc phục (Sửa `Y` thành `L`) |
| **3** | `map_00093.png` | **YRRCD** | `YPRCD` | `YRRCD` | 🟢 Model 72 khắc phục (Sửa `P` thành `R`) |
| **4** | `map_00142.png` | **KEREA** | `KEREA` | `KENE4` | 🔴 Model 72 sai mới (Nhầm `R` $\rightarrow$ `N`, `A` $\rightarrow$ `4`) |
| **5** | `map_00178.png` | **7Q9WX** | `7C9WX` | `7Q9WX` | 🟢 Model 72 khắc phục (Sửa `C` thành `Q`) |
| **6** | `map_00189.png` | **P4RCU** | `P4RCU` | `P4KCU` | 🔴 Model 72 sai mới (Nhầm `R` $\rightarrow$ `K`) |
| **7** | `map_00198.png` | **FLEUE** | `FLFUE` | `FLEUE` | 🟢 Model 72 khắc phục (Sửa `F` thành `E`) |
| **8** | `map_00232.png` | **E3QPD** | `E3QPD` | `E3QLD` | 🔴 Model 72 sai mới (Nhầm `P` $\rightarrow$ `L`) |
| **9** | `map_00255.png` | **DKEAL** | `LKEAL` | `DKEAL` | 🟢 Model 72 khắc phục (Sửa `L` thành `D`) |
| **10** | `map_00298.png` | **FJKWN** | `FJKWN` | `FJNWN` | 🔴 Model 72 sai mới (Nhầm `K` $\rightarrow$ `N`) |
| **11** | `map_00326.png` | **KAEAX** | `KAAAX` | `KALAX` | ❌ Cả hai đều đoán sai (M32 nhầm `E` $\rightarrow$ `A`, M72 nhầm `E` $\rightarrow$ `L`) |
| **12** | `map_00335.png` | **NTMKR** | `NTMKR` | `N7MKR` | 🔴 Model 72 sai mới (Nhầm `T` $\rightarrow$ `7`) |
| **13** | `map_00407.png` | **WWRJV** | `VWRJV` | `WWRJV` | 🟢 Model 72 khắc phục (Sửa `V` thành `W`) |

---

## 🔬 Phân tích và Đánh giá

### 1. Nhận xét về Xu hướng Huấn luyện (Epoch 32 vs Epoch 72)
* **Khả năng khái quát hóa**: Khi tiếp tục huấn luyện từ epoch 32 lên 72, mô hình học được cách phân biệt tốt hơn giữa các ký tự có độ tương đồng cao về cấu trúc hình học (ví dụ: nét ngang của `F` vs `E`, độ cong của `Q` vs `C`, nét xiên chéo của `W` vs `V`).
* **Sự cân bằng (Trade-off)**: Mặc dù Model 72 sửa được tới **7 lỗi** của Model 32, nó lại làm hỏng **5 ảnh** vốn dĩ Model 32 đã làm đúng. Điều này cho thấy sự dịch chuyển nhẹ trong ranh giới quyết định (decision boundary) của bộ phân lớp ký tự. Ví dụ, Model 72 có xu hướng nhầm ký tự `R` sang `N` hoặc `K` nhiều hơn.
* **Trường hợp lỗi chung (`map_00326.png` - `KAEAX`)**: Ký tự thứ 3 (`E`) bị đoán sai thành `A` ở Model 32 và `L` ở Model 72. Đây là dấu hiệu của việc ảnh bị nhiễu nền hoặc nét vẽ đè lấp mất thông tin đặc trưng của chữ `E`.

### 2. Khuyến nghị Sử dụng
* **Lựa chọn tối ưu**: **Model 72** (`best-epoch072.ckpt`) là lựa chọn tối ưu hơn để triển khai API nhờ độ chính xác vượt trội hơn (**98.80%**).
* **Chiến lược cải tiến thêm (nếu cần)**:
  * Thu thập thêm các mẫu chứa ký tự dễ nhầm lẫn như: `R`/`N`/`K`, `T`/`7`, `P`/`L`.
  * Áp dụng kỹ thuật tăng cường dữ liệu (Data Augmentation) tập trung vào việc tạo độ nghiêng và nhiễu nét ký tự để mô hình tăng tính bền bỉ (robustness).
