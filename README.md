# captcha-map-solver

End-to-end PyTorch / Lightning solver cho bộ CAPTCHA `map_*.png` (500 ảnh, 128×128 RGBA, nhãn 5 ký tự, alphabet 24 ký hiệu). Kiến trúc theo phong cách [PyCAPTCHA](https://github.com/ZiYang-xie/PyCAPTCHA): backbone CNN → 5 đầu softmax × 24 lớp.

> Phân tích chi tiết bộ dữ liệu và lý do chọn kiến trúc này nằm trong [`research.md`](./research.md).

## Cấu trúc

```
TrainAI/
├── data/                 # 500 ảnh + metadata.csv (đã có sẵn)
├── configs/
│   ├── default.yaml      # config train mặc định
│   └── splits.json       # sinh tự động lần đầu chạy train.py
├── src/
│   ├── dataset.py        # alphabet, encode/decode, DataModule
│   ├── model.py          # ResnetMultiHead / ConvMultiHead + LightningModule
│   └── config.py
├── train.py              # entrypoint train + test
├── evaluate.py           # đánh giá + confusion matrix mức ký tự
├── predict.py            # inference 1 ảnh
├── analyze_images.py     # phân tích định lượng dataset
├── research.md           # tài liệu nghiên cứu
└── requirements.txt
```

## Cấu hình máy train mục tiêu

- GPU: NVIDIA RTX 3060 8 GB VRAM (CUDA 12.8+)
- CPU: Intel i5-12400F
- RAM: 16 GB

Thiết lập mặc định trong `configs/default.yaml` đã được tối ưu cho cấu hình này (batch 32, mixed precision `16-mixed`, num_workers 4).

## Cài đặt trên máy train

### 1. Clone repo và tạo môi trường ảo

```powershell
git clone https://github.com/cvx1806/captcha-map-solver.git
cd captcha-map-solver
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Cài PyTorch khớp với CUDA 12.8

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> Nếu bạn dùng CUDA 12.4 thay `cu128` bằng `cu124`, hoặc xem [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) để lấy lệnh phù hợp với phiên bản CUDA của driver.

### 3. Cài các gói còn lại

```powershell
pip install -r requirements.txt
```

### 4. Kiểm tra GPU

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Output mong đợi: `True NVIDIA GeForce RTX 3060`.

## Train

```powershell
python train.py --config configs/default.yaml
```

Lần đầu sẽ tự sinh `configs/splits.json` (400/50/50 deterministic theo seed 42). Logs đi vào `tb_logs/`, checkpoint vào `checkpoints/baseline-resnet18/`.

Theo dõi:

```powershell
tensorboard --logdir tb_logs
```

Mục tiêu: **val sequence accuracy ≥ 0.90** sau 30–50 epoch trên RTX 3060 (~1–2 giờ).

### Đổi backbone

Trong `configs/default.yaml`:

```yaml
solver:
  backbone: resnet18   # chọn: resnet18 | resnet34 | conv
  pretrained: true
```

`conv` là phiên bản nhẹ nhất (PyCAPTCHA `model_conv` đã rescale cho input 128×128).

### Đổi precision nếu OOM

```yaml
solver:
  precision: 16-mixed   # hoặc bf16-mixed cho Ampere/Ada
  batch_size: 16        # giảm khi VRAM căng
```

## Đánh giá

```powershell
python evaluate.py --ckpt checkpoints/baseline-resnet18/last.ckpt --split test
```

In ra sequence accuracy, character accuracy, top-15 cặp ký tự nhầm lẫn, và 20 ví dụ sai đầu tiên.

## Inference 1 ảnh

```powershell
python predict.py --ckpt checkpoints/baseline-resnet18/last.ckpt --image data/map_00000.png
```

## Phân tích dataset

```powershell
python analyze_images.py
```

In ra: bbox text, hash trùng pixel, foreground HSV, edge density.

## Lộ trình nâng cấp (theo `research.md`)

1. **Tier 1 — Multi-head CNN**: `python train.py --config configs/default.yaml`
   - Backbone resnet18 / conv. Train nhanh nhưng dễ overfit trên 400 mẫu.
2. **Tier 2 — CRNN + CTC**: `python train.py --config configs/crnn.yaml`
   - Học theo trục thời gian, ít memorize hơn.
3. **Tier 3 — Fine-tune TrOCR (KHUYẾN NGHỊ)**: `python train.py --config configs/trocr.yaml`
   - Pretrained encoder mang sẵn shape prior, fit tốt với 400 mẫu.
   - Thêm `pip install transformers sentencepiece` (đã có trong `requirements.txt`).
   - Predict: `python predict.py --ckpt checkpoints/trocr-small/last.ckpt --image data/map_00000.png --task trocr --config configs/trocr.yaml`

## License

MIT — kiến trúc tham khảo từ [ZiYang-xie/PyCAPTCHA](https://github.com/ZiYang-xie/PyCAPTCHA) (MIT).
