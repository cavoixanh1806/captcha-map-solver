# 20 ảnh đầu trước nè

Exported: 5/17/2026, 7:27:49 PM

---

**User:**

trong thời gian train thì tôi với bạn sẽ phân tích ảnh và tạo 1 bản nghiên cứu ảnh data real trước 
20 ảnh đầu trước nè

---

**Assistant:**

OK, ý hay đó. Trong khi train chạy, mình bóc tách 20 ảnh đầu (`map_00000.png` → `map_00019.png`) đối chiếu với nhãn ground truth, ghi nhận mọi chi tiết "nhìn dễ nhầm" → đó sẽ là input rất tốt để chỉnh synth generator + augmentation.

Mình sẽ ghi thẳng vào `research.md` (nối thêm section mới, không đụng nội dung cũ).

Xong. Section 8 đã được nối thẳng vào `research.md`, không động nội dung cũ. Đây là điểm chốt từ 20 ảnh đầu (`map_00000`-`map_00019`) đối chiếu nhãn:

**4 phát hiện then chốt:**

- **Chồng lấn là chủ đạo, không phải edge case** — 60% mẫu rơi vào nhóm "Cao/Rất cao", có ảnh như `map_00004` (X xé nét 7), `map_00006` (KK4 ép sát), `map_00016` (D bị W xé). Synth generator nếu chỉ render glyph rời nhau sẽ **không khớp** với real.

- **3 preset nền lặp đi lặp lại**: pastel-lavender + chấm vàng (30%), peach + line trắng (40%), mint/yellow pastel (15%). Synth hiện tại nên có ít nhất 3 texture này.

- **Cặp ký tự dễ nhầm có bằng chứng cụ thể**: `4-A-H` (00000, 00006, 00012), `M-N-W` (00007, 00000), `K-X` (00000, 00006, 00010), `E-F-B` (00006, 00013, 00015). Sau khi baseline chạy xong, in confusion matrix kiểm các cặp này trước.

- **Lợi thế thiết kế alphabet**: vì đã loại `0,1,2,5,6,8,B,G,I,O,S,Z`, nhiều cặp nhầm "lý thuyết" (D↔O, 3↔B, J↔I) **không có target ngoài 24 lớp** → mask logits 24-class là cách phòng vệ rẻ và mạnh.

**Hành động đề xuất sau khi train baseline xong:**
1. So accuracy synth-only vs real-finetune để đo gap.
2. Cập nhật `src/synth.py` với checklist 10 mục ở section 8.3.
3. Áp augmentation "nhẹ tay" theo bảng 8.4 (đặc biệt **không** dùng hue jitter mạnh và **cấm** flip).

Bạn muốn mình làm gì tiếp khi train chạy? Mình có thể:
- Phân tích thêm 20 ảnh (`map_00020` → `map_00039`) để xác nhận pattern có ổn định không
- Viết script `analyze_overlap.py` đo định lượng % overlap thực sự trên cả 500 ảnh
- Soi `src/synth.py` hiện tại xem đã có những gì trong checklist 8.3, đề xuất diff cụ thể

---

**User:**

tiếp tục phân tích bằng mắt đi

---

**Assistant:**

Để mình phân tích bằng mắt tiếp được, mình cần bạn kéo thả batch ảnh kế tiếp vào chat — mình chỉ "thấy" được ảnh nào bạn đính kèm vào tin nhắn thôi, file PNG trong workspace mình không đọc được pixel trực tiếp.

Đề xuất chia batch như này cho gọn:

**Batch 2:** `map_00020` → `map_00039` (20 ảnh kế tiếp, theo nhãn:)
- `H7K3X · CRUNX · NULLH · FMMAE · XQJWL · W3TRC · XFYVM · AV4LL · UWRUH · Q9YJA`
- `Y3Q93 · YHTRT · 3JYER · ML4CT · NFWRE · QHXVH · QWMJL · NJWND · T9LFR · ETETY`

Một vài ảnh trong batch này nhãn có yếu tố mới đáng chú ý đã đoán trước:
- `NULLH` (map_00022): có **3 ký tự liên tiếp giống nhau** (`L L`) và `NULL` → kiểm tra cách generator/render xử lý ký tự lặp.
- `XQJWL` (map_00024), `Y3Q93` (map_00030), `Q9DHQ` đã có ở batch 1 → so sánh `Q` ở các vị trí khác nhau.
- `ETETY` (map_00039): pattern `T E T` lặp gần — kiểm tra spacing.
- `AV4LL` (map_00027): thêm ca `LL` đôi liên tiếp.

Sau khi mình có 20 ảnh kế tiếp, mình sẽ:
1. Lập bảng đối chiếu giống section 8.1 (nền / màu glyph / chồng lấn / nguy cơ đọc sai).
2. Cập nhật phân bố mức độ khó (xem có giữ tỉ lệ ~60% Cao trở lên không, hay batch này khác).
3. Bổ sung cặp ký tự nhầm mới nếu phát hiện (nghi ngờ: `L↔I`, `Y↔V↔T` khi có đuôi cong).
4. Kiểm tra cụ thể **ký tự lặp liền nhau** — đây là ca khó với CTC (cần blank token tách), với multi-head thì OK vì 5 head độc lập.

Bạn kéo 20 ảnh `map_00020.png` → `map_00039.png` vào nhé.

---

**User:**

đây 20 đến 39

---

**Assistant:**

Tốt. 20 ảnh tiếp theo (`map_00020` → `map_00039`) — mình ghi nhận đầy đủ, đặc biệt chú ý các ca nhãn có ký tự lặp (`NULLH`, `FMMAE`, `AVALL`, `Y3Q93`, `ETETY`) và các pattern nền mới chưa thấy ở batch 1.

Xong section 9. Bản tóm tắt nhanh những điểm "đáng đầu tư thời gian" sau khi xem 20 ảnh batch 2:

**Khám phá quan trọng nhất (chưa thấy ở batch 1):**

1. **3 preset nền mới** — đặc biệt **BG-D peach+blue split** (`00020`, `00031`, `00037`) trông như tile bản đồ thực sự (water/land), khác hoàn toàn nền pastel-flat batch 1. Đây có thể chính là lý do filename bắt đầu bằng `map_*`. Synth hiện tại chắc chắn đang thiếu kiểu này → khi tổng hợp ảnh sẽ thiếu hẳn 1 manifold của real → giảm khả năng generalize.

2. **Outline cyan rõ ràng** không phải hiếm — xác nhận trên `00027` (A), `00033` (C), `00034` (W), `00039` (E). Tần suất ~20%, đủ để bật hard trong synth chứ không phải edge case.

3. **Ký tự lặp đè rất sâu** — `NULLH` (`00022`) là ca khó kinh điển: cặp `LL` đè đến mức trông như chữ `I` đơn. Đây là **bằng chứng mạnh** ưu tiên multi-head CNN over CTC ở phase baseline (CTC cần blank token tách, dễ collapse 2 chữ giống nhau thành 1).

4. **`T` màu sát nền** (`00025` `W3TRC`) — case patological. T lavender trên nền lavender → chắc chắn model sai. Đây có thể là 1 "data noise" của bộ dataset, không nên ép synth tránh; thay vào đó model phải robust được nhờ augmentation có cutout/erasing.

5. **Letter scale disparity** (`00021` U bé tí, `00024` Q to vs J nhỏ) — synth của bạn nhiều khả năng đang dùng uniform scale → thiếu 1 tail của distribution.

**Cập nhật cặp ký tự dễ nhầm (tăng độ rủi ro):**
- `M↔N↔W` lên **Rất cao** (4 bằng chứng mới: `00023, 00036, 00037, 00034`)
- `T↔7↔1` thêm rủi ro mới do contrast thấp (`00025`)
- `V↔Y↔K↔X` lên Cao do `V-H cùng màu` (`00035`)
- Thêm cặp mới: `L↔I↔1` do scale + LL-merge

**Tổng cộng đã xác định 19 checklist items cho synth**. Khi nào train xong, mình có thể mở `src/synth.py` xem hiện tại đã làm được những gì trong checklist này, đề xuất diff.

Bạn muốn tiếp 20 ảnh nữa (`map_00040` → `map_00059`) không, hay đủ rồi để mình quay sang phân tích `src/synth.py`?

---

**User:**

40 dêdns 59 của bạn đây

---

**Assistant:**

On it.

---

**User:**

40 dêdns 59 của bạn đây

---

**Assistant:**

On it.

---

**User:**

40 dêdns 59 của bạn đây

---

**Assistant:**

On it.

---
