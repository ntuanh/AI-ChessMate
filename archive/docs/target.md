# 🎯 TARGET — AI Chuyên Gia Cờ Vua Qua Camera (chạy trên AIBOX 8550)

> Một "**đại kiện tướng ảo**" vừa là **ĐỐI THỦ** của bạn, vừa là **CHUYÊN GIA/HLV phân tích**.
> **Input: camera soi vào MÀN HÌNH ĐIỆN THOẠI** đang chạy app cờ (bàn cờ 2D). Sau mỗi
> nước đi của đối thủ, nó **đánh giá thế trận hiện tại**, **nói ra ý tưởng/nhận định**
> như một huấn luyện viên (giọng neural tự nhiên), rồi **đọc to nước đi tiếp theo**.

---

## 1. Mục tiêu
Xây dựng một "kỳ thủ kiêm bình luận viên AI" chạy trên **AIBOX 8550 (QCS8550)** có thể:
- Nhìn **bàn cờ vua thật** qua camera, hiểu vị trí các quân.
- **Chơi như một đối thủ** ở 2 chế độ: người dùng chọn cầm **Trắng** hoặc **Đen**.
- Sau mỗi nước đi → **phân tích thế trận như chuyên gia**: ai đang lợi, tại sao, kế
  hoạch nên làm gì, cạm bẫy cần tránh… và **nói ra bằng giọng nói**.
- **Đọc to nước đi** của chính nó để người chơi thực hiện trên bàn thật.

---

## 2. Vai trò kép: Đối thủ + Chuyên gia

### 2.1. Là ĐỐI THỦ
- Chọn màu quân (Trắng đi trước / Đen đi sau).
- Tự tính và **đi nước của mình** (đọc to cho người chơi đặt hộ quân trên bàn thật).
- Có thể chỉnh **độ khó** (giới hạn độ sâu / Elo của engine).

### 2.2. Là CHUYÊN GIA PHÂN TÍCH (điểm nhấn chính)
Sau **mỗi** nước đi (của đối thủ hoặc của mình), AI phát biểu như một HLV:
- **Đánh giá thế trận**: "Trắng lợi +0.8 — kiểm soát trung tâm tốt."
- **Nhận định ý tưởng**: "Đối thủ vừa mở cột c, đang nhắm tấn công cánh hậu."
- **Gợi ý kế hoạch**: "Nên nhập thành sớm để an toàn Vua."
- **Cảnh báo cạm bẫy**: "Cẩn thận nước Mã e5 ăn đôi."
- **Khen/chê nước đi**: "Nước đi hay!" / "Nước này hơi yếu, để lộ Tốt d."

> Mức độ bình luận có thể chọn: *ngắn gọn* (chỉ điểm số + 1 câu) hoặc *chi tiết*
> (giải thích như dạy học).

---

## 3. Luồng hoạt động một nước
```
   Đối thủ di chuyển quân trên bàn thật
                │  (nói "tôi đi rồi" hoặc bấm phím)
                ▼
   [1] Camera chụp bàn cờ (Logitech C270 - /dev/video2)
                │
                ▼
   [2] THỊ GIÁC: dò bàn 8x8 + nhận diện quân -> so thế cờ trước
       -> suy ra NƯỚC ĐI của đối thủ (cập nhật FEN)
                │
                ▼
   [3] ENGINE (Stockfish): tính điểm đánh giá + nước tốt nhất + các biến chính
                │
                ▼
   [4] BỘ BÌNH LUẬN: biến số liệu engine -> lời nhận định tự nhiên
       ("Trắng lợi nhẹ, nên phát triển quân nhẹ, coi chừng nước Mã e5...")
                │
                ▼
   [5] ÂM THANH (loa/TTS): đọc to NHẬN ĐỊNH + NƯỚC ĐI của AI
                │
                ▼
   Người chơi thực hiện -> lặp lại
```

---

## 4. Kiến trúc kỹ thuật

### 4.1. Phần cứng (đã có trên AIBOX ✅)
- **Camera**: Logitech C270 HD (`/dev/video2`) — **soi vào màn hình điện thoại** chạy app cờ.
- **Âm thanh**: card âm thanh (mic + loa) đã nhận.
- **Tính toán**: CPU Kryo 8 lõi + NPU/DSP Hexagon (chạy model thị giác qua QNN/ONNX).

### 4.2. Khối phần mềm
| Khối | Nhiệm vụ | Công nghệ | Tình trạng trên AIBOX |
|------|----------|-----------|------------------------|
| **Vision – đọc bàn 2D trên màn hình** | Dò khung → nắn phẳng → occupancy 8×8 → suy nước đi | OpenCV | ✅ đã test 100% trên bàn render giả lập màn hình ĐT |
| **Board state** | Quản thế cờ, so 2 khung → ra nước đi | `python-chess` | ⏳ cần `pip install chess` |
| **Engine** | Điểm đánh giá + nước tốt nhất + biến chính | **Stockfish** (aarch64) | ❌ chưa cài |
| **Bộ bình luận** ⭐ | Số liệu engine → lời nhận định như chuyên gia | Template luật + (tùy chọn) LLM nhỏ local | ⏳ tự viết |
| **TTS (đọc to)** | Chữ → giọng nói | **Piper** (neural) / espeak dự phòng | ✅ đã cài, giọng tự nhiên |
| **ASR (mic → lệnh)** | Nhận lệnh giọng nói (tùy chọn) | Vosk / Whisper.cpp | ⏳ tùy chọn |
| **Điều phối** | Nối các khối, quản vòng chơi | Python | ⏳ tự viết |

### 4.3. Bộ bình luận chuyên gia — làm sao "nghe giống HLV"?
- **Cách 1 (offline, chắc chắn):** dựa vào số liệu Stockfish (điểm eval, nước tốt
  nhất, quân bị treo, kiểm soát trung tâm, an toàn Vua...) → ghép thành câu theo
  **mẫu (template)** tiếng Việt. Không cần internet.
- **Cách 2 (tự nhiên hơn):** thêm một **mô hình ngôn ngữ nhỏ chạy local** (trên
  NPU) để diễn đạt mượt hơn. Nặng hơn, làm sau.

---

## 5. Chạy trên AIBOX — có cần internet không?
- **Để CHẠY: KHÔNG cần internet.** Code đẩy sang bằng `adb push` qua cáp USB, chạy
  thẳng trên AIBOX. OpenCV + onnxruntime đã có sẵn.
- **Chỉ cần internet 1 lần** để **cài** phần còn thiếu: `python-chess`, `stockfish`,
  `espeak-ng`. Sau đó chơi **hoàn toàn offline**.
- Chia mạng cho AIBOX **không cần dây LAN**: dùng **`adb reverse` + proxy trên laptop**
  qua chính cáp USB (đã kiểm tra: chạy được).

---

## 6. Các giai đoạn triển khai
- [ ] **GĐ0** – Camera + loa + mic hoạt động; cài `python-chess`, `stockfish`, `espeak-ng`.
- [ ] **GĐ1** – Thị giác: dò khung bàn cờ 8x8 từ ảnh camera.
- [ ] **GĐ2** – Nhận diện 12 loại quân + ô trống → dựng FEN.
- [ ] **GĐ3** – So 2 thế cờ liên tiếp → suy ra nước đi đối thủ.
- [ ] **GĐ4** – Tích hợp Stockfish: điểm đánh giá + nước tốt nhất + biến chính.
- [ ] **GĐ5** – Bộ bình luận chuyên gia (template) + TTS đọc to.
- [ ] **GĐ6** – Vai trò đối thủ (AI tự đi) + 2 chế độ Trắng/Đen + độ khó.
- [ ] **GĐ7** – (tùy chọn) ASR nhận lệnh giọng nói; bình luận bằng LLM local.

---

## 7. Câu hỏi cần làm rõ
1. Bình luận muốn kiểu **ngắn gọn** hay **chi tiết như dạy học**? Tiếng Việt hay Anh?
2. AI chỉ **đọc gợi ý** để người tự đặt quân, hay sau này gắn **tay máy** tự đi?
3. Bàn cờ vật lý: loại quân / màu nền cụ thể? (ảnh hưởng cách nhận diện thị giác)
4. Độ khó đối thủ: cố định hay chỉnh được (giới hạn Elo)?

---
*Ghi chú: thông tin thiết bị & cách kết nối AIBOX xem file `note.txt`.*
*Khả thi kỹ thuật: OpenCV + onnxruntime đã có sẵn; Stockfish + espeak-ng cài 1 lần là đủ chạy offline.*
