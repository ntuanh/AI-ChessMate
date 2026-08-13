# 📋 TRẠNG THÁI DỰ ÁN — sáng dậy đọc file này

Cập nhật: rạng sáng 2026-08-12 (làm autonomous trong đêm).

## ✅ ĐÃ XONG & ĐÃ TEST TRÊN AIBOX
Toàn bộ **"bộ não"** chạy tốt, đã chạy thật trên AIBOX 8550:

1. **Engine cờ** — Stockfish 14.1 (`/usr/games/stockfish`), có chỉnh độ khó / Elo.
2. **Phân tích chuyên sâu** (`analysis.py`) — vật chất, an toàn Vua, cấu trúc Tốt
   (chồng/cô lập/thông), kiểm soát trung tâm, phát triển quân, Xe cột mở, cặp Tượng,
   **phát hiện quân bị treo** (SEE), phương án chính (PV) dịch sang ký hiệu.
3. **Bình luận từng nước** (`commentary.py`) — mô tả nước đi tiếng Việt, chấm chất
   lượng (chính xác / thiếu chính xác ?! / sai lầm ? / rất tệ ??), đánh giá thế trận.
4. **Đối thoại hỏi–đáp** (`assistant.py`) — trả lời: "đánh giá thế cờ", "nên đi gì",
   "có gì nguy hiểm", "vật chất thế nào", "kế hoạch", "tại sao"…
5. **Đọc to giọng NEURAL tự nhiên** (`speaker.py`) — mặc định dùng **Piper**
   (giọng thật, dễ nghe), tự lùi về espeak nếu thiếu. So sánh giọng ở
   `captures/sosanh/` (1_PIPER_moi vs 2_espeak_cu).
6. **Chơi tương tác** (`main.py`) + **tự test mô phỏng** (`simulate.py`) — đã chạy OK.

### Cách chạy nhanh (trên AIBOX)
```bash
cd /data/kit
python3 -m chess_ai.simulate --plies 40          # xem AI đánh + bình luận
python3 -m chess_ai.main --color white --skill 12 # chơi thật
```

## 👁️ THỊ GIÁC — CAMERA DI ĐỘNG soi màn hình (nâng cấp lớn 2026-08-12 chiều)
Pipeline MỚI trong `gridfind.py` (+ `watch.py` viết lại): **không cần camera cố định**.
Bám bàn bằng ORB mỗi frame → mất dấu thì tự dò lại → mọi kết quả TỰ CHẤM ĐIỂM caro
(lưới lệch = vứt frame, không bao giờ xuất kết quả sai). Tận dụng tri thức cờ vua:
- Lưới caro 8x8 = chữ ký tự kiểm + tín hiệu để refine 4 góc (leo đồi + quét pha + snap ô).
- Thế khai cuộc đã biết → TỰ HỌC ngưỡng occupancy (64 mẫu gán nhãn miễn phí mỗi phiên).
- Màu quân → tự xác định hướng bàn; giữa ván dò lại thì khớp occupancy với game state.
- Phong cấp: mặc định Hậu (occupancy không phân biệt được quân phong).

**Kết quả test (chạy trên AIBOX):**
- Ảnh THẬT `python3 -m chess_ai.gridfind_test real`: dò đúng lưới 5/5 ảnh dò được
  (kể cả phòng tối r720, ảnh nghiêng); 3 ảnh fail đều chính đáng (mờ tịt/bàn cụt/frame đen).
- Camera RUNG mô phỏng `gridfind_test synth`: 40/40 nước, 0 lần mất bàn, khớp khai cuộc 64/64.
- `vision_test` cũ vẫn 40/40 cả 3 chế độ. Smoke test frame camera thật: caro=8.2, lưới khớp.

**Dùng:** màn hình để THẾ KHAI CUỘC → `python3 -m chess_ai.watch calib` (tự học hết,
xem `captures/watch_warped.png` kiểm tra mắt) → `python3 -m chess_ai.watch play`.
LƯU Ý: camera chỉ 1 tiến trình giữ được — đang bị `/data/coach_server.py` (port 8090) giữ;
muốn chạy watch thì dừng coach_server (hoặc ngược lại).

**Còn lại (tuỳ chọn):**
- [ ] Test `watch play` nguyên ván với người thật (cần camera rảnh).
- [ ] Nếu occupancy chập chờn app lạ: thay bằng CNN nhị phân tí hon (data tự gán nhãn từ game state).

## 🎤 NGHE BẰNG GIỌNG NÓI (đã dựng, chờ test giọng thật)
- [x] Cài **Vosk** + model tiếng Việt (`models_asr/vosk-model-small-vn-0.4`).
- [x] Module `listener.py` (mic `arecord` → chữ) + gắn vào `main.py` qua cờ `--listen`.
- [ ] **Test với giọng NGƯỜI THẬT** (đêm qua chỉ test được bằng giọng máy espeak nên
      nhận sai — vd "đánh giá thế cờ" → "vị sơ chế"). Sáng bạn nói vào mic để kiểm tra:
      ```bash
      python3 -m chess_ai.main --color white --skill 12 --listen
      ```
- [ ] Nếu nhận kém: chỉnh thời gian thu (`--listen-seconds`), hoặc thêm từ điển giới hạn.

## 🔜 TÙY CHỌN NÂNG CAO
- [ ] Bình luận mượt hơn bằng **LLM nhỏ chạy trên NPU**.
- [ ] Giọng đọc tự nhiên hơn (Piper thay espeak-ng).

## ⚙️ HẠ TẦNG ĐÃ DỰNG SẴN
- Internet-qua-USB: proxy trên laptop + `adb reverse` (đã cài stockfish/espeak/chess).
- Đã dọn 43GB rác coredump + chặn tái phát (xem `note.txt`).
- Đã tạo user `thunopro` (khoá không đọc source user khác).
- Code đồng bộ ở AIBOX: `/data/kit/chess_ai`. Đẩy lại bằng `bash tools/deploy.sh`.

## ⚠️ ĐIỂM CÒN THÔ (sẽ tinh chỉnh)
- Câu trả lời "có gì nguy hiểm" đôi khi báo "có nước chiếu" hơi chung chung.
- Giọng espeak-ng tiếng Việt nghe máy móc — có thể đổi sang Piper (giọng thật hơn) sau.
