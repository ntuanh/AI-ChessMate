# ♟️ Chuyên gia Cờ vua AI qua Camera — AIBOX 8550

Một "đại kiện tướng ảo" chạy trên AIBOX 8550 (Qualcomm QCS8550): **vừa là đối thủ,
vừa là huấn luyện viên**. Nhìn bàn cờ thật qua camera, phân tích thế trận như chuyên
gia, nói ra nhận định/lời khuyên bằng giọng nói, và đối thoại hỏi–đáp với người chơi.

Xem mục tiêu chi tiết trong [target.md](target.md), trạng thái trong [STATUS.md](STATUS.md).

## Kiến trúc phần mềm (`chess_ai/`)
| File | Vai trò |
|------|---------|
| `engine.py` | Bọc **Stockfish** (đi nước + phân tích điểm) |
| `analysis.py` | **Phân tích thế trận chuyên sâu** — trái tim: vật chất, an toàn Vua, cấu trúc Tốt, trung tâm, cột mở, quân treo… |
| `commentary.py` | Bình luận từng nước đi (mô tả + chất lượng + đánh giá + ý tưởng) |
| `assistant.py` | **Đối thoại** hỏi–đáp tiếng Việt về thế cờ |
| `speaker.py` | **Đọc to** (espeak-ng, 7 giọng nam/nữ/thì thầm) |
| `coach.py` | Bộ điều phối tổng (`Coach`) |
| `vision.py` | **Thị giác** camera → nước đi (occupancy). *Cần hiệu chỉnh bàn thật.* |
| `main.py` | Chơi tương tác qua bàn phím/giọng nói |
| `simulate.py` | Tự test bằng mô phỏng (không cần bàn thật) |

## Cách chạy (trên AIBOX)
```bash
cd /data/kit

# 1) Tự test mô phỏng (AI đánh một ván + bình luận + tạo giọng nói)
python3 -m chess_ai.simulate --plies 40

# 2) Chơi tương tác — bạn cầm Trắng, AI cầm Đen, độ khó 12
python3 -m chess_ai.main --color white --skill 12 --voice nam_chuan

#   Trong lúc chơi, gõ:
#     e4 / Nf3 / O-O        -> đi nước
#     đánh giá thế cờ        -> hỏi trợ lý
#     nên đi gì / có gì nguy hiểm không / vật chất thế nào
#     phân tích              -> phân tích chi tiết
#     giọng nu_chuan         -> đổi giọng đọc
#     thoát
```

## Điều khiển từ laptop (qua USB/adb)
```bash
# đẩy code mới sang AIBOX
bash tools/deploy.sh
# chạy mô phỏng từ laptop
ADB=".../platform-tools/adb"; $ADB shell "cd /data/kit && python3 -m chess_ai.simulate --plies 30"
```

## Yêu cầu (đã cài sẵn trên AIBOX)
- Python 3.10, `python-chess`, OpenCV 4.13, numpy
- `stockfish` (/usr/games/stockfish), `espeak-ng`, `aplay` (alsa-utils)

## Giọng nói mẫu
Nghe thử tại [captures/voices/](captures/voices/) (7 giọng).
