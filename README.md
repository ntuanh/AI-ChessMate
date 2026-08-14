# KIT — Huấn luyện viên cờ vua nhìn qua camera, chạy trọn trên AIBOX 8550

Camera Logitech C270 cắm vào AIBOX, soi vào bàn cờ Lichess trên màn hình laptop.
Hệ thống dựng lại thế cờ, bám từng nước đi, và gợi ý nước tốt nhất bằng Stockfish.

**Toàn bộ thị giác, mô hình học sâu và engine chạy trên AIBOX.** Laptop chỉ là cửa
sổ xem — rút dây ra, hệ thống vẫn chạy.

> 📄 **[SOLUTION.md](SOLUTION.md) — đọc file này trước.** Toàn bộ cách làm, chuỗi
> Edge AI từ dữ liệu đến NPU, các con số đo được, và cả những hướng đã thất bại.

## Con số chính

| | |
|---|---|
| CNN đọc quân trên **NPU Hexagon HTP** | **9,8 ms** (CPU: 479 ms — nhanh hơn 49 lần) |
| Độ chính xác INT8 so với fp32 | **trùng 100%**, cùng đạt 99,98% |
| Một vòng nhận diện | 64–74 ms |
| Độ trễ bắt một nước đi | **~0,4 giây** |
| Nhận diện đúng cả bàn (1000 khung thật) | 92,3% |

## Chạy

```bash
bash tools/install_aibox.sh     # LẦN ĐẦU: cài dịch vụ, tự bật khi khởi động máy
bash tools/run_aibox.sh         # hàng ngày: đẩy code mới + khởi động lại
```

Rồi mở `http://127.0.0.1:8090`.

## Kiểm chứng — không cần camera

```bash
python3 tests/test_reader.py              # 31 phép thử logic bám nước đi
python3 tools/replay_server.py --audit    # chấm độ chính xác trên dữ liệu thật
python3 tools/replay_server.py --moves    # chấm khả năng bắt đúng nước
```

## Cấu trúc

```
chess_ai/     thư viện lõi (thị giác, mô hình, bám nước đi, engine)
tools/        máy chủ web, sidecar NPU, script dựng và triển khai
tests/        bộ thử tự động + ảnh mẫu
docs/         hướng dẫn phụ
archive/      các hướng đã thử trước đó, giữ để tham khảo
```

Chi tiết từng file: xem mục 9 của [SOLUTION.md](SOLUTION.md).

## Yêu cầu

- **AIBOX**: Stockfish (`/usr/games/stockfish`), OpenCV, `python-chess`, QAIRT SDK
  cho NPU. User chạy dịch vụ phải thuộc nhóm `video` và `system`.
- **Máy tính thường** (chỉ để chạy test và phát lại): xem `requirements.txt`.
