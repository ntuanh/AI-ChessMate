# archive/ — các hướng đã thử trước đó

Thư mục này **không nằm trong đường chạy chính**. Giữ lại để tham khảo và để thấy
dự án đã đi qua những hướng nào trước khi chốt kiến trúc hiện tại.

Đường chạy thật chỉ gồm `chess_ai/` + `tools/coach_server.py` + `tools/npu_server.py`.
Xem [SOLUTION.md](../SOLUTION.md).

## Mã Python

| File | Từng dùng để làm gì | Vì sao không dùng nữa |
|---|---|---|
| `board_vision.py`, `recognizer.py` | các bản nhận diện bàn cờ đời đầu | thay bằng `rectify.py` + `gridfind.py`, có tự chấm điểm |
| `watch.py`, `vision_test.py`, `gridfind_test.py`, `simulate.py` | công cụ soi và thử nghiệm rời | thay bằng `tests/` và `tools/replay_server.py` |
| `coach.py`, `assistant.py`, `listener.py`, `main.py` | đường chạy dòng lệnh có hỏi–đáp bằng giọng nói | thay bằng máy chủ web `coach_server.py` |
| `llm_commentary.py` | bình luận bằng mô hình ngôn ngữ | chưa đấu dây vào đường chạy chính (xem mục "Hướng đi tiếp") |

## Script

`scripts/` chứa các tiện ích chạy một lần hoặc đã hết vai trò: cầu nối SSH, cài gói
`.deb`, thu thập ô cờ, đọc khung hình lẻ, huấn luyện ngưỡng đời đầu.

## Tài liệu

`docs/` chứa nhật ký làm việc và các bản kế hoạch cũ. Nội dung còn giá trị đã được
gộp vào `SOLUTION.md`.

## Lưu ý

Mã ở đây đã đổi `from . import x` thành `from chess_ai import x` khi chuyển thư mục,
nên chạy được nếu đứng ở gốc dự án. Nhưng chúng **không được kiểm thử** và có thể
đã lạc hậu so với API hiện tại.
