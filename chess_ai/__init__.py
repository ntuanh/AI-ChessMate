"""chess_ai — Huấn luyện viên cờ vua qua camera, chạy trên AIBOX 8550.

Toàn cảnh cách hoạt động: xem SOLUTION.md ở gốc dự án.

Thị giác — dựng lại thế cờ từ khung hình camera:
  rectify     dò khung bàn, nắn thẳng, bám 4 góc real-time
  gridfind    căn lưới 8×8 trên ảnh đã nắn, ngưỡng ô có quân/trống
  piece_net   CNN đọc quân — ONNX trên CPU, QNN trên NPU Hexagon
  tracker3    BÁM NƯỚC ĐI: đọc 64 ô → so với thế cờ đang giữ → lọc bằng luật cờ
  reader      các hàm suy luận thế cờ, tách rời camera nên test được độc lập
  vision      tiện ích camera

Cờ vua — biến thế cờ thành lời khuyên:
  engine      bọc Stockfish (đi nước + phân tích)
  analysis    phân tích thế trận
  commentary  sinh lời bình cho từng nước đi
  render      vẽ bàn cờ đối chiếu để soi bằng mắt
  speaker     đọc to
  llm         nối mô hình ngôn ngữ (chưa dùng trong đường chạy chính)

Các hướng đã thử trước đó nằm ở archive/.
"""
__version__ = "1.0.0"
