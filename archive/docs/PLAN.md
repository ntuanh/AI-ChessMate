# 🗺️ KIT — Mục tiêu, hiện trạng & chiến thuật đi tiếp

*Cập nhật: 12/08/2026 16:30. Mọi dòng "✅ đã kiểm chứng" đều được chạy thật trên AIBOX trong phiên này.*

---

## 1. Mục tiêu

Một **"đại kiện tướng ảo"** chạy trên AIBOX 8550, vừa là **đối thủ** vừa là **huấn luyện viên**:

> Camera soi vào **màn hình điện thoại** đang chạy app cờ → đọc thế cờ → Stockfish phân tích
> → bình luận như HLV bằng tiếng Việt → **nói ra loa** → người chơi đi tiếp.

Chạy **hoàn toàn offline**. Chi tiết: [target.md](target.md).

---

## 2. Đã xong

### 2.1. Bộ não — ✅ chạy thật, vừa kiểm chứng lại hôm nay

Chạy `python3 -m chess_ai.simulate --plies 8` trên AIBOX, kết quả thật:

```
[5. AI] Tôi đi: Tốt ăn d4 (cxd4). Thế trận: Đen hơi lợi (+0.7). Ý tưởng mạnh: cxd5.
💬 "nên đi nước nào" -> AI: Theo tôi, bạn nên cân nhắc Nxd4.
Cảnh báo: Trắng đang treo Tốt ở h4 (thiệt ~1). Cấu trúc Tốt Đen: 1 Tốt chồng.
Phương án gợi ý: Nxd4 dxc4 e3 e5.
--- ĐÃ TẠO 8 FILE GIỌNG NÓI --- (PIPER + 7 giọng espeak)
```

| Khối | File | Trạng thái |
|---|---|---|
| Engine Stockfish 14.1 | `engine.py` | ✅ |
| Phân tích chuyên sâu (vật chất, an toàn Vua, quân treo/SEE, PV) | `analysis.py` | ✅ |
| Bình luận + chấm chất lượng nước đi | `commentary.py` | ✅ |
| Đối thoại hỏi–đáp tiếng Việt | `assistant.py` | ✅ |
| **Tổng hợp giọng nói** (Piper neural + espeak) | `speaker.py` | ✅ **tạo file .wav OK** |
| Chơi tương tác / mô phỏng | `main.py`, `simulate.py` | ✅ |

### 2.2. Thị giác — ✅ code xong, ❌ chưa chạm bàn thật

`vision.py` + `render.py`, tự test đạt **40/40 nước** trên bàn 2D giả lập (kể cả nghiêng phối cảnh).
Cách làm bền: không nhận diện *loại* quân, chỉ cần biết ô nào **có** quân → so trước/sau → khớp nước hợp lệ.

### 2.3. Hạ tầng — ✅ dựng trong phiên này

- **Kiểm toán phần cứng AIBOX** — phát hiện 4 sai lệch so với `note.txt` (mục 3).
- **Kiểm kê ổ đĩa & 7 project** đang có trên máy → biết chỗ nào được đụng, chỗ nào không.
- **Quy tắc cấm xoá user** — ghi vào `note.txt` mục 0 + bộ nhớ dài hạn.
- **Truy cập từ xa cho cả nhóm** qua Tailscale — 4 người ngoài đã SSH vào AIBOX thành công.
- **Tự phục hồi khi rút/cắm cáp USB** — `tools/aibox_bridge.sh` + systemd service, đã test.

---

## 3. 🔴 Chặn cứng: AIBOX KHÔNG CÓ LOA

```
$ aplay -l
**** List of PLAYBACK Hardware Devices ****      ← RỖNG
$ arecord -l
card 0: WEBCAM [C270 HD WEBCAM]                  ← chỉ có mic của webcam
```

`note.txt` ghi "card âm thanh (mic + loa) đã nhận" — **không còn đúng**. `lsusb` không thấy sound card nào.

**Mức nghiêm trọng: cao nhất.** Toàn bộ giá trị của dự án nằm ở chỗ AI **nói ra** nhận định. Piper vẫn tạo file `.wav` bình thường — chỉ là không có đường phát ra.

**Chiến thuật (làm ngay, rẻ nhất trước):**

| # | Cách | Chi phí | Đánh giá |
|---|---|---|---|
| 1 | **Cắm loa/USB sound card vào cổng USB AIBOX** | ~50–100k | ⭐ Dứt điểm, đúng kiến trúc. Cắm là `aplay -l` thấy ngay |
| 2 | Phát tiếng **trên laptop**: `adb pull` file .wav rồi `aplay` ở laptop | 0đ | Chữa cháy để dev tiếp, nhưng phá mất tính offline độc lập |
| 3 | Xuất qua HDMI | 0đ | Cần màn hình có loa cắm vào AIBOX |

→ **Làm cách 1.** Trong lúc chờ mua, dùng cách 2 để không tắc việc phát triển.

---

## 4. Còn lại & chiến thuật

### 🥇 Ưu tiên 1 — Hiệu chỉnh thị giác với camera + điện thoại thật

Đây là **rủi ro kỹ thuật lớn nhất còn lại**. Bàn render giả lập luôn "sạch"; màn hình thật có loá, phản chiếu, tự đổi độ sáng.

Chiến thuật — **giảm rủi ro theo từng nấc, không nhảy thẳng vào tích hợp**:

1. **Chụp trước, tính sau.** Dùng `tools/stream_calib.py` chụp ~20 ảnh màn hình điện thoại ở các thế cờ khác nhau. Có bộ ảnh thật rồi mới tinh chỉnh — tránh vừa sửa code vừa canh camera.
2. **Chốt vật lý trước khi chốt tham số.** Camera vuông góc màn hình, tắt đèn trần chiếu thẳng, khoá độ sáng điện thoại ở mức cố định (tắt tự động sáng). Sai ở bước này thì chỉnh ngưỡng bao nhiêu cũng vô ích.
3. **Dò ngưỡng bằng số, không bằng mắt.** Quét `occupancy_grid(thresh=...)` trên bộ ảnh đã chụp, chọn ngưỡng có biên an toàn rộng nhất — không chọn ngưỡng "vừa đủ đúng".
4. **Xác định hướng bàn** (nhìn từ phía Trắng hay Đen) → chỉnh `vision._sq_index`.
5. **Ghép `BoardWatcher` vào `main.py`** — chỉ làm sau khi 4 bước trên đã ổn định.
6. Chơi nhiều app cờ khác nhau → mỗi app canh 1 lần, lưu thành preset.

> **Cửa thoát hiểm:** nếu occupancy trên màn hình thật không đủ tin cậy, chuyển sang bám **màu ô sáng/tối + độ lệch màu quân** thay vì ngưỡng độ sáng tuyệt đối. Đừng vội nhảy sang model học sâu — tốn NPU mà chưa chắc bền hơn.

### 🥈 Ưu tiên 2 — Nghe bằng giọng người thật

Vosk + model VN đã cài, `listener.py` đã gắn vào `main.py` qua cờ `--listen`.
Đêm qua chỉ test được bằng giọng máy espeak → nhận sai ("đánh giá thế cờ" → "vị sơ chế"). **Kết quả đó không có giá trị tham khảo** — giọng máy khác giọng người quá xa.

Chiến thuật:
1. Test lại bằng **giọng người thật**: `python3 -m chess_ai.main --color white --skill 12 --listen`
2. Nếu nhận kém → **giới hạn từ điển** về đúng ~15 câu lệnh hay dùng. Đây là đòn hiệu quả nhất: thu hẹp không gian tìm kiếm ăn đứt việc tinh chỉnh model.
3. Chỉnh `--listen-seconds` cho khớp nhịp nói.
4. Vẫn giữ bàn phím làm đường dự phòng — giọng nói là tiện ích, không phải đường sống.

### 🥉 Ưu tiên 3 — Dọn nợ hạ tầng

| Việc | Vì sao | Cách |
|---|---|---|
| **Đổi mật khẩu `thunopro`** | 4 người ngoài đã vào được, mật khẩu cũ nằm trong `note.txt` | `adb shell 'passwd thunopro'` |
| **Chặn coredump** | pulseaudio vẫn crash-loop (`NRestarts=5`/12 phút), từng ăn 43GB | Đổi `core_pattern`, hoặc `ulimit -c 0` |
| **Bật linger** | Để bridge sống cả khi chưa đăng nhập | `sudo loginctl enable-linger luongminhthu` |
| **Sửa `STATUS.md`/`target.md`** | Đang ghi sai: "8 lõi", "loa đã nhận" | Cập nhật theo mục 3 dưới |

### Tùy chọn — làm sau, chỉ khi phần lõi đã chạy

- Bình luận mượt hơn bằng **LLM nhỏ trên NPU** (Hexagon + QNN 2.47 đã có sẵn ở `/home/hkt`).
- Cắt bớt câu chung chung kiểu "có nước chiếu" trong phần cảnh báo nguy hiểm.

---

## 5. Thông số AIBOX — bản đã kiểm chứng

Sửa lại 4 chỗ sai trong `note.txt`:

| Mục | Ghi cũ | **Thực tế** |
|---|---|---|
| CPU | "Kryo 8 lõi" | ❌ **6 lõi**: 3×2.0GHz + 2×2.8GHz + 1×3.19GHz |
| Âm thanh | "mic + loa đã nhận" | ❌ **Không có thiết bị phát**, chỉ còn mic webcam |
| adb shell | "đang là root" | ❌ Vào là `uid=2000(adb)`, phải `adb root` trước |
| Coredump | "đã dọn xong" | ⚠️ pulseaudio **vẫn crash-loop**, sẽ đầy lại |

Đúng như ghi: QCS8550 (soc_id 603), Ubuntu 22.04.2 / kernel 5.15.170, RAM 16GB, đĩa 94G còn 45G (53%), camera C270 ở `/dev/video2` (MJPG/YUYV, tối đa 1280×720), NPU/DSP Hexagon sẵn sàng, nhiệt 43–46°C idle.

Phần mềm trên AIBOX: Stockfish 14.1 (`/usr/games/stockfish`), Piper (`/data/piper/piper`) + `vi.onnx`, espeak-ng, Vosk VN, python-chess 1.11.2, OpenCV 4.13.0, numpy 2.2.6.

---

## 6. Truy cập từ xa (đã chạy)

```
Team ══Tailscale══> laptop 100.106.160.24:2222 ══socat══> :2223 ══adb/USB══> AIBOX:22
```

Lệnh cho team — **không đổi kể cả khi laptop đổi wifi/4G**:
```bash
ssh -p 2222 thunopro@100.106.160.24
```

**Vì sao không chạy Tailscale thẳng trên AIBOX:** `tailscaled` chỉ đẩy logtail qua HTTP proxy, còn control plane và WireGuard (UDP) thì không → kẹt vĩnh viễn. Proxy HTTP là tầng ứng dụng; Tailscale cần route mạng tầng 3 thật. **Đừng thử lại đường này.**

Tự phục hồi khi rút/cắm cáp: `tools/aibox_bridge.sh` + `aibox-bridge.service` (đã test: xoá forward → 5s sau tự dựng lại).
```bash
bash tools/aibox_bridge.sh --status     # xem toàn cảnh + ai đang kết nối
```

---

## 7. ⛔ Quy tắc bất di bất dịch

**TUYỆT ĐỐI KHÔNG XOÁ USER NÀO** trên AIBOX (`hkt`, `nhchien`, `qrobot`, `thunopro`, `root`…), kể cả khi hết dung lượng. Máy dùng chung của lab, mỗi user là một project nghiên cứu riêng. Chi tiết: `note.txt` mục 0.

Khi cần chỗ trống, chỉ đụng vào (và phải hỏi trước): `/root/.cache/pip` (3.6G), các file `.tar.gz` đã giải nén. **Không** đụng `/home/*`, `/root/ai-rov`, `/opt/qcom`.

---

## 8. Việc tiếp theo, theo đúng thứ tự

1. **Mua/cắm loa USB** vào AIBOX — gỡ chặn cứng, mọi thứ khác đang chờ cái này.
2. **Đổi mật khẩu `thunopro`** — rủi ro bảo mật đang mở.
3. **Chụp bộ ảnh hiệu chỉnh** camera + điện thoại thật → bắt đầu Ưu tiên 1.
4. Test ASR bằng giọng thật.
5. Dọn nợ hạ tầng (coredump, linger, sửa tài liệu).
