## ⚠️ QUY TẮC TIẾT KIỆM TOKEN (BẮT BUỘC)
1. **Tuyệt đối ngắn gọn**: Đi thẳng vào vấn đề. Bỏ qua mọi câu chào hỏi, xin lỗi, cám ơn hay các đoạn văn dẫn dắt không cần thiết.
2. **Không lặp lại context**: Không nhắc lại yêu cầu của người dùng, không giải thích lại những gì đã ghi trong CLAUDE.md trừ khi được hỏi.
3. **Giới hạn code output**: Khi sửa code, **KHÔNG BAO GIỜ** in lại toàn bộ file. Chỉ in ra các đoạn code cần sửa (diff) kèm 2-3 dòng ngữ cảnh xung quanh để xác định vị trí.
4. **Dùng Bullet point**: Ưu tiên trình bày bằng gạch đầu dòng thay vì các đoạn văn dài.
5. **Không giải thích code dài dòng**: Chỉ giải thích logic khi thực sự có thay đổi phức tạp hoặc khi người dùng yêu cầu `explain`

# KIT — HLV cờ vua soi camera vào bàn Lichess


Cập nhật: 2026-08-13

## Mục tiêu

Camera Logitech C270 cắm vào AIBOX, soi vào bàn cờ Lichess trên màn hình laptop.
Hệ thống dựng "bàn cờ phụ" là thế cờ mà AI nhìn thấy, **liên tục bám nước đi** từ
thế khai cuộc, dùng Stockfish gợi ý nước và (dự kiến) LLM đánh giá tổng quan.
Giao diện web ở `http://127.0.0.1:8090`.

Nguyên tắc bám nước đi mà chủ dự án đã chốt: *xây trạng thái ở thế đầu tiên, rồi
track dần từng nước; điều quan trọng là ô hiện tại CÓ QUÂN hay KHÔNG, không cần biết
là quân gì* — và **kết hợp cả hai** đường (occupancy + danh tính quân), không bỏ
đường nào.

## Mọi thứ chạy ở đâu

| Thành phần | Chạy ở đâu | Ghi chú |
|---|---|---|
| Thị giác (rectify/OpenCV) | **AIBOX** aarch64 | trong tiến trình `coach_server.py` |
| PieceNet (437k tham số) | **AIBOX** | **NPU Hexagon HTP** qua QNN, INT8; lùi về ONNX CPU nếu sidecar chết |
| Stockfish | **AIBOX** | `/usr/games/stockfish`, tiến trình con |
| Web server + bàn phụ | **AIBOX** | `ThreadingHTTPServer` |
| Laptop | **chỉ là cửa sổ xem** | `adb forward tcp:8090`; không byte ảnh nào rời AIBOX. Rút laptop ra AIBOX vẫn chạy |
| **LLM đánh giá tổng quan** | **CHƯA NỐI** | xem Kế hoạch #7 |

## Kiến trúc

```
capture_thread          recog_thread                         _advice_worker
─────────────           ────────────────────────────         ──────────────
cap.read() ──► latest["frame"], latest["seq"]
                   │
                   ├─ Tracker.update  (bám 4 góc, 51ms)
                   ├─ rectify         (nắn 512×512 + chuẩn màu, 31ms)
                   ├─ OccupancyModel.predict  → mặt nạ 8×8 có/trống
                   ├─ explain_occ     (nở cây nước đi, 0.1–64ms)
                   │     ├─ 1 phương án            → chốt
                   │     └─ bằng điểm (ăn quân)    → PieceNet phân giải (10ms, NPU)
                   ├─ bỏ phiếu 3 hoặc 5 frame
                   └─ chốt nước ──► ADVICE_Q ────────────────► engine.analyse
```

## Bám nước đi: ĐỌC TRỰC TIẾP (`chess_ai/tracker3.py`) — thay tổ hợp + bỏ phiếu

```
S = đọc 64 ô → {trống, trắng, đen}      pixel lo "có/không", PieceNet lo "màu"
Δ = S khác THẾ CỜ ĐANG GIỮ ở ô nào       (neo vào thế cờ, KHÔNG neo vào frame trước)
Δ ổn định 3 frame → tra hình dạng ra (from,to) → LỌC bằng legal_moves → chốt
```

Luật cờ chuyển từ **máy sinh** sang **máy lọc**. Hết nở cây, hết giới hạn 2 nước.

Đo trên 274 cặp khung thật cách nhau đúng 1 nước:

| | Mới | Cũ (`explain_occ`) |
|---|---|---|
| Chốt ngay, 1 nước duy nhất | **96,4%** | 91,6% |
| Nhập nhằng, phải chờ | **0,0%** | 2,2% |
| Không giải thích được | **3,6%** | 6,2% |
| Thời gian | 10,5 ms | 14,1 ms |

Phân vai hai nguồn, đã đo chứ không phỏng đoán (1000 khung Lichess thật):

| Việc | Nguồn | Độ chính xác |
|---|---|---|
| Ô có quân hay không | pixel `OccupancyModel` | 0 ô bỏ sót |
| Quân màu gì | `PieceNet.read3` (3 lớp) | **99,96%** |

`read3` **cộng dồn** softmax 6 lớp trắng / 6 lớp đen chứ không argmax 13 lớp rồi
map: đọc nhầm Tốt thành Tượng vẫn ra đúng màu. Sai số giảm 8 lần so với 13 lớp
(99,562% → 99,948%), không train lại một giây nào.

### Hai thứ đã thử và THẤT BẠI, đừng làm lại

1. **Pixel đọc màu quân**: 3 quy tắc (trung vị / cực trị / thân quân) đều ~**50%**,
   đúng bằng tung đồng xu. Quân Lichess màu nào cũng gồm thân sáng + viền tối nên
   cân bằng sáng/tối vô nghĩa. Màu **bắt buộc** phải qua model.
2. **Đòi nước khớp Δ tuyệt đối**: nghe chặt chẽ nhưng chỉ đạt 67,9%, thua cả đường
   cũ. Occupancy dư ~0,5 ô mỗi khung nên một ô nhiễu ở góc bàn cũng giết nước đúng.
   Phải chấm bằng **sai lệch nhỏ nhất**.

### Vì sao kết hợp hai đường

Occupancy **không thể** phân biệt nước ăn quân: ô đích vốn đã có quân, nên `exd5` và
`exf5` cho mặt nạ **y hệt nhau**. Phong cấp cũng vô hình với occupancy. Vì vậy:

- **occupancy đi trước** — nhẹ, bền với ánh sáng, đủ cho mọi nước thường;
- **PieceNet chỉ vào đúng hai chỗ occupancy bó tay**: (a) nhiều nước bằng điểm,
  (b) occupancy không giải thích được gì.

`explain_occ` trả về **tất cả** phương án bằng điểm, không bao giờ tự chọn một cái
(trước đây nó chọn cái đầu tiên — tức tung xúc xắc ở mọi nước ăn quân).

## Các module

| File | Vai trò |
|---|---|
| `chess_ai/rectify.py` | Dò khung bàn, nắn thẳng, chuẩn hoá màu, `Tracker` bám real-time |
| `chess_ai/reader.py` | `placement`/`board_from_grid`/`locate_from_start`/`explain`/`explain_occ`/`pick_by_identity` |
| `chess_ai/gridfind.py` | `OccupancyModel` (ngưỡng sáng từng ô), `expected_mask`, `save_calib`/`load_calib` |
| `chess_ai/piece_net.py` | `PieceNet` (ONNX CPU) + `NpuPieceNet` (NPU qua socket), `predict` (13 lớp `.PNBRQKpnbrqk`), `white_is_bottom`, `occupancy` |
| `tools/npu_server.py` | Sidecar python3.12 giữ graph QNN, nghe `/tmp/kit_npu.sock` |
| `tools/build_npu.sh` | Dựng DLC INT8 cho NPU (chạy 1 lần trên AIBOX) |
| `tools/coach_server.py` | Web server + 3 luồng + toàn bộ vòng nhận diện |
| `tools/install_aibox.sh` | **Cài 1 lần**: systemd unit + bật lúc boot |
| `tools/run_aibox.sh` | Hàng ngày: đẩy code mới + restart dịch vụ |
| `tools/test_reader.py` | 31 phép thử, **không cần camera/model** |
| `tools/test_rectify.py` | Bộ 13 frame thật, sinh contact sheet để soi bằng mắt |

### Hằng số quan trọng

```
rectify:  SIZE=512  CELL=64  SCORE_SIZE=256
          ACCEPT=5.0        # frame nắn đúng đạt 11–20; quad sai ≤ 1.2
          EARLY_ACCEPT=10.0 # đạt mức này thì khỏi quét 600 mồi toàn khung
          MIN_INSIDE=0.995  # bàn phải nằm trọn trong khung
reader:   CONF_MIN=0.60  MAX_UNSURE=2  VOTES=3  RESYNC_AFTER=12
server:   need = 3 nếu (dư 0 ô và 1 nước) ngược lại 5
```

## Cơ chế giữ độ chính xác (đừng nới bất kỳ cái nào)

1. **Quad phải tự chứng minh**: điểm caro ≥ 5.0. Khoảng cách đo được rất rộng —
   frame đúng 11–20, quad sai ≤ 1.2.
2. **Tinh chỉnh 3 tầng**: `board_likeness` (FFT, trơn, không phụ thuộc pha) →
   `_phase_sweep` (thoát cao nguyên caro=0) → leo đồi trên caro. Bỏ tầng nào cũng
   chết một loại lệch riêng. Xếp hạng mồi bằng caro là vô nghĩa: quad đúng lệch 20px
   cho caro ~0.9 còn bản đã tinh chỉnh cho 13.28.
3. **`_quad_ok`**: chặn quad suy biến (diện tích, tỉ lệ cạnh, lồi, chống mảnh vụn).
   Thiếu nó `chk1.jpg` từng cho dương tính giả.
4. **`frame_coverage` tách riêng**: hình học đúng và dữ liệu đủ là hai loại lỗi khác
   nhau — bàn bị cắt thì gắn nhãn riêng, không cho dùng.
5. **Thống kê bền (trung vị/MAD)**: tay che vài ô không kéo lệch điểm.
6. **Chốt nước qua 3 cửa**: dư ≤ 1 ô, **và** đủ 3–5 phiếu trên frame KHÁC NHAU,
   **và** nhập nhằng thì từ chối chứ không đoán.
7. **`try_resync` chỉ nhận lời giải khớp 100% và DUY NHẤT.**

## Số đo thật trên AIBOX (có bàn trong khung)

| | |
|---|---|
| `detect` lúc canh bàn (1 lần) | **1506 ms** |
| `Tracker.update` mỗi frame | **51 ms** |
| `rectify` | **31 ms** |
| `explain_occ` khớp / nở sâu 2 | **0.1 ms / 64 ms** |
| `PieceNet.predict` **NPU** | **9.8 ms** (4.7 ms suy luận + 5 ms cắt ô/IPC) |
| `PieceNet.predict` CPU (đường lùi) | **479 ms** |
| `locate_from_start` 0/1/2/3 nước | 0.4 / 5 / 66 / 76 ms |
| Một vòng nhận diện | ~125 ms (khớp) → ~190 ms (không khớp) |

**Độ trễ bắt nước đi**: đường nhanh 3 phiếu × 125 ms ≈ **0.4 s**; đường chậm 5 phiếu
+ animation trừ phiếu ≈ **1.1–1.3 s**; nước ăn quân thêm PieceNet ≈ **2 s+**.

Chuỗi nhân quả của đường chậm: **Lichess tô sáng 2 ô của nước vừa đi → ngưỡng
occupancy học trên ô KHÔNG tô sáng nên sinh 1 ô lệch dư → `need` nhảy 3→5 → cộng
animation trượt quân ~0.2 s làm trừ phiếu**.

## Kết quả kiểm chứng

- `tools/test_reader.py`: **31/31 đạt** (chạy trên AIBOX).
- `tools/test_rectify.py`: **7/7 frame có bàn đúng, 0 dương tính giả**; `coach_raw`
  ra đúng verdict "BÀN BỊ CẮT"; 5 frame rác bị loại.
- Chạy thật: bám 25/25 frame, Stockfish gợi ý đúng, 19 nước tích luỹ.

## Các lỗi đã sửa (đừng tái phạm)

| Lỗi | Nguyên nhân thật |
|---|---|
| `run_aibox.sh` báo "không có camera" dù C270 đang cắm | AIBOX có `chess_coach.service` với `Restart=always`; `pkill` xong nó mọc lại trong ~1s và giữ tiếp `/dev/video2` → phải `systemctl stop` |
| coach_server lùi về CPU dù sidecar đang lên | Socket cũ chưa xoá nên `test -S` tưởng đã sẵn sàng trong lúc graph còn nạp 20s → `npu_server` xoá socket TRƯỚC khi nạp graph |
| kit_npu chết ngay khi `systemctl restart` | HTP cần vài giây nhả context tiến trình cũ; `Restart` tức thì → `Failed to check DLC has cache or not` rồi systemd bỏ cuộc → `RestartSec=10` |
| Sidecar NPU làm `run_aibox.sh` treo | `adb shell` chờ MỌI con đóng stdout; để lại `& echo started` là treo → chuyển hướng cho cả subshell nền rồi mới `&` |
| Occupancy cho margin âm | **Không phải moiré.** Warp đang bắt vào viền laptop + browser nên lưới lệch nửa bàn. Là lỗi định vị bàn |
| `chk1.jpg` dương tính giả (2.38) | Quad suy biến hình mảnh vụn → thêm `_quad_ok` |
| Đường quét toàn khung không bao giờ chạy | `if best_s < ACCEPT` khiến quad tầm tầm chặn mất đường quét → luôn xét cả hai nguồn |
| Nước ăn quân bị chọn bừa | `detect_move_occ` lấy phương án đầu → `explain_occ` trả tất cả, PieceNet tách |
| `build_reference` treo vĩnh viễn | PieceNet không chắc thì trả False, mà không nhánh nào gọi lại → rơi xuống đường occupancy |
| Bỏ mọi frame | `if model is None` chặn cả vòng khi `model` được phép None → chặn theo "không còn nguồn đọc nào" |
| **Cầm Đen + Trắng đi trước → bàn ĐỨNG IM** | `board_from_grid` thử lượt Trắng trước; thế sau 1.e4 với "Trắng đi" vẫn hợp lệ nên nhận sai lượt → `explain` chỉ sinh nước Trắng → nước Đen không bao giờ khớp. Sửa bằng `locate_from_start` |
| Stockfish chặn vòng nhận diện | `push_advice` gọi `engine.analyse` (0.35 s) ngay trong vòng → chuyển sang `ADVICE_Q` + luồng riêng |
| Ảnh camera bị delay tăng dần | Vòng quay hình ngủ 0.05 s trong khi driver vẫn xếp frame → frame đọc ra ngày càng cũ. Bỏ ngủ, chỉ hãm việc nén |
| `run_aibox.sh` treo | `adb shell` chờ tiến trình nền đóng stdout → `setsid nohup … </dev/null` |
| `run_aibox.sh` báo không có camera | Dò camera **trước** khi tắt server cũ (server còn giữ `/dev/video2`) → tắt trước, dò sau |
| `np.cross` lỗi | numpy 2 bỏ cross cho vector 2D → `cross2` tự viết |

## Cách chạy

Phần mềm **chạy trọn vẹn trên AIBOX**: source ở `/home/khacthu/kit`, hai systemd
unit tự bật khi bật máy. Laptop chỉ là cửa sổ xem — rút laptop ra vẫn chạy.

```bash
bash tools/install_aibox.sh          # LẦN ĐẦU: cài + bật lúc boot + chạy
bash tools/install_aibox.sh status   # xem trạng thái 2 dịch vụ
bash tools/run_aibox.sh              # hàng ngày: đẩy code mới + restart
bash tools/run_aibox.sh log          # xem log

adb shell 'cd ~khacthu/kit && python3 tools/test_reader.py'   # 31 phép thử
```

| Dịch vụ | User | Python | Việc |
|---|---|---|---|
| `kit_npu.service` | khacthu | 3.12 (venv) | PieceNet trên NPU, nghe `/tmp/kit_npu.sock` |
| `chess_coach.service` | khacthu | 3.10 (hệ thống) | Thị giác + Stockfish + web 8090 |

`khacthu` phải ở group **`video`** (mở `/dev/video2`) và **`system`** (mở
`/dev/adsprpc-smd` để dùng NPU). Thiếu group nào thì `install_aibox.sh` báo ngay.

Không đóng đinh đường dẫn nữa: mọi module lấy gốc từ `KIT_ROOT` (mặc định suy từ
vị trí file), nên chuyển thư mục hay đổi user chỉ cần sửa systemd unit.

Máy laptop **không có** `cv2` và `python-chess`, nên mọi test phải chạy trên AIBOX.

## NPU (Hexagon HTP) cho PieceNet

Dựng **một lần**, cần internet ở bước cài venv:

```bash
python3 tools/usb_proxy.py &     # laptop chia internet qua USB
bash tools/build_npu.sh          # venv 3.12 + ONNX→DLC→INT8 + đo + đối chiếu
```

Sau đó `install_aibox.sh` dựng `kit_npu.service`. Kiểm tra: `adb shell journalctl -u kit_npu -n 20`.

Vì sao ăn thua lớn: PieceNet tốn **164 MMAC/ô × 64 ô = 21 GFLOP mỗi bàn**, mà 6 lõi
Kryo chỉ kéo được ~50 GFLOPS → 430 ms. HTP làm cùng khối lượng đó trong 4.7 ms.

Những chỗ dễ vấp (đã trả giá):

- **API python của QAIRT bắt buộc python 3.12**, hệ thống là 3.10 → phải chạy sidecar
  riêng, không nạp thẳng vào `coach_server`. Giao tiếp bằng unix socket, gửi ô dạng
  uint8 (786KB) thay vì float32 (3MB) — giá trị y hệt vì pixel vốn là số nguyên 0–255.
- Gói `python3.12` của deadsnakes **không kèm `libpython3.12.so`**, mà QAIRT `dlopen`
  nó → phải cài thêm gói `libpython3.12`.
- `onnx` phải là **1.16.x**: bản 1.22 bỏ `onnx.version` nên converter QAIRT 2.48 gãy.
  Và python 3.12 bỏ `distutils` → cần `setuptools<81`.
- `qnn-net-run` nhận DLC qua `--dlc_path`, **không phải** `--model` (`--model` chỉ
  nhận `.so`; đưa nhầm chỉ báo cụt lủn "Initialization failure").
- **Phải gọi `Inferencer.setup()` một lần rồi giữ**. Gọi `run()` trực tiếp khiến nó
  biên dịch lại graph mỗi lần: 2130 ms/lần, tức chậm hơn cả CPU.
- FP32 không chạy trên HTP — bắt buộc lượng tử hoá.

Độ chính xác: hiệu chuẩn INT8 bằng 100 frame thật, kiểm định trên 100 frame **khác**
(6400 ô): NPU INT8 cho kết quả **trùng 100%** với ONNX CPU fp32, độ chính xác 99.98%
cả hai. Nên đây là đổi tốc độ lấy *không gì cả*.

## Kế hoạch tiếp

Xếp theo mức thiệt hại. 1–3 là vỏ thuần, tổng ~20 dòng, đánh đúng độ trễ hiện tại.

1. **Tự hiệu chỉnh ngưỡng occupancy khi khớp 100%** — mỗi khi dư = 0, lấy chính thế
   cờ đang đúng làm nhãn và fit lại `OccupancyModel`. Nhãn miễn phí và luôn đúng.
   Gỡ được đúng chuỗi nhân quả gây độ trễ (highlight Lichess → lệch dư → 5 phiếu),
   và xoá luôn hiện tượng `n > 32` đo được ~4%. *(~10 dòng)*
2. **Nhớ kết quả PieceNet theo mặt nạ occupancy** — 3 phiếu cùng mặt nạ thì danh
   tính quân không thể khác, nên đọc 1 lần thay vì 3. Cắt ~1 s ở nước ăn quân.
   *(~6 dòng)*
3. **Chỉ tính phiếu trên frame có occupancy đứng yên so với frame trước** — hết cảnh
   bộ đếm bị reset/trừ giữa lúc quân đang trượt. *(~3 dòng)*
4. **Nâng `try_resync`** — hiện chỉ thử ±2 nước, bỏ lỡ 3 nước trở lên là không có
   đường tự về. Bí quá lâu (~40 nhịp) mà PieceNet đọc chắc → dựng lại thế cờ bằng
   `board_from_grid`. *(~10 dòng)*
5. **Mất bàn → thử `rectify.verify` với quad đã lưu trước khi quét toàn khung** —
   camera thường bị chạm rồi về gần chỗ cũ, khỏi trả giá 1506 ms. *(~5 dòng)*
6. **Vào giữa ván không biết lượt đi** — `locate_from_start` chỉ dò 3 nước. Giữ **cả
   hai giả thuyết lượt**, nước đầu tiên giải thích được sẽ tự loại giả thuyết sai.
   *(~15 dòng)*
7. **Nối LLM đánh giá tổng quan** — `chess_ai/llm.py` và `llm_commentary.py` đã có,
   trỏ `OLLAMA_URL` mặc định `127.0.0.1:11434`, tức thiết kế ban đầu **đã là chạy
   local**. Nhưng `coach_server.py` chưa gọi tới, và AIBOX chưa cài `ollama`. Hai
   hướng, cần chủ dự án quyết:
   - **Ollama trên AIBOX**: giữ được "toàn bộ trên thiết bị", nhưng CPU ARM nên chỉ
     model cỡ `qwen2.5:1.5b` chạy nổi, mỗi lời bình vài giây. **Phải ở luồng riêng**,
     tuyệt đối không nằm trong vòng hiển thị.
   - **Gọi LLM ngoài**: nhanh và thông minh hơn nhiều, nhưng phá vỡ tính chất trên
     và phải gửi FEN ra ngoài.
8. **Fine-tune PieceNet pha 2** bằng dữ liệu thật đang được dump ở
   `~khacthu/kit/captures/dataset` (warp + FEN, nhãn tự động đúng vì chỉ dump khi khớp
   100%): `--real-dir … --real-p 0.5 --init piece_net.pt`. Pha 1 (thuần tổng hợp)
   đạt val_acc 99.5–100% nhưng trên frame thật vẫn đọc tốt c2/e2 thành Tượng và rơi
   quân hàng cuối ở frame tối. Đây là lý do occupancy phải là đường chính.

**Một phương án cố tình KHÔNG làm**: hạ số phiếu từ 3 xuống 2. Nhanh thật nhưng đánh
đổi độ chính xác, mà chủ dự án đã yêu cầu rõ độ chính xác phải luôn được giữ. Mục 1–3
lấy lại phần lớn thời gian đó mà không nới chốt chặn nào.

## Bàn cờ số `digital_board/` (v1.0) — nửa không cần phần cứng

Bàn cờ trên trình duyệt, bấm chuột để đi, Stockfish gợi ý nước tiếp theo. Chạy trên
máy thường, chỉ phụ thuộc `python-chess`. Chi tiết: `docs/DIGITAL_BOARD.md`.

```bash
run.bat / ./run.sh                    # tự cài dep + tải Stockfish + mở :8090
python -m digital_board.server --port 8090 --movetime 300 --no-browser
python tools/get_stockfish.py --from <path>   # có sẵn binary thì khỏi tải
python tests/test_digital_board.py    # 63 phép thử, exit 0 là đạt
```

- **Hai nửa KHÔNG import nhau.** `chess_ai/` = camera + NPU trên AIBOX;
  `digital_board/` = bàn cờ số. Cả hai cùng cổng 8090 → chạy lần lượt.
- **Cố ý có hai lớp bọc Stockfish.** `chess_ai.engine.ChessEngine` đóng đinh
  `/usr/games/stockfish`, không cache, không có `loss` → không chạy được trên Windows,
  nơi bàn cờ số phải chạy khi chưa có phần cứng. Chỗ duy nhất hai bên gặp nhau là
  `digital_board.engine.find_binary()`, có đọc `chess_ai.config.STOCKFISH_PATH` (trong
  `try/except`, vì máy chạy bàn cờ số không có `cv2`) để trên AIBOX dùng chung binary.
- **Binary Stockfish KHÔNG commit** (~80 MB, GPL-3, theo nền tảng). `engine/` bị
  gitignore; `tools/get_stockfish.py` tra releases API chứ không đóng đinh URL vì tag
  có đổi (`sf_17` → `sf_18`).
- **`.gitignore` phải viết `engine/*` chứ không phải `engine/`** — git không đi vào thư
  mục đã loại nên `!engine/.gitkeep` sẽ không bao giờ có tác dụng.
- Đo được: gợi ý ở movetime 300 ms ≈ 550 ms trọn vòng lúc nguội, lần lặp lại ~0 ms và
  báo `cached`. `1.e4 c5` → `Nf3`, eval +0.38.

## Ràng buộc

- **NGHIÊM CẤM xoá user trên AIBOX** — không `userdel`/`deluser`, không xoá home của
  bất kỳ ai, kể cả khi cần chỗ trống. AIBOX có 11 user.
- Máy GPU để train: `ssh user@100.95.255.65` qua Tailscale (mật khẩu hỏi chủ dự án;
  dùng `export SSHPASS=…; sshpass -e`, không truyền mật khẩu trong argv).
- Đo tốc độ **phải có bàn trong khung**. Frame không có bàn cho đường xấu nhất
  (mọi mồi lang thang) nên số đo sai lệch hoàn toàn — đã bị nhầm một lần.
- Còn một mắt xích **chưa đo**: tốc độ frame thật của C270. Đang ở phơi sáng tự động,
  và C270 tự hạ xuống 7.5 fps khi thiếu sáng — nếu vậy mỗi frame cách 133 ms thay vì
  33 ms và mọi con số độ trễ phải nhân thêm.
