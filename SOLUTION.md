# KIT — Huấn luyện viên cờ vua nhìn qua camera, chạy trọn trên AIBOX

Một chiếc camera Logitech C270 cắm vào AIBOX 8550, soi vào bàn cờ Lichess đang mở
trên màn hình laptop. Hệ thống tự dựng lại thế cờ nó nhìn thấy, bám theo từng nước
đi, và gợi ý nước tốt nhất bằng Stockfish. Toàn bộ thị giác, mô hình học sâu và
engine đều chạy **trên AIBOX**; laptop chỉ là một cửa sổ để xem. Rút laptop ra,
hệ thống vẫn chạy.

Giao diện: `http://127.0.0.1:8090`

---

## 0. Đối chiếu nhanh với tiêu chí chấm

| Tiêu chí | Ở đâu trong tài liệu | Bằng chứng ngắn gọn |
|---|---|---|
| **Tính sáng tạo** | Mục 2, 3 | Dùng luật cờ vua làm **bộ lọc** thay vì bộ sinh: 282 ms → 10,5 ms, và 96,4% chốt ngay. Neo vào thế cờ nên đọc nhầm một khung chỉ gây chậm chứ không gây lệch. |
| **Đúng tinh thần Edge AI** | Mục 5, 5b | **Hai mô hình** cùng chạy trên NPU Hexagon HTP: PieceNet (thị giác, 9,8 ms) và Whisper-Small (giọng nói tiếng Việt). Thị giác + mô hình + Stockfish + web đều trên thiết bị. **Rút dây laptop ra, hệ thống vẫn chạy** — laptop chỉ là màn hình. |
| **Làm chủ quy trình model → convert → optimize → deploy → app** | Mục 5 (toàn bộ chuỗi), 5b | Tự sinh dữ liệu → train PyTorch → ONNX → DLC → lượng tử hoá INT8 bằng ảnh thật → đối chiếu độ chính xác → systemd tự chạy khi bật máy → ứng dụng web. Mỗi mắt xích đều có script chạy lại được. |
| **Demo hoạt động thực tế** | Mục 11, 12 | Chạy thật với camera C270, có bảng chẩn đoán số liệu ngay trên giao diện. Kịch bản demo và phương án dự phòng ở mục 12. |

---

## 1. Bài toán khó ở chỗ nào

Nhìn thì đơn giản: chụp bàn cờ, nhận diện 64 ô, xong. Thực tế có ba cái bẫy.

**Bẫy thứ nhất — đọc đúng từng ô không có nghĩa là đọc đúng cả bàn.** Mô hình của
chúng tôi đọc đúng **98,07%** mỗi ô. Nghe rất cao. Nhưng một thế cờ cần đúng
*đồng thời* cả 64 ô, và khi đo trên dữ liệu thật thì tỉ lệ đúng cả bàn chỉ còn
**85,2%**. Cứ 7 khung hình thì có 1 khung cho ra thế cờ sai. Không thể xây một
huấn luyện viên trên nền đó.

**Bẫy thứ hai — sai một lần là sai vĩnh viễn.** Nếu so khung hình này với khung
hình trước để tìm nước đi, chỉ cần một lần đọc nhầm là thế cờ trôi lệch và không
bao giờ tự về.

**Bẫy thứ ba — camera không đứng yên.** Người ta chạm vào bàn, ánh sáng đổi, C270
tự chỉnh phơi sáng, Lichess tô sáng hai ô của nước vừa đi.

## 2. Ý tưởng cốt lõi: dùng luật cờ vua làm bộ lọc, không phải bộ sinh

Đây là quyết định thiết kế quan trọng nhất của cả dự án.

Cách làm thông thường là *sinh* mọi nước đi hợp lệ rồi dựng lại bàn cờ cho từng
nước để xem cái nào khớp ảnh. Cách đó tốn kém: ở thế giữa ván có ~30 nước hợp lệ,
nếu phải tính cả hai nước liên tiếp là ~900 thế cờ phải dựng và so — **đo được
282 ms mỗi khung**, và vẫn chết cứng khi lỡ mất nhiều hơn hai nước.

Chúng tôi làm ngược lại:

```
S  = đọc thẳng 64 ô  →  {trống, trắng, đen}
Δ  = S khác THẾ CỜ ĐANG GIỮ ở những ô nào
Δ  ổn định 3 khung  →  tra hình dạng Δ ra (from, to)  →  LỌC bằng legal_moves  →  chốt
```

Luật cờ vua chuyển từ vai trò *máy sinh* sang vai trò *máy lọc*. Việc tra một
nước có hợp lệ hay không chỉ là một phép tìm trong `legal_moves`.

**Neo vào thế cờ, không neo vào khung hình trước.** Thế cờ chỉ thay đổi khi chốt
được một nước *hợp lệ*, nên nó là cái mốc không bao giờ trôi. Đọc nhầm một khung
chỉ gây chậm, không gây lệch — khác hẳn kiểu so khung-với-khung.

Kết quả, đo trên 274 cặp khung thật cách nhau đúng một nước:

| | Cách mới (lọc) | Cách cũ (sinh) |
|---|---|---|
| Chốt ngay, đúng một nước | **96,4%** | 91,6% |
| Nhập nhằng, phải chờ | **0,0%** | 2,2% |
| Không giải thích được | **3,6%** | 6,2% |
| Thời gian | **10,5 ms** | 14,1 ms |

## 3. Hai nguồn đọc, mỗi nguồn chỉ được hỏi đúng câu nó giỏi

| Câu hỏi | Nguồn trả lời | Độ chính xác |
|---|---|---|
| Ô này **có quân hay không**? | Ngưỡng pixel (`OccupancyModel`) | 92,3% đúng cả bàn |
| Quân đó **màu gì**? | `PieceNet.read3` (mạng CNN) | **99,96%** |

Chúng tôi không hỏi "quân gì" — vì để bám nước đi thì **không cần biết**. Biết ô
đích đổi chủ từ đen sang trắng là đủ để phân biệt một nước ăn quân.

Hai chi tiết kỹ thuật đáng nói:

**`read3` cộng dồn xác suất theo màu thay vì lấy nhãn thắng rồi quy ra màu.** Mô
hình đọc nhầm Tốt thành Tượng vẫn cho ra đúng màu. Chỉ đổi cách gộp xác suất, sai
số giảm **8 lần** (99,562% → 99,948%) mà không train lại một giây nào.

**Chấm điểm hai tầng.** Tầng quyết định là *đếm số ô lệch* (thang cứng); tầng phá
hoà mới dùng xác suất màu. Chúng tôi từng thử cho xác suất quyết định luôn và tỉ
lệ tụt từ 96,4% xuống 86,9% — mô hình quá tự tin, nên một ô đọc sai màu tốn tới
hàng chục đơn vị `−log p`, đắt hơn cả phương án "không đi nước nào", khiến nước
đúng bị loại. Xác suất chỉ đáng tin khi so sánh *tương đối* giữa các phương án đã
ngang điểm.

### Ba luật chặn trước khi ghép nước

Rẻ, và loại được phần lớn khung hình rác:

1. **Số quân không bao giờ tăng.** Không nước cờ nào sinh thêm quân. Ảnh đọc ra
   nhiều quân hơn thế cờ đang giữ nghĩa là pixel đang báo thừa vì loá — bỏ khung.
2. **Một nước ăn nhiều nhất một quân.** Thiếu từ hai quân trở lên là bàn tay đang
   che, không phải nước đi.
3. **Chống dội ngược.** Nước đảo ngược đúng nước vừa chốt gần như luôn là đọc nhầm
   giữa lúc quân đang trượt trên màn hình.

## 4. Những thứ chúng tôi thử và **thất bại** — và vì sao vẫn kể ra

Phần này quan trọng ngang phần thành công, vì mỗi thất bại đều loại bỏ một giả
thuyết nghe rất hợp lý.

**Dùng pixel để đọc màu quân — hỏng.** Ba quy tắc khác nhau cho ra 50–81%. Lý do
vật lý: quân cờ Lichess màu nào cũng gồm thân sáng lẫn viền tối, nên cân bằng
sáng/tối không mang thông tin màu. Màu **bắt buộc** phải qua mô hình.

**Đòi nước đi khớp Δ tuyệt đối — hỏng.** Nghe rất chặt chẽ, nhưng chỉ đạt 67,9%.
Pixel báo thừa trung bình ~0,5 ô mỗi khung, nên một ô nhiễu ở góc bàn cũng đủ
giết nước đi đúng. Phải chấm bằng **sai lệch nhỏ nhất**, không phải khớp hoàn hảo.

**Cắt đều bốn cạnh ảnh để tránh viền bàn — hỏng.** Tỉ lệ tụt 86,9% → 73,5%. Lệch
chỉ ở *một* cạnh, nên co đều cả bốn là tự kéo lệch cả lưới.

**Tự hiệu chỉnh ngưỡng bằng cách thay hẳn mô hình cũ — hỏng.** Chi tiết ở mục 6;
đây là lỗi tinh vi nhất chúng tôi tìm ra.

## 5. Toàn bộ chuỗi Edge AI: từ không có dữ liệu đến mô hình chạy trên NPU

Đây là phần chúng tôi muốn trình bày kỹ nhất, vì nó đi trọn vẹn từ đầu đến cuối.

```
[1] Sinh dữ liệu  →  [2] Huấn luyện  →  [3] Xuất ONNX  →  [4] Chuyển DLC
                                                                 ↓
[7] Ứng dụng web  ←  [6] Triển khai systemd  ←  [5] Lượng tử hoá INT8 + đối chiếu
```

### Bước 1 — Sinh dữ liệu, vì chúng tôi khởi đầu với con số không

Không có sẵn bộ ảnh nào chụp *camera soi màn hình đang chạy cờ*. Nên chúng tôi
sinh vô hạn dữ liệu tổng hợp (`tools/train/gen_cells.py`), mô phỏng đúng điều kiện
thật: 12 bộ quân Lichess × nhiều màu bàn × nhiễu của cảnh quay màn hình — mờ,
moiré, loá, lệch lưới, nén JPEG, thay đổi phơi sáng, highlight nước đi.

Chi tiết đáng chú ý: mỗi mẫu được dựng trên khung **3×3 ô** rồi mới cắt ô giữa.
Ngoài đời lưới luôn lệch đôi chút nên ô cắt ra sẽ dính mép quân hàng xóm — mô hình
phải học cách phớt lờ chúng ngay từ đầu.

### Bước 2 — Huấn luyện

`PieceNet`: CNN nhỏ **437 nghìn tham số**, 13 lớp (ô trống + 6 loại quân × 2 màu),
đầu vào 64×64 BGR. Huấn luyện bằng PyTorch trên máy GPU riêng, dữ liệu sinh ngay
trong lúc train chứ không lưu ra đĩa.

Một quyết định nhỏ nhưng trả cổ tức về sau: **đưa phép chuẩn hoá vào bên trong mô
hình** (`(x−127,5)/63,75`). Nhờ vậy phía suy luận chỉ cần đẩy ảnh uint8 vào, không
phải nhớ hệ số — và quan trọng hơn, dải đầu vào cố định 0–255 khiến việc lượng tử
hoá ở bước 5 dễ và ổn định hơn nhiều.

### Bước 3 và 4 — Xuất ONNX rồi chuyển sang DLC

ONNX là bản chạy được trên CPU và cũng là đường lùi khi NPU trục trặc.
`qairt-converter` chuyển sang định dạng DLC của Qualcomm, **cố định batch = 64 ô**
đúng bằng một bàn cờ, để một lần gọi xử lý trọn vẹn một khung hình.

### Bước 5 — Lượng tử hoá INT8, hiệu chuẩn bằng ảnh thật

FP32 không chạy được trên HTP, nên lượng tử hoá là **bắt buộc** chứ không phải
tuỳ chọn. Rủi ro thường trực là mất độ chính xác.

Chúng tôi hiệu chuẩn bằng **100 khung hình thật** đã lưu, rồi kiểm định trên **100
khung khác** (6400 ô) mà mô hình chưa từng thấy trong lúc hiệu chuẩn.

> **Kết quả: NPU INT8 cho kết quả trùng 100% với ONNX CPU fp32**, cả hai cùng đạt
> 99,98%. Nghĩa là đổi tốc độ mà **không trả giá gì cả**.

Vì sao đáng đổi: `PieceNet` tốn **164 MMAC mỗi ô × 64 ô = 21 GFLOP cho mỗi bàn
cờ**. Sáu nhân CPU Kryo chỉ kéo được ~50 GFLOPS, tức **479 ms** — quá chậm để chạy
mỗi khung, và đó chính là lý do ban đầu cả pipeline phải xoay quanh pixel. Trên
NPU: **9,8 ms** (4,7 ms suy luận + 5 ms cắt ô và truyền dữ liệu). **Nhanh hơn 49
lần**, và chính nó cho phép gọi mạng nơ-ron ở *mọi* khung hình thay vì thỉnh thoảng.

### Bước 6 — Triển khai

Hai dịch vụ systemd tự bật khi cắm điện AIBOX:

| Dịch vụ | Python | Việc |
|---|---|---|
| `kit_npu.service` | 3.12 (môi trường riêng) | giữ graph QNN, nghe unix socket |
| `chess_coach.service` | 3.10 (hệ thống) | thị giác + Stockfish + web 8090 |

**Vì sao phải tách hai tiến trình:** API Python của QAIRT **bắt buộc Python 3.12**,
trong khi hệ thống chạy 3.10. Không thể nạp thẳng vào tiến trình chính, nên phải
tách một tiến trình phụ giữ sẵn graph và giao tiếp qua unix socket. Ô cờ được gửi
dạng `uint8` (786 KB) thay vì `float32` (3 MB) — giá trị y hệt vì pixel vốn là số
nguyên 0–255, mà tiết kiệm được 4 lần băng thông.

### Những chỗ đã vấp và trả giá

Ghi lại đầy đủ, vì mỗi cái đều tốn hàng giờ:

- **Phải gọi `Inferencer.setup()` một lần rồi giữ nguyên.** Gọi `run()` trực tiếp
  khiến nó biên dịch lại graph mỗi lần: **2130 ms**, tức *chậm hơn cả CPU*. Đây là
  cái bẫy nguy hiểm nhất vì mọi thứ vẫn "chạy đúng", chỉ là chậm khủng khiếp.
- `qnn-net-run` nhận DLC qua `--dlc_path`, **không phải** `--model` (cờ đó chỉ nhận
  file `.so`; đưa nhầm chỉ báo cụt lủn "Initialization failure").
- Gói `python3.12` của deadsnakes **không kèm `libpython3.12.so`**, mà QAIRT lại
  `dlopen` nó → phải cài thêm gói `libpython3.12`.
- `onnx` phải đúng bản **1.16.x**: bản 1.22 bỏ `onnx.version` nên converter gãy.
- Dịch vụ NPU chết ngay khi khởi động lại: HTP cần vài giây nhả ngữ cảnh của tiến
  trình cũ → phải đặt `RestartSec=10`.
- Tiến trình chính lùi về CPU dù NPU đã sẵn sàng: socket cũ chưa xoá nên phép kiểm
  tra tưởng đã sẵn sàng trong khi graph còn đang nạp 20 giây → phải xoá socket
  **trước** khi nạp graph.

### Bước 7 — Ứng dụng

Máy chủ web ba luồng ngay trên AIBOX, kèm bảng chẩn đoán trực tiếp (mục 11).

---

## 5b. Mô hình thứ hai trên NPU: ra lệnh bằng giọng nói tiếng Việt

Ngoài `PieceNet`, chúng tôi đưa thêm **Whisper-Small lượng tử hoá** lên chính con
NPU đó, để điều khiển hệ thống bằng tiếng Việt:

```
nói "đánh cho tôi nước tiếp theo"
  → Whisper-Small trên Hexagon NPU  (~1 s)  → chữ
  → lớp hiểu lệnh                           → goi_y
  → Stockfish                               → hiện + đọc "Nước đi tốt nhất: e4"
```

Encoder và decoder là hai QNN context binary riêng: encoder chạy **một lần** mỗi
câu nói, decoder chạy **một lần mỗi token** sinh ra. Hai chi tiết khiến việc nối
chuỗi khớp từng byte, đọc thẳng từ `metadata.json`:

- các cache `k/v_cache_self_*_out` mang **đúng cùng** scale và zero-point với
  `..._in`, nên cache đầu ra của một bước được đưa thẳng vào bước sau mà không
  phải lượng tử hoá lại;
- cache cross-attention của encoder khớp đầu vào decoder cùng tên, nên chỉ ghi
  một lần rồi tái sử dụng cho mọi token.

### Bài học đắt nhất: mô hình bịa rất trôi chảy

Whisper **bịa** khi nghe tiếng ồn. Trên chính board này, tiếng ồn phòng nhiều lần
cho ra câu *"Hãy subscribe cho kênh Ghiền Mì Gõ…"* — một câu hoàn chỉnh, tự tin,
và hoàn toàn không có thật. Nếu không chặn, mọi lớp phía sau sẽ xử lý một mệnh
lệnh bịa và engine sẽ **thi hành nó thật**.

Nên kiến trúc ba lớp, mà **thứ tự quan trọng hơn bản thân từng lớp**:

1. **Cổng tin cậy đứng đầu.** Điều thú vị: chỉ số `no_speech` có sẵn của Whisper
   **vô dụng trên board này** — cho 0,72–0,90 với cả tiếng nói sạch lẫn im lặng.
   Phải dùng `avg_logprob` với ngưỡng −1,0 mới tách được.
2. **Khớp từ khoá** — 0–1 ms, phủ hầu hết cách nói thường gặp.
3. **Mô hình ngôn ngữ đứng cuối, và bị ràng buộc bằng grammar GBNF** vào đúng tập
   lệnh cho phép, trong đó luôn có `khong_hieu` để nó còn đường từ chối.

**Vì sao mô hình ngôn ngữ phải đứng cuối và phải bị ràng buộc:** đưa thẳng chữ
nhận dạng thô cho nó và bảo "sửa lại giúp" thì một câu quảng cáo YouTube sẽ được
biến thành một mệnh lệnh cờ nghe rất hợp lý — tức là **rửa nhiễu thành mệnh lệnh**.

Bù lại, nó làm được đúng việc mà từ khoá chịu thua: đọc xuyên qua chữ nghe lệch.
Mô hình lượng tử hoá nghe "tấn công cánh phải" thành "tính công cảnh phải" — sai
2/4 âm tiết, biểu thức chính quy bó tay, mô hình ngôn ngữ vẫn hiểu.

Một đánh đổi đã đo: có grammar thì llama.cpp phải kiểm ràng buộc trên toàn bộ
151.936 token ở **mỗi bước**, tốc độ tụt từ ~6 xuống ~1,7 token/giây. Nghĩa là độ
dài câu trả lời **chính là** độ trễ — nên bắt nó trả lời đúng **một từ**.

`tools/test_voice.py` chạy được cả chuỗi mà không cần người ngồi trước micro: dùng
giọng tổng hợp thay lời nói, nhưng âm thanh thật, log-mel thật, NPU thật.

> **Trạng thái:** đã chạy được và kiểm chứng riêng, **chưa đấu vào `coach_server`**.
> Mã nguồn ở `chess_ai/voice.py`, `whisper/`, `tools/install_voice.sh`. File trọng
> số (`encoder.bin`, `decoder.bin` — 358 MB) không đưa lên GitHub; `install_voice.sh`
> chép chúng vào đúng chỗ trên thiết bị.

## 6. Ba lỗi tìm ra bằng cách mô phỏng lại trên dữ liệu thật

Chúng tôi dựng một bộ phát lại (`tools/replay_server.py`) chạy đúng vòng nhận diện
trên **1000 khung hình thật đã lưu kèm thế cờ đúng**, không cần camera. Nhãn luôn
đúng vì hệ thống chỉ lưu khung khi nhận diện khớp 100%. Nhờ vậy mọi thay đổi đều
**đo được** thay vì phỏng đoán.

### Lỗi 1 — Lưới cắt ô sát mép bàn

Phóng to ảnh đã nắn thẳng thì thấy mép trái vẫn còn dính **~14 px nền trắng ngoài
bàn**, trong khi lõi ô chỉ cách mép ô 15 px. Biên an toàn gần bằng không: hễ camera
rung là dải nền lọt vào lõi và ô trống bị đọc thành có quân. **42% toàn bộ lỗi dồn
vào đúng một cột.**

Vì sao mãi không lộ ra: cột a ở thế khai cuộc **luôn có quân**, nên lỗi bị che
hoàn toàn. Đến nước 10–15, khi tốt cột a đã đi hoặc bị ăn, ô mới trống ra và lỗi
có sẵn từ đầu mới hiện hình. Đó chính là lý do người dùng thấy "chơi tới nước 10,
15 thì nhận diện kém hẳn".

Cách sửa: căn lại lưới ngay trên ảnh đã nắn, leo đồi trên tương phản caro của nền
ô, **bốn cạnh độc lập nhau**.

### Lỗi 2 — Tính rất lâu rồi bảo đảm không ra kết quả

Khi lệch từ 2 ô trở lên, đường dự phòng nở cây nước đi sâu 2 tầng: 729 thế cờ,
**282 ms**. Nhưng điều kiện chấp nhận là "còn lệch ≤ 1 ô", mà 4 ô lệch thì một
nước đi (chỉ đổi 2–4 ô) không kéo nổi xuống. **Toàn bộ 729 phép tính đó chắc chắn
bị vứt.** Càng nhiễu thì càng tính lâu và càng chắc chắn vô ích.

Cách sửa: chặn ngay ở cửa vào. Từ 4 ô lệch trở lên thì đó là lỗi căn lưới hoặc ánh
sáng, không phải nước đi — bằng chứng là 291 ô báo thừa so với đúng **1** ô bỏ sót.

### Lỗi 3 — Cơ chế tự hiệu chỉnh đang làm hệ thống tệ đi

Đây là lỗi tinh vi nhất. Hệ thống tự hiệu chỉnh ngưỡng bằng cách lấy thế cờ đang
giữ làm nhãn — nhãn miễn phí và luôn đúng. Nghe hoàn hảo. Đo ra: **90,9% → 87,8%**.

Hai nguyên nhân, và cả hai đều thuộc cùng một họ sai lầm:

1. **Thay hẳn mô hình cũ bằng mô hình vừa khớp.** Nếu khung đó có một kênh đặc
   trưng không tách được, phép gán vứt luôn ngưỡng tốt của kênh đó đi.
2. **Đòi khớp tuyệt đối mới cho phép hiệu chỉnh.** Khi lệch mãn tính, điều kiện đó
   không bao giờ đạt → cơ chế tự sửa **không bao giờ chạy, đúng lúc cần nó nhất**.

Cùng một họ sai lầm với hàm tự đồng bộ `try_resync`: nó đòi khớp 100% tuyệt đối,
trong khi ô báo thừa là lỗi *hình học* — không thế cờ nào trên đời làm nó biến
mất. Hàm đó đốt 122 ms mỗi lần gọi và **cấu trúc bảo đảm luôn thất bại**.

> **Bài học rút ra:** một điều kiện đặt quá chặt không làm hệ thống an toàn hơn —
> nó làm cơ chế bảo vệ tự vô hiệu hoá, im lặng.

### Kết quả tổng hợp trên 1000 khung thật

| | Ban đầu | Sau khi sửa |
|---|---|---|
| Khung khớp hoàn toàn | 86,9% | **92,3%** |
| Ô báo thừa | 291 | **113** |
| Khung lệch ≥2 ô (rơi vào đường chậm) | 72 | **24** |
| Ở nước 10–14 | 70,7% | **82,9%** |
| Ở nước 20–24 | 81,4% | **94,9%** |

## 7. Số đo thật trên AIBOX

| Bước | Thời gian |
|---|---|
| Dò bàn lần đầu (một lần lúc canh) | 1506 ms |
| Bám 4 góc mỗi khung | 51 ms |
| Nắn ảnh | 31 ms |
| Đọc ô có quân/trống | 15 ms |
| **PieceNet trên NPU** | **9,8 ms** |
| PieceNet trên CPU (đường lùi) | 479 ms |
| Bám nước đi (`tracker3`) | 10,5 ms |
| **Một vòng nhận diện** | **64–74 ms** |

Độ trễ từ lúc người chơi đi một nước đến lúc hệ thống nhận ra: khoảng **0,4 giây**.

## 8. Kiến trúc

```
capture_thread            recog_thread                          _advice_worker
──────────────            ─────────────────────────────         ──────────────
cap.read() ──► latest["frame"]
                   │
                   ├─ Tracker.update      bám 4 góc bàn            51 ms
                   ├─ rectify             nắn thẳng 512×512        31 ms
                   ├─ grid_box            căn lại lưới 8×8          5 ms
                   ├─ OccupancyModel      ô nào có quân            15 ms
                   ├─ PieceNet.read3      quân màu gì (NPU)        10 ms
                   ├─ tracker3.detect     Δ → lọc bằng luật cờ     10 ms
                   └─ chốt nước ──► ADVICE_Q ──────────────────► Stockfish
```

Ba luồng tách rời có lý do: Stockfish tốn 0,35 giây mỗi lần phân tích. Trước đây
nó nằm ngay trong vòng nhận diện, nên mỗi lần chốt được một nước là vòng đó đứng
lại — và nước trả lời của đối thủ bị bắt muộn đúng bằng khoảng đó.

## 9. Cấu trúc mã nguồn

```
chess_ai/            thư viện lõi
  rectify.py         dò khung bàn, nắn thẳng, bám 4 góc real-time
  gridfind.py        căn lưới 8×8, ngưỡng ô có quân/trống
  piece_net.py       CNN đọc quân — ONNX trên CPU, QNN trên NPU
  tracker3.py        BÁM NƯỚC ĐI — trái tim của giải pháp
  reader.py          các hàm suy luận thế cờ, tách rời camera để test được
  engine.py          bọc Stockfish
  analysis.py        phân tích thế trận
  commentary.py      sinh lời bình từng nước
  render.py          vẽ bàn cờ đối chiếu
  speaker.py         đọc to
  vision.py          tiện ích camera
  voice.py           LỆNH BẰNG GIỌNG NÓI: Whisper (NPU) → hiểu lệnh → coach
  llm.py             nối mô hình ngôn ngữ (dùng ở lớp 3 của đường giọng nói)

tools/
  coach_server.py    máy chủ web + ba luồng + toàn bộ vòng nhận diện
  npu_server.py      tiến trình phụ giữ graph QNN trên NPU
  replay_server.py   PHÁT LẠI trên dữ liệu đã lưu — công cụ tìm ra cả ba lỗi
  build_npu.sh       dựng mô hình INT8 cho NPU (chạy một lần)
  install_aibox.sh   cài dịch vụ systemd, tự chạy khi bật máy
  run_aibox.sh       đẩy code mới + khởi động lại
  install_voice.sh   cài bộ Whisper lên thiết bị (chạy một lần)
  test_voice.py      kiểm chứng cả chuỗi giọng nói, không cần micro
  train/             sinh dữ liệu và huấn luyện PieceNet

whisper/             Whisper-Small lượng tử hoá chạy trên NPU
  whisper_npu.py     điều khiển encoder/decoder qua QNN
  asr.py             log-mel, greedy search, cổng tin cậy chống bịa
  (encoder.bin, decoder.bin — 358 MB, không đưa lên GitHub)

tests/
  test_reader.py     31 phép thử, không cần camera hay mô hình
  test_rectify.py    bộ khung hình thật, sinh ảnh ghép để soi bằng mắt
  frames/            ảnh mẫu cho hai bộ thử trên

archive/             mã nguồn của các hướng đã thử trước đó, giữ để tham khảo
docs/                hướng dẫn phụ
```

## 10. Chạy thử

```bash
# LẦN ĐẦU trên AIBOX: cài dịch vụ + bật lúc khởi động máy
bash tools/install_aibox.sh

# Hàng ngày: đẩy code mới rồi khởi động lại
bash tools/run_aibox.sh

# Mở trình duyệt
http://127.0.0.1:8090
```

Kiểm chứng **không cần camera**, chạy được ngay trên máy tính thường:

```bash
python3 tests/test_reader.py                 # 31 phép thử logic bám nước đi
python3 tools/replay_server.py --audit       # chấm độ chính xác trên dữ liệu thật
python3 tools/replay_server.py --moves       # chấm khả năng bắt đúng nước
```

## 11. Bảng chẩn đoán trực tiếp

Giao diện web có một thẻ **Chẩn đoán trực tiếp** hiển thị ngay trên màn hình:

- số ô đang lệch (chính là các chấm đỏ trên bàn đối chiếu);
- tỉ lệ khung khớp hoàn toàn và tỉ lệ khung lệch ≥2 ô, tích luỹ theo phiên;
- biểu đồ 60 khung gần nhất — nhìn được lúc nào hệ thống chớm kẹt;
- thời gian từng bước: bám, nắn, đọc ô, mạng nơ-ron, cả vòng;
- **đối chứng trực tiếp giữa hai nguồn đọc**, chỉ đếm chứ không đổi hành vi.

Mọi con số đo được ngoài đời đều kiểm chứng lại được ngay tại đây, không phải mở
log ra đọc.

## 12. Kịch bản demo và phương án dự phòng

Trình diễn trực tiếp luôn có rủi ro: đèn phòng thi khác đèn ở nhà, camera bị chạm,
mạng chập chờn. Chúng tôi chuẩn bị ba lớp phòng thủ.

**Lớp 1 — chạy thật.** Cắm điện AIBOX, mở trình duyệt, đi vài nước. Bảng chẩn đoán
cho thấy số liệu chạy ngay trước mắt ban giám khảo.

**Lớp 2 — video demo quay sẵn.** Đề phòng ánh sáng phòng thi gây khó.

**Lớp 3 — phát lại không cần camera.** `python3 tools/replay_server.py` chạy lại
toàn bộ vòng nhận diện trên dữ liệu đã lưu. Kể cả khi camera hỏng hoàn toàn,
chúng tôi vẫn chứng minh được hệ thống hoạt động và vẫn đưa ra được con số.

### Khoảnh khắc nên quay vào video

1. **Rút dây laptop ra, hệ thống vẫn chạy.** Đây là cảnh đắt nhất — chứng minh
   đúng tinh thần Edge AI chỉ trong năm giây, không cần giải thích gì thêm.
2. Đi một nước, đồng hồ bấm giây cho thấy hệ thống bắt kịp trong **~0,4 giây**.
3. Cận cảnh bảng chẩn đoán: tỉ lệ khớp, thời gian từng bước, **PieceNet 9,8 ms**.
4. Một nước **ăn quân** — thứ mà ngưỡng pixel vĩnh viễn mù, phải nhờ mạng nơ-ron
   phân giải.
5. Lấy tay che một góc bàn: hệ thống **từ chối đoán** thay vì đoán bừa, rồi tự hồi
   phục khi bỏ tay ra.

## 13. Gợi ý dàn ý slide (20 phút)

Ban tổ chức khuyên ưu tiên demo và cách triển khai model, tránh giới thiệu chung
dài dòng. Dàn ý dưới đây theo đúng tinh thần đó.

| Phút | Nội dung | Ghi chú trình bày |
|---|---|---|
| 0–2 | **Mở màn bằng demo luôn**, không slide giới thiệu | Chiếu cảnh rút dây laptop mà hệ thống vẫn chạy. Nói: *"Cái laptop này chỉ là màn hình thôi ạ. Rút ra vẫn chạy."* |
| 2–4 | Bài toán: 98% mỗi ô nghe rất giỏi, nhưng cả bàn chỉ 85,2% | Một slide, hai con số, để khán giả tự thấy vấn đề |
| 4–7 | **Ý tưởng cốt lõi**: luật cờ làm bộ lọc, không phải bộ sinh | Sơ đồ 4 dòng ở mục 2. Con số: 282 ms → 10,5 ms |
| 7–12 | **Chuỗi Edge AI đầy đủ** (phần nặng ký nhất) | Sơ đồ 7 bước ở mục 5. Nhấn: 479 ms → 9,8 ms, **INT8 trùng 100% với fp32** |
| 12–16 | **Demo trực tiếp** + bảng chẩn đoán | Đi vài nước, có nước ăn quân. Chỉ vào số liệu đang chạy |
| 16–18 | Kết quả và ba lỗi tìm được nhờ đo | Bảng "trước/sau" ở mục 6 |
| 18–20 | Hỏi đáp | |

### Vài câu mở đầu cho đỡ khô

- *"Tụi em không dạy máy chơi cờ. Cờ thì Stockfish giỏi hơn tụi em nhiều rồi. Tụi
  em dạy máy **nhìn** bàn cờ."*
- *"Model đọc đúng 98% mỗi ô. Nghe giỏi đúng không ạ? Nhưng bàn cờ có 64 ô, và
  phải đúng hết cùng lúc. 98% mỗi ô ra 85% cả bàn — tức là cứ 7 khung hình thì có
  1 khung nhìn ra thế cờ sai."*
- *"Chỗ này tụi em sai. Cơ chế tự sửa lỗi mà tụi em rất tự hào ấy ạ — đo ra mới
  biết nó đang làm hệ thống **tệ đi** 3%."* (thành thật thường ăn điểm)
- Về NPU: *"CPU chạy hết 479 mili giây. NPU: 9,8. Nhanh hơn 49 lần, mà độ chính
  xác thì **trùng khít 100%**, không mất một ô nào."*

### Ba con số nên nhắc đi nhắc lại

Nếu ban giám khảo chỉ nhớ được ba điều, hãy để họ nhớ ba điều này:

1. **9,8 ms trên NPU** — nhanh hơn CPU 49 lần, độ chính xác không mất gì.
2. **0,4 giây** từ lúc người chơi đi tới lúc hệ thống nhận ra.
3. **Rút laptop ra vẫn chạy** — mọi thứ nằm trên thiết bị.

## 14. Hướng đi tiếp

**Đổi vai hai nguồn đọc.** Khi đối chứng công bằng trên cùng 1000 khung, mạng CNN
đọc "ô có quân hay không" **tốt hơn hẳn** ngưỡng pixel:

| Nguồn | Đúng cả bàn | Khung lệch ≥2 ô |
|---|---|---|
| Pixel (đang dùng) | 92,3% | 24 |
| CNN | 98,7% | 3 |
| **CNN, pixel phá hoà khi CNN lưỡng lự** | **99,4%** | **0** |

Điều này **ngược với giả định ban đầu** của chính dự án. Đáng tin vì hai lẽ: bộ dữ
liệu vốn do đường pixel lưu ra nên thiên vị cho pixel, vậy mà pixel vẫn thua; và
936/1000 khung được lưu *sau* khi mô hình được huấn luyện nên không thể rò rỉ.
Chi phí gần bằng không vì vòng nhận diện đã gọi CNN mỗi khung rồi.

Chưa áp dụng vì cần thêm thời gian chạy thật trước khi đổi một nguyên tắc nền tảng.

**Các hướng khác:** nối mô hình ngôn ngữ để bình luận tổng quan (mã đã có, chưa
đấu dây); huấn luyện giai đoạn hai cho PieceNet bằng dữ liệu thật đang tự tích luỹ;
mở rộng khả năng tự đồng bộ khi lỡ nhiều hơn hai nước.

---

## Điều chúng tôi tâm đắc nhất

Không phải con số 99,4%, mà là **cách tìm ra nó**.

Ba lỗi nghiêm trọng nhất trong hệ thống đều **vô hình trong lúc chạy thật**: chúng
chỉ hiện ra khi phát lại trên dữ liệu đã lưu kèm đáp án đúng. Lỗi lưới sát mép nấp
sau thế khai cuộc suốt 10 nước đầu. Đường tính toán 282 ms trông như "hệ thống
đang suy nghĩ". Cơ chế tự hiệu chỉnh trông như đang giúp ích trong khi thực tế nó
kéo lùi.

Và cả ba đều được sửa nhờ cùng một thói quen: **đo trước, sửa sau; không có số thì
không kết luận.** Mỗi con số trong tài liệu này đều đến từ một lần chạy thật, kể cả
những con số chứng minh chúng tôi đã sai.
