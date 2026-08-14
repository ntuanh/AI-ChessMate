# Nội dung slide — KIT (20 phút)

Bản thảo từng slide. Cột **Nói** là lời thoại gợi ý, cứ sửa cho hợp giọng bạn.
Nguyên tắc: **chữ trên slide ít thôi, số to, demo nhiều**.

Ban tổ chức đã dặn: ưu tiên demo, cách triển khai model lên thiết bị, và kết quả.
Tránh giới thiệu chung dài dòng. Dàn bài dưới đây bám đúng lời dặn đó.

---

## Slide 1 — Mở màn (0:00–0:30)

**Trên slide:** chỉ một dòng chữ to
> ## KIT
> ### Dạy máy **nhìn** bàn cờ
> *Chạy trọn trên AIBOX 8550*

**Nói:**
> "Chào thầy cô và các bạn. Nhóm em làm một huấn luyện viên cờ vua. Nhưng em xin
> nói ngay: tụi em **không** dạy máy chơi cờ — cờ thì Stockfish giỏi hơn tụi em
> nhiều rồi ạ. Tụi em dạy máy **nhìn**."

---

## Slide 2 — Demo trước, giới thiệu sau (0:30–2:00)

**Trên slide:** video, không chữ.

**Nội dung video:** camera soi bàn cờ → đi một nước → hệ thống bắt được → gợi ý
nước đi hiện lên. Rồi **rút dây laptop ra** — hệ thống vẫn chạy.

**Nói:**
> "Đây là sản phẩm đang chạy ạ. Camera soi vào màn hình, em đi một nước… máy bắt
> được sau khoảng 0,4 giây và gợi ý nước tiếp theo.
>
> Và bây giờ em rút cái laptop ra. *(rút)* …Vẫn chạy ạ. Vì cái laptop này chỉ là
> **màn hình** thôi. Toàn bộ thị giác, mô hình AI, và cả engine cờ đều nằm trong
> chiếc hộp AIBOX này."

> 💡 *Đây là 90 giây quan trọng nhất của cả bài. Nó chứng minh "Edge AI thật" mà
> không cần một chữ giải thích nào.*

---

## Slide 3 — Bài toán (2:00–4:00)

**Trên slide:** hai con số thật to, cạnh nhau

> # 98,07%
> đọc đúng **mỗi ô**
>
> # 85,2%
> đọc đúng **cả bàn**

**Nói:**
> "Model của tụi em đọc đúng 98% mỗi ô. Nghe giỏi đúng không ạ?
>
> Nhưng bàn cờ có **64 ô**, và phải đúng **hết cùng lúc** thì thế cờ mới đúng.
> 98% mỗi ô ra 85% cả bàn. Tức là cứ **7 khung hình thì có 1 khung nhìn ra thế cờ
> sai**. Không thể xây một huấn luyện viên trên nền đó được ạ.
>
> Nên bài toán thật của tụi em không phải là 'nhận diện chính xác hơn'. Mà là:
> **làm sao đúng cả bàn, liên tục, trong suốt cả ván.**"

---

## Slide 4 — Ý tưởng cốt lõi (4:00–7:00)

**Trên slide:** sơ đồ 4 dòng

```
S  = đọc thẳng 64 ô  →  {trống, trắng, đen}
Δ  = S khác THẾ CỜ ĐANG GIỮ ở ô nào
Δ  ổn định 3 khung  →  tra ra (from, to)
                    →  LỌC bằng luật cờ  →  chốt
```

Kèm bảng nhỏ:

| | Cách thường | **Cách của tụi em** |
|---|---|---|
| Thời gian | 282 ms | **10,5 ms** |
| Chốt ngay | 91,6% | **96,4%** |

**Nói:**
> "Cách thường làm là **sinh** ra mọi nước đi hợp lệ, rồi dựng lại bàn cờ cho từng
> nước xem cái nào khớp ảnh. Giữa ván có khoảng 30 nước, tính hai nước liên tiếp
> là 900 thế cờ phải dựng — 282 mili giây, mà vẫn chết cứng nếu lỡ mất nhiều nước.
>
> Tụi em làm ngược lại: **đọc thẳng bàn cờ, xem nó khác thế cờ đang giữ ở ô nào,
> rồi mới dùng luật cờ để LỌC.** Luật cờ vua từ vai trò *máy sinh* chuyển thành
> *máy lọc*. 282 mili giây xuống còn 10.
>
> Và một điểm nữa em rất thích: tụi em neo vào **thế cờ**, không neo vào khung hình
> trước. Thế cờ chỉ đổi khi chốt được một nước hợp lệ. Nên đọc nhầm một khung chỉ
> làm **chậm**, chứ không làm **lệch**. Nếu so khung với khung thì sai một lần là
> sai vĩnh viễn ạ."

---

## Slide 5 — Chuỗi Edge AI (7:00–9:00)

**Trên slide:** sơ đồ 7 bước

```
[1] Tự sinh dữ liệu → [2] Train PyTorch → [3] ONNX → [4] DLC
                                                       ↓
[7] Ứng dụng web ← [6] systemd ← [5] INT8 + đối chiếu độ chính xác
```

**Nói:**
> "Đây là toàn bộ đường đi của model, từ lúc chưa có gì đến lúc chạy trên NPU.
>
> Bước một: tụi em **không có dữ liệu**. Không ai đi chụp sẵn ảnh camera soi màn
> hình đang chơi cờ cả. Nên tụi em tự sinh — mô phỏng đúng cái nhiễu của cảnh quay
> màn hình: mờ, moiré, loá, lệch lưới, nén JPEG, cả cái highlight nước đi của
> Lichess nữa ạ.
>
> Một chi tiết tụi em khá tâm đắc: mỗi mẫu được dựng trên khung **3×3 ô** rồi mới
> cắt ô giữa. Vì ngoài đời lưới luôn lệch một chút, ô cắt ra sẽ dính mép quân bên
> cạnh — nên model phải học cách phớt lờ chúng ngay từ đầu."

---

## Slide 6 — NPU: con số đắt nhất bài (9:00–12:00)

**Trên slide:** ba con số

> # 479 ms → 9,8 ms
> **nhanh hơn 49 lần**
>
> ## Độ chính xác: **trùng 100%**
> INT8 trên NPU == fp32 trên CPU

**Nói:**
> "PieceNet tốn 21 GFLOP cho mỗi bàn cờ. Sáu nhân CPU của AIBOX kéo được khoảng 50
> GFLOPS, tức **479 mili giây**. Quá chậm để chạy mỗi khung hình — và đó chính là
> lý do lúc đầu cả hệ thống của tụi em phải né mạng nơ-ron ra.
>
> Đưa xuống **NPU Hexagon HTP**, lượng tử hoá INT8: **9,8 mili giây**. Nhanh hơn
> 49 lần.
>
> Nhưng lượng tử hoá thì thường mất độ chính xác. Nên tụi em hiệu chuẩn bằng 100
> khung thật, rồi kiểm định trên **100 khung khác** mà nó chưa từng thấy — 6400 ô.
> Kết quả: **NPU INT8 cho ra kết quả trùng khít 100% với CPU fp32.** Không sai một
> ô nào ạ.
>
> Tức là tụi em đổi tốc độ lấy… **không gì cả**."

**Nếu còn thời gian, kể một cái bẫy:**
> "À có một chỗ tụi em mất mấy tiếng ạ. Lúc đầu chạy trên NPU mà nó tốn **2130
> mili giây** — chậm hơn cả CPU. Hoá ra mỗi lần gọi nó **biên dịch lại graph**.
> Phải setup một lần rồi giữ nguyên. Cái bẫy này khó chịu ở chỗ mọi thứ vẫn chạy
> **đúng**, chỉ là chậm khủng khiếp thôi."

---

## Slide 7 — Demo trực tiếp (12:00–16:00)

**Trên slide:** không cần slide, chuyển sang màn hình thật.

**Trình tự demo:**
1. Mở `http://127.0.0.1:8090`, cho thấy camera và bàn cờ đối chiếu.
2. Đi một nước thường → máy bắt được.
3. **Đi một nước ăn quân** → *"Chỗ này pixel chịu chết ạ, vì ô đích vốn đã có quân
   rồi. Phải nhờ mạng nơ-ron đọc màu mới phân biệt được."*
4. **Lấy tay che một góc bàn** → *"Nó không đoán bừa ạ. Nó từ chối, chờ thêm bằng
   chứng."* Bỏ tay ra → tự hồi phục.
5. Chỉ vào **bảng chẩn đoán**: tỉ lệ khớp, PieceNet 9,8 ms, thời gian mỗi vòng.

> ⚠️ **Dự phòng:** nếu ánh sáng phòng thi gây khó → chuyển sang video quay sẵn.
> Nếu camera hỏng hẳn → chạy `python3 tools/replay_server.py`, phát lại toàn bộ
> trên dữ liệu đã lưu, vẫn ra số liệu thật.

---

## Slide 8 — Kết quả và ba lỗi tự tìm ra (16:00–18:00)

**Trên slide:** bảng trước/sau

| | Trước | Sau |
|---|---|---|
| Nhận đúng cả bàn | 86,9% | **92,3%** |
| Khung phải chờ thêm | 72 | **24** |
| Ở nước 10–14 | 70,7% | **82,9%** |

**Nói:**
> "Phần này em xin kể thật ạ. Tụi em dựng một bộ **phát lại**: chạy đúng vòng nhận
> diện trên 1000 khung hình thật đã lưu kèm đáp án đúng. Nhờ nó mà tìm ra ba lỗi
> mà **chạy thật không bao giờ thấy được**.
>
> Lỗi thứ nhất: lưới cắt ô sát mép bàn quá, còn dính 14 pixel nền trắng. Nhưng cột
> đó ở thế khai cuộc **luôn có quân**, nên lỗi bị che suốt 10 nước đầu. Đến nước
> 10–15 quân mới đi khỏi, ô trống ra, lỗi mới hiện. Đúng lúc người dùng kêu 'chơi
> tới giữa ván là nó nhận kém hẳn'.
>
> Lỗi thứ ba là cái em nhớ nhất. Tụi em có một cơ chế **tự hiệu chỉnh** — nghe rất
> hay. Đo ra mới biết nó đang làm hệ thống **tệ đi 3%** ạ. Lý do: điều kiện cho
> phép nó chạy đặt quá chặt, nên khi hệ thống lệch mãn tính thì nó **không bao giờ
> được kích hoạt** — đúng lúc cần nó nhất.
>
> Bài học tụi em rút ra: **một điều kiện đặt quá chặt không làm hệ thống an toàn
> hơn — nó làm cơ chế bảo vệ tự vô hiệu hoá, trong im lặng.**"

---

## Slide 9 — Chốt (18:00–18:30)

**Trên slide:** ba dòng

> # 9,8 ms
> mạng nơ-ron trên NPU — nhanh hơn CPU 49 lần, không mất độ chính xác
>
> # 0,4 giây
> từ lúc người chơi đi tới lúc máy nhận ra
>
> # Rút dây vẫn chạy
> mọi thứ nằm trên thiết bị

**Nói:**
> "Nếu thầy cô chỉ nhớ ba điều về bài của tụi em, em mong là ba điều này ạ.
>
> Và điều tụi em tâm đắc nhất không phải con số nào cả — mà là cái thói quen:
> **đo trước, sửa sau. Không có số thì không kết luận.** Mọi con số trong bài này
> đều từ một lần chạy thật, kể cả những con số chứng minh tụi em đã sai.
>
> Em xin hết ạ."

---

## Slide 10 — Hỏi đáp (18:30–20:00)

**Trên slide:** ảnh sản phẩm + link GitHub.

### Chuẩn bị sẵn cho các câu hay hỏi

**"Sao không dùng camera chụp bàn cờ thật?"**
> "Được ạ, nhưng bàn thật thêm bài toán bóng đổ và quân bị che khuất. Tụi em chọn
> bàn 2D trên màn hình để tập trung giải cho xong bài toán **bám nước đi liên tục**
> trước. Phần thị giác thì `rectify` đã xử lý được góc nghiêng rồi ạ."

**"Model bao nhiêu tham số? Có nhỏ quá không?"**
> "437 nghìn tham số ạ. Nhỏ là **cố ý** — vì nó phải chạy trên mọi khung hình.
> Và độ chính xác 99,98% cho thấy nhỏ vậy là đủ cho bài toán này."

**"Nếu NPU hỏng thì sao?"**
> "Có đường lùi ạ. Hệ thống tự quay về ONNX trên CPU, chậm hơn nhưng vẫn chạy.
> Và ngay cả khi không có model, đường đọc bằng pixel vẫn hoạt động."

**"Làm sao biết 99,98% không phải do học thuộc dữ liệu?"**
> "Tụi em kiểm định trên tập **held-out** riêng ạ. Với phần đối chứng ở cuối bài,
> 936 trên 1000 khung được lưu **sau** khi model đã train xong, nên không thể rò
> rỉ được."

**"Còn gì chưa làm?"**
> "Có ạ, và em xin nói thẳng: tụi em vừa đo ra rằng **mạng CNN đọc ô có quân hay
> không còn tốt hơn cả đường pixel** — 99,4% so với 92,3%. Điều này **ngược với
> giả định ban đầu** của chính tụi em. Tụi em chưa đổi, vì muốn chạy thật thêm
> trước khi thay một nguyên tắc nền tảng ạ."

---

## Danh sách ảnh cần chụp

Chụp trong lúc chơi thật:

- [ ] Toàn cảnh: AIBOX + camera + màn hình laptop đang chơi cờ
- [ ] Cận cảnh giao diện web: bàn đối chiếu + gợi ý nước đi
- [ ] Cận cảnh **bảng chẩn đoán** (số liệu đang chạy — quan trọng)
- [ ] Khoảnh khắc **rút dây laptop** mà hệ thống vẫn chạy
- [ ] Ảnh warp 512×512 có lưới ô, để minh hoạ bước cắt ô
- [ ] Ảnh chấm đỏ khi lệch, để minh hoạ phần chẩn đoán

## Nhắc cuối

- Đặt đồng hồ. **Demo 4 phút là ít nhất** — đừng để bị cắt vì slide dài.
- Mở sẵn `http://127.0.0.1:8090` **trước khi lên**, đừng canh camera trước mặt
  ban giám khảo.
- Video demo tải sẵn về máy, đừng phụ thuộc mạng phòng thi.
