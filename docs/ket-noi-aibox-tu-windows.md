# 🪟 Hướng dẫn SSH vào AIBOX (cho người dùng Windows)

> Mục tiêu: từ máy Windows bất kỳ (mạng nào cũng được) SSH vào AIBOX qua Tailscale.
> Cần làm 3 phần: **(A) Vào tailnet** → **(B) Cài Tailscale** → **(C) SSH vào**.

**Trước khi bắt đầu** (người quản trị = chủ laptop phải bảo đảm):
- Đang chạy `bash tools/aibox_ssh_gateway.sh` trên laptop (cổng đang mở).
- Đã **mời email của bạn** vào tailnet (bước A dưới).
- Thông tin đăng nhập AIBOX: user **`thunopro`**, mật khẩu **`Thunopro@2026`**.
- Địa chỉ: **`100.106.160.24`** cổng **`2222`**.

---

## A. Nhận lời mời vào tailnet (làm 1 lần)
1. Mở **email** (đúng email đã báo cho người quản trị) → tìm thư mời từ **Tailscale**.
2. Bấm nút **“Accept invitation”** trong email.
3. Trình duyệt mở ra → **đăng nhập** bằng Google / Microsoft / GitHub (chọn cái khớp với email đó).
4. Xong → bạn đã là thành viên tailnet. (Chưa cần làm gì thêm trên web.)

> Nếu chưa nhận được email mời: báo người quản trị mời lại đúng địa chỉ email của bạn.

---

## B. Cài Tailscale trên Windows (làm 1 lần)
1. Vào **https://tailscale.com/download/windows** → tải **Tailscale for Windows**.
2. Chạy file `.exe` vừa tải → bấm **Install** → chờ xong.
3. Sau khi cài, có **biểu tượng Tailscale** ở khay đồng hồ (góc phải dưới, gần loa/wifi).
4. Bấm vào biểu tượng đó → **Log in...** → trình duyệt mở → **đăng nhập cùng tài khoản** đã dùng ở bước A.
5. Xong: biểu tượng Tailscale chuyển sang **Connected**. Máy bạn giờ đã ở trong mạng riêng.

**Kiểm tra nhanh** (không bắt buộc): mở **PowerShell**, gõ:
```powershell
tailscale status
```
Thấy dòng có `100.106.160.24 ... luongminhthu-thinkpad-x1-carbon-5th` là ổn.

---

## C. SSH vào AIBOX
Windows 10/11 có sẵn lệnh `ssh`.

1. Mở **PowerShell** (bấm nút Start → gõ `powershell` → Enter).
2. Gõ lệnh (copy nguyên dòng):
   ```powershell
   ssh -p 2222 thunopro@100.106.160.24
   ```
3. Lần đầu nó hỏi:
   ```
   Are you sure you want to continue connecting (yes/no/[fingerprint])?
   ```
   → gõ **yes** rồi Enter.
4. Nó hỏi **password** → gõ **`Thunopro@2026`** rồi Enter.
   *(Lưu ý: khi gõ mật khẩu màn hình KHÔNG hiện gì cả — cứ gõ rồi Enter.)*
5. Thấy dấu nhắc đổi thành `thunopro@rov-test:~$` → **bạn đã vào AIBOX!** 🎉

Thoát: gõ `exit`.

---

## 🧩 Nếu bị lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| `ssh: command not found` / không nhận lệnh ssh | Vào **Settings → Apps → Optional features → Add** → cài **OpenSSH Client**. Hoặc dùng **PuTTY** (xem dưới). |
| `Connection timed out` / `Connection refused` | (1) Kiểm tra Tailscale đang **Connected**. (2) Báo quản trị xem `aibox_ssh_gateway.sh` còn chạy + AIBOX còn cắm USB không. |
| `Permission denied` | Sai mật khẩu — đúng phải là `Thunopro@2026` (chú ý chữ hoa T, ký tự @). |
| Không thấy `100.106.160.24` trong `tailscale status` | Bạn chưa vào tailnet — làm lại phần A (nhận lời mời). |

### Cách khác: dùng PuTTY (giao diện, không cần gõ lệnh)
1. Tải **PuTTY** tại https://www.putty.org → cài.
2. Mở PuTTY:
   - **Host Name**: `100.106.160.24`
   - **Port**: `2222`
   - Bấm **Open**.
3. Hỏi security alert → bấm **Accept**.
4. `login as:` → gõ `thunopro` → Enter.
5. `password:` → gõ `Thunopro@2026` → Enter. Vào được là xong.

---

## Tóm tắt 1 dòng
> Cài Tailscale + đăng nhập → `ssh -p 2222 thunopro@100.106.160.24` → mật khẩu `Thunopro@2026`.
