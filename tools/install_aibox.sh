#!/usr/bin/env bash
# =============================================================================
#  install_aibox.sh — cài KIT thành PHẦN MỀM TỰ CHẠY trên AIBOX 8550.
#
#  Sau khi chạy script này, AIBOX tự lo mọi thứ: cắm điện là hai dịch vụ tự lên,
#  không cần laptop, không cần adb. Laptop chỉ còn là cửa sổ xem (adb forward).
#
#      kit_npu.service     PieceNet trên NPU Hexagon (python3.12 + QNN)
#      chess_coach.service Thị giác + Stockfish + web 8090  (python3.10)
#      cả hai chạy dưới user khacthu, code ở /home/khacthu/kit
#
#  Dùng:  bash tools/install_aibox.sh          # cài + bật + chạy
#         bash tools/install_aibox.sh status   # xem trạng thái
# =============================================================================
set -euo pipefail

ADB="${ADB:-adb}"
USR="${KIT_USER:-khacthu}"
DEST="${KIT_DEST:-/home/$USR/kit}"
QSDK="${QSDK:-/opt/qcom/qairt-new/qairt/2.48.0.260626}"
ARCH=aarch64-oe-linux-gcc11.2
KIT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$KIT"

if [ "${1:-}" = "status" ]; then
  "$ADB" shell "systemctl status kit_npu chess_coach --no-pager -n 5 2>&1 | grep -E 'Loaded|Active|●'"
  exit 0
fi

echo ">> [1/5] đẩy code lên $DEST"
"$ADB" wait-for-device
"$ADB" shell "mkdir -p $DEST/tools $DEST/models $DEST/captures"
"$ADB" push chess_ai "$DEST/" | tail -1
"$ADB" push tools "$DEST/" | tail -1
[ -f models/piece_net.onnx ] && "$ADB" push models/piece_net.onnx "$DEST/models/" | tail -1

echo ">> [2/5] kiểm tra tiền đề trên AIBOX"
"$ADB" shell "set -e
  id $USR >/dev/null || { echo '!! chưa có user $USR'; exit 1; }
  id -nG $USR | grep -qw video  || echo '   !! $USR chưa ở group video — sẽ không mở được camera'
  id -nG $USR | grep -qw system || echo '   !! $USR chưa ở group system — sẽ không dùng được NPU'
  test -f $DEST/qnn_build/piece_net_q.dlc && echo '   DLC NPU: có' || echo '   DLC NPU: THIẾU (chạy tools/build_npu.sh)'
  test -x $DEST/qnnenv/bin/python && echo '   venv 3.12: có' || echo '   venv 3.12: THIẾU (chạy tools/build_npu.sh)'
  # stockfish nằm ở /usr/games, không có trong PATH mặc định của adb shell
  { command -v stockfish >/dev/null || test -x /usr/games/stockfish; } \
     && echo '   stockfish: có' || echo '   stockfish: THIẾU'
  chown -R $USR:$USR $DEST"

echo ">> [3/5] ghi systemd unit"
# RestartSec=10: HTP cần vài giây nhả context của tiến trình cũ. Restart tức thì
# thì lần khởi động sau vấp 'Failed to check DLC has cache or not' rồi systemd
# bỏ cuộc vì "start request repeated too quickly".
"$ADB" shell "cat > /etc/systemd/system/kit_npu.service <<EOF
[Unit]
Description=KIT PieceNet NPU sidecar (Hexagon HTP)
After=network.target

[Service]
Type=simple
User=$USR
WorkingDirectory=$DEST
Environment=KIT_ROOT=$DEST
Environment=LD_LIBRARY_PATH=$QSDK/lib/$ARCH
Environment=ADSP_LIBRARY_PATH=$QSDK/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;/dsp
Environment=PYTHONPATH=$QSDK/lib/python
ExecStart=$DEST/qnnenv/bin/python $DEST/tools/npu_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/chess_coach.service <<EOF
[Unit]
Description=Chess Coach Server (KIT)
After=network.target kit_npu.service
Wants=kit_npu.service

[Service]
Type=simple
User=$USR
WorkingDirectory=$DEST
Environment=KIT_ROOT=$DEST
ExecStart=/usr/bin/python3 $DEST/tools/coach_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload"

echo ">> [4/5] bật khi khởi động máy + chạy ngay"
"$ADB" shell "systemctl reset-failed kit_npu chess_coach 2>/dev/null
  systemctl enable kit_npu chess_coach >/dev/null 2>&1
  rm -f /tmp/kit_npu.sock
  systemctl restart kit_npu"
# Nạp graph QNN mất ~20s; chess_coach phải lên SAU, vì piece_net.load() chỉ chọn
# NPU khi socket đã tồn tại tại thời điểm import.
for _ in $(seq 1 15); do
  "$ADB" shell "test -S /tmp/kit_npu.sock && echo ok" | grep -q ok && break
  sleep 2
done
"$ADB" shell "systemctl restart chess_coach"
sleep 8

echo ">> [5/5] kết quả"
"$ADB" shell "systemctl is-enabled kit_npu chess_coach | tr '\n' ' '; echo '(bật lúc boot)'
  systemctl is-active kit_npu chess_coach | tr '\n' ' '; echo '(đang chạy)'
  journalctl -u chess_coach --since '-2min' --no-pager | grep 'PieceNet:' | tail -1 || echo '   (chưa thấy dòng PieceNet)'"

"$ADB" forward --remove tcp:8090 2>/dev/null || true
"$ADB" forward tcp:8090 tcp:8090
cat <<EOF

================================================================
  XONG — AIBOX tự chạy, không cần laptop nữa.
================================================================
  Xem từ laptop:  adb forward tcp:8090 tcp:8090 → http://127.0.0.1:8090
  Xem trên mạng:  http://<IP-Tailscale-AIBOX>:8090
  Log:            adb shell journalctl -u chess_coach -f
  Trạng thái:     bash tools/install_aibox.sh status
EOF
