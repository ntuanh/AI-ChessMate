#!/bin/bash
# Đẩy code mới lên AIBOX rồi khởi động lại dịch vụ. Dùng HÀNG NGÀY khi sửa code.
#
#   bash tools/run_aibox.sh          # đẩy code + restart + mở cổng 8090
#   bash tools/run_aibox.sh log      # xem log đang chạy
#
# Lần đầu (hoặc sau khi cài lại máy) thì chạy tools/install_aibox.sh — nó dựng
# systemd unit để AIBOX tự chạy khi bật máy, không cần laptop.
#
# Phần mềm chạy TOÀN BỘ trên AIBOX; laptop chỉ là cửa sổ xem qua `adb forward`.
set -e
ADB="${ADB:-adb}"
KIT="$(cd "$(dirname "$0")/.." && pwd)"
USR="${KIT_USER:-khacthu}"
DEST="${KIT_DEST:-/home/$USR/kit}"
PORT="${PORT:-8090}"

cd "$KIT"

if [ "$1" = "log" ]; then
  "$ADB" shell "journalctl -u chess_coach -n 60 --no-pager"
  exit 0
fi

"$ADB" wait-for-device
if ! "$ADB" shell "systemctl is-enabled chess_coach 2>/dev/null" | grep -q enabled; then
  echo "!! chưa cài dịch vụ — chạy: bash tools/install_aibox.sh"
  exit 1
fi

echo ">> đẩy code lên $DEST"
"$ADB" push chess_ai "$DEST/" | tail -1
"$ADB" push tools "$DEST/" | tail -1
[ -f models/piece_net.onnx ] && "$ADB" push models/piece_net.onnx "$DEST/models/" | tail -1
"$ADB" shell "chown -R $USR:$USR $DEST"

# Sidecar NPU chỉ cần khởi động lại khi chính nó đổi; nạp graph mất ~20s nên
# đừng đụng vào nếu không cần. coach_server thì luôn restart để lấy code mới.
echo ">> khởi động lại dịch vụ"
"$ADB" shell "systemctl restart chess_coach"
sleep 6

"$ADB" forward --remove tcp:$PORT 2>/dev/null || true
"$ADB" forward tcp:$PORT tcp:$PORT

"$ADB" shell "systemctl is-active kit_npu chess_coach | tr '\n' ' '; echo '(kit_npu chess_coach)'
  journalctl -u chess_coach --since '-1min' --no-pager | grep 'PieceNet:' | tail -1"

cat <<EOF

================================================================
  MỞ TRÌNH DUYỆT:   http://127.0.0.1:$PORT
================================================================
  xem log tiếp:     bash tools/run_aibox.sh log
EOF
