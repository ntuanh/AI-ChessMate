#!/bin/bash
# Đẩy code + model lên AIBOX, chạy web HLV cờ ở đó, rồi MỞ CỔNG VỀ MÁY NÀY.
# Dùng:  bash tools/run_aibox.sh            (tự dò camera)
#        KIT_CAM=0 bash tools/run_aibox.sh  (chỉ định camera)
#        bash tools/run_aibox.sh log        (chỉ xem log, không deploy lại)
#
# Model chạy TRÊN AIBOX; máy này chỉ là cửa sổ xem qua `adb forward`, nên mở
# http://127.0.0.1:8090 trên laptop là thấy đúng giao diện đang chạy ở AIBOX.
set -e
ADB="${ADB:-adb}"
KIT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/data/kit
PORT="${PORT:-8090}"

cd "$KIT"

if [ "$1" = "log" ]; then
  "$ADB" shell "tail -n 60 $DEST/coach.log"
  exit 0
fi

echo ">> chờ AIBOX (cắm USB, bật adb)..."
"$ADB" wait-for-device
echo ">> thiết bị: $("$ADB" devices | sed -n 2p)"

echo ">> đẩy code + model"
"$ADB" shell "mkdir -p $DEST/models $DEST/tools $DEST/captures"
"$ADB" push chess_ai "$DEST/" | tail -1
"$ADB" push tools/coach_server.py "$DEST/tools/" | tail -1
if [ -f models/piece_net.onnx ]; then
  "$ADB" push models/piece_net.onnx "$DEST/models/" | tail -1
else
  echo "   (chưa có models/piece_net.onnx — sẽ chỉ chạy đường occupancy)"
fi

echo ">> kiểm tra thư viện trên AIBOX"
"$ADB" shell "cd $DEST && python3 - <<'PY'
for m in ('numpy','cv2','chess','onnxruntime'):
    try:
        mod=__import__(m); print(' ', m, getattr(mod,'__version__','ok'))
    except Exception:
        print(' ', m, 'THIẾU')
PY"

# Camera: C270 có thể ra video0/1/2 tuỳ thứ tự nhận thiết bị.
CAM="${KIT_CAM:-}"
if [ -z "$CAM" ]; then
  echo ">> dò camera"
  CAM="$("$ADB" shell "cd $DEST && python3 - <<'PY'
import cv2
for i in (2,0,1,3,4):
    c=cv2.VideoCapture(i)
    ok,_=c.read(); c.release()
    if ok:
        print(i); break
else:
    print('')
PY" | tr -d '\r\n ')"
  [ -z "$CAM" ] && { echo "!! không mở được camera nào — kiểm tra C270 đã cắm vào AIBOX chưa"; exit 1; }
fi
echo ">> dùng /dev/video$CAM"

echo ">> khởi động lại server"
"$ADB" shell "pkill -f coach_server.py 2>/dev/null; sleep 1; true"
"$ADB" shell "cd $DEST && KIT_CAM=$CAM nohup python3 tools/coach_server.py > $DEST/coach.log 2>&1 &"
sleep 6

echo ">> mở cổng $PORT về máy này"
"$ADB" forward --remove tcp:$PORT 2>/dev/null || true
"$ADB" forward tcp:$PORT tcp:$PORT

echo ">> log khởi động:"
"$ADB" shell "tail -n 20 $DEST/coach.log" || true

cat <<EOF

================================================================
  MỞ TRÌNH DUYỆT:   http://127.0.0.1:$PORT
================================================================
  xem log tiếp:     bash tools/run_aibox.sh log
  đổi camera:       KIT_CAM=0 bash tools/run_aibox.sh
EOF
