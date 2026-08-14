#!/bin/bash
# Đẩy code chess_ai từ laptop sang AIBOX (qua USB/adb) và (tùy chọn) chạy thử.
# Dùng: bash tools/deploy.sh [simulate|main]
set -e
ADB="${ADB:-$(command -v adb || echo /usr/bin/adb)}"
KIT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${KIT_DEST:-/home/khacthu/kit}"

echo ">> push $KIT/chess_ai -> AIBOX:$DEST"
"$ADB" shell "mkdir -p $DEST"
"$ADB" push "$KIT/chess_ai" "$DEST/" | tail -1

case "$1" in
  simulate) "$ADB" shell "cd $DEST && python3 -m chess_ai.simulate --plies 40" ;;
  main)     "$ADB" shell "cd $DEST && python3 -m chess_ai.main --no-speak" ;;
  *)        echo ">> Xong. Chạy: $ADB shell \"cd $DEST && python3 -m chess_ai.simulate\"" ;;
esac
