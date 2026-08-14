#!/usr/bin/env bash
# =============================================================================
#  install_voice.sh — bật đường GIỌNG NÓI cho coach_server. Chạy MỘT LẦN.
#
#  Chép bộ Whisper-Small-Quantized (chạy trên NPU Hexagon) vào KIT_ROOT/whisper
#  và cấp cho user chạy dịch vụ đúng hai nhóm nó cần.
#
#  Chạy TRÊN BOARD bằng một user có sudo (vd tuananh):
#      bash tools/install_voice.sh
# =============================================================================
set -euo pipefail

SRC="${WHISPER_SRC:-/home/tuananh/whisper}"
USR="${KIT_USER:-khacthu}"
DEST="${KIT_DEST:-/home/$USR/kit}/whisper"
SUDO="${SUDO:-sudo}"

# Đúng những file worker cần lúc chạy. Không chép cả thư mục: enc_out/ và các
# file .raw trung gian nặng vài trăm MB mà không dùng đến.
FILES=(whisper_worker whisper_npu.py asr.py encoder.bin decoder.bin
       metadata.json vocab.json added_tokens.json)

echo ">> [1/4] chép bộ Whisper $SRC -> $DEST"
$SUDO mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  test -e "$SRC/$f" || { echo "!! thiếu $SRC/$f"; exit 1; }
  $SUDO cp -f "$SRC/$f" "$DEST/$f"
done
$SUDO chown -R "$USR:$USR" "$DEST"
$SUDO chmod +x "$DEST/whisper_worker"
du -sh "$DEST"

# Hai nhóm này là hai cái bẫy đã tốn thời gian một lần rồi, và cả hai đều báo
# lỗi CHỆCH HƯỚNG:
#   * thiếu `audio` -> `arecord -l` nói "no soundcards found" (nghe như không có
#     phần cứng) trong khi /proc/asound/cards liệt kê card rõ ràng;
#   * thiếu `system` -> không mở được /dev/adsprpc-smd, tức NPU câm.
echo ">> [2/4] nhóm audio + system cho $USR"
for g in audio system; do
  id -nG "$USR" | tr ' ' '\n' | grep -qx "$g" || $SUDO usermod -aG "$g" "$USR"
done
echo "   $USR: $(id -nG "$USR")"

echo ">> [3/4] kiểm tra micro"
# PulseAudio chạy --system giữ độc quyền pcmC0D0c rồi kẹt ở 'activating start';
# ở trạng thái đó arecord báo 'Device or resource busy' và pactl cũng timeout.
if fuser -s /dev/snd/pcmC0D0c 2>/dev/null; then
  echo "   !! có tiến trình đang giữ micro:"; $SUDO fuser -v /dev/snd/pcmC0D0c || true
  echo "   -> nếu là pulseaudio: sudo systemctl stop pulseaudio"
else
  echo "   micro rảnh"
fi

echo ">> [4/4] khởi động lại dịch vụ"
$SUDO systemctl restart chess_coach
sleep 8
$SUDO journalctl -u chess_coach --since '-30s' --no-pager | grep -E 'Whisper|LLM|PieceNet' || true

cat <<'EOF'

XONG. Mở http://127.0.0.1:8090 -> thẻ "Ra lệnh bằng giọng nói" -> nhấn rồi nói:
    "đánh cho tôi nước tiếp theo"

Nhóm mới chỉ có hiệu lực với tiến trình khởi động SAU khi usermod chạy — dịch vụ
vừa restart ở trên nên đã có. Kiểm tra riêng lớp hiểu lệnh, không cần micro:
    python3 -m chess_ai.voice "đánh cho tôi nước tiếp theo"
EOF
