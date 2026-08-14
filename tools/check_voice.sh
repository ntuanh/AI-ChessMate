#!/usr/bin/env bash
# =============================================================================
#  check_voice.sh — kiểm tra đường giọng nói sau khi BẬT NGUỒN. Chạy 10 giây.
#
#      bash tools/check_voice.sh          # chỉ kiểm tra
#      bash tools/check_voice.sh --thu    # kiểm tra xong thì nghe thử 4 giây
#
#  Mỗi mục hỏng đều in kèm ĐÚNG câu lệnh để sửa, vì thứ tự nguyên nhân ở đây
#  hay đánh lừa: micro bị chiếm thì `arecord` báo "Device or resource busy",
#  còn thiếu nhóm `audio` thì báo "no soundcards found" — nghe như hỏng phần
#  cứng trong khi thực ra chỉ là quyền.
# =============================================================================
PORT="${PORT:-8090}"
USR="${KIT_USER:-khacthu}"
bad=0
ok()   { printf "  \033[32mOK\033[0m   %s\n" "$1"; }
err()  { printf "  \033[31mHONG\033[0m %s\n     -> %s\n" "$1" "$2"; bad=$((bad+1)); }
warn() { printf "  \033[33mLUU Y\033[0m %s\n     -> %s\n" "$1" "$2"; }

echo "== 1. dich vu =="
for u in chess_coach kit_npu; do
  if [ "$(systemctl is-active $u)" = active ]; then ok "$u dang chay"
  else err "$u KHONG chay" "sudo systemctl start $u"; fi
done

echo "== 2. micro =="
# Bẫy số 1 sau khi bật nguồn: PulseAudio chạy --system, kẹt ở 'activating' mà
# vẫn giữ độc quyền pcmC0D0c. Đo được: arecord -> 'Device or resource busy',
# và parecord ra file 44 byte (chỉ có header WAV, 0 frame). Tức là ở trạng thái
# đó KHÔNG có đường nào thu được âm thanh.
if holder=$(fuser /dev/snd/pcmC0D0c 2>/dev/null) && [ -n "$holder" ]; then
  err "micro dang bi tien trinh khac giu ($(ps -o comm= -p $holder | tr '\n' ' '))" \
      "sudo systemctl stop pulseaudio    # xem muc 5"
else
  ok "micro ranh"
fi
if id -nG "$USR" | tr ' ' '\n' | grep -qx audio; then ok "$USR co nhom audio"
else err "$USR THIEU nhom audio" "sudo usermod -aG audio $USR && sudo systemctl restart chess_coach"; fi

echo "== 3. LLM =="
if [ "$(curl -s --max-time 5 http://127.0.0.1:8081/health | grep -c '"ok"')" = 1 ]; then
  ok "llama-server 8081 san sang"
else
  # Không phải lỗi chặn: lớp từ khoá vẫn giải quyết được "đánh cho tôi nước
  # tiếp theo" trong 0 ms. Chỉ những cách nói lạ mới cần tới LLM.
  warn "llama-server 8081 khong tra loi (cau quen thuoc VAN chay, cau la thi khong)" \
       "sudo systemctl start llamasrv"
fi

echo "== 4. Whisper trong coach_server =="
st=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/status")
case "$st" in
  "") err "coach_server khong tra loi o cong $PORT" "sudo systemctl restart chess_coach" ;;
  *'"voice": "loi"'*) err "Whisper bao loi" "sudo journalctl -u chess_coach | grep Whisper | tail -3" ;;
  *'"voice": "tat"'*) warn "Whisper con dang nap (~2 giay sau khi dich vu len)" "doi roi chay lai" ;;
  *) ok "Whisper san sang" ;;
esac

echo "== 5. sau khi bat nguon =="
if [ "$(systemctl is-enabled pulseaudio 2>/dev/null)" = enabled ]; then
  warn "pulseaudio dang 'enabled' -> no se quay lai va chiem micro moi lan bat may" \
       "sudo systemctl disable --now pulseaudio   (hoac stop tay sau moi lan bat may)"
else
  ok "pulseaudio da tat han"
fi

echo
if [ "$bad" -gt 0 ]; then echo ">> $bad muc hong, sua theo goi y ben tren."; exit 1; fi
echo ">> San sang. Mo http://127.0.0.1:$PORT roi nhan nut micro, hoac:"
echo "   curl \"http://127.0.0.1:$PORT/listen?s=4\"   # noi trong 4 giay"

if [ "${1:-}" = "--thu" ]; then
  echo
  echo ">>> NOI DI (4 giay): \"danh cho toi nuoc tiep theo\""
  curl -s --max-time 5 "http://127.0.0.1:$PORT/listen?s=4" >/dev/null
  for _ in $(seq 1 20); do
    sleep 1
    curl -s --max-time 5 "http://127.0.0.1:$PORT/status" | python3 -c '
import json,sys
s=json.load(sys.stdin)
if s["voice"] in ("xong","khong_hieu","im_lang","loi"):
    print("    nghe : %r" % (s["voice_text"] or "(khong co)"))
    print("    lenh : %s [%s]" % (s["voice_cmd"] or s["voice"], s["voice_src"]))
    print("    NUOC DI: %s   %s" % (s["hint"], s["voice_ans"] or s["voice_err"]))
    sys.exit(7)' && continue
    break
  done
fi
