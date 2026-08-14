#!/usr/bin/env bash
# =============================================================================
#  llm_tunnel.sh — DAN LLM TU MAY GPU VE TAN AIBOX
#
#  Duong di:
#     AIBOX:11434  ==adb reverse(USB)==>  laptop:11434
#                  ==ssh -L qua Tailscale==>  GPU 100.95.255.65:11434 (Ollama)
#
#  Nho vay: AIBOX KHONG can internet, Ollama KHONG phai mo ra mang
#  (van chi nghe 127.0.0.1 tren may GPU) -> an toan.
#
#  Dung:
#     ./llm_tunnel.sh            -> chay vong lap giu tunnel song
#     ./llm_tunnel.sh --status   -> xem trang thai
#     ./llm_tunnel.sh --once     -> dung 1 lan roi thoat
#     ./llm_tunnel.sh --test     -> hoi thu LLM 1 cau
# =============================================================================
set -uo pipefail

GPU_HOST="${GPU_HOST:-100.95.255.65}"
GPU_USER="${GPU_USER:-user}"
PORT="${OLLAMA_PORT:-11434}"
SERIAL="${AIBOX_SERIAL:-79f3af64}"
INTERVAL="${LLM_INTERVAL:-10}"

# Mat khau: uu tien bien moi truong, sau do file 600
PASS_FILE="${GPU_PASS_FILE:-$HOME/.config/kit/gpu_ssh_pass}"
GPU_PASS="${GPU_SSH_PASS:-}"
[ -z "$GPU_PASS" ] && [ -r "$PASS_FILE" ] && GPU_PASS="$(head -1 "$PASS_FILE")"

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}"
LOG="$STATE_DIR/llm-tunnel.log"
LOCK="$STATE_DIR/llm-tunnel.lock"
mkdir -p "$STATE_DIR"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ExitOnForwardFailure=yes -o ServerAliveInterval=15
          -o ServerAliveCountMax=3 -o ConnectTimeout=10)

LAST_MSG=""
log() { local m="$*"; [ "$m" = "$LAST_MSG" ] && return 0; LAST_MSG="$m"
        printf '%s  %s\n' "$(date '+%F %T')" "$m" | tee -a "$LOG"; }
log_always() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

tunnel_up()  { ss -tln 2>/dev/null | grep -q "127.0.0.1:$PORT "; }
ollama_ok()  { curl -s -m 4 "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1; }
reverse_ok() { adb -s "$SERIAL" reverse --list 2>/dev/null | grep -q "tcp:$PORT"; }
device_ok()  { adb devices 2>/dev/null | awk -v s="$SERIAL" '$1==s && $2=="device"{f=1} END{exit !f}'; }

ensure_tunnel() {
    ollama_ok && return 0
    pkill -f "ssh.*-L $PORT:127.0.0.1:$PORT.*$GPU_HOST" 2>/dev/null
    sleep 0.5
    if [ -z "$GPU_PASS" ]; then
        # khong co mat khau -> thu khoa SSH
        ssh -N -L "$PORT:127.0.0.1:$PORT" "${SSH_OPTS[@]}" \
            "$GPU_USER@$GPU_HOST" >>"$LOG" 2>&1 &
    else
        sshpass -p "$GPU_PASS" ssh -N -L "$PORT:127.0.0.1:$PORT" "${SSH_OPTS[@]}" \
            "$GPU_USER@$GPU_HOST" >>"$LOG" 2>&1 &
    fi
    for _ in $(seq 1 10); do
        sleep 1
        ollama_ok && { log_always "[sua] dung lai tunnel laptop:$PORT -> $GPU_HOST (Ollama)"; return 0; }
    done
    log "[loi] khong dung duoc tunnel toi $GPU_HOST:$PORT"
    return 1
}

ensure_reverse() {
    device_ok || { log "[cho] chua thay AIBOX qua USB"; return 1; }
    reverse_ok && return 0
    if adb -s "$SERIAL" reverse "tcp:$PORT" "tcp:$PORT" >/dev/null 2>&1; then
        log_always "[sua] dung lai adb reverse AIBOX:$PORT -> laptop:$PORT"
        return 0
    fi
    log "[loi] khong tao duoc adb reverse $PORT"
    return 1
}

reconcile() {
    local ok=0
    ensure_tunnel  || ok=1
    ensure_reverse || ok=1
    [ $ok -eq 0 ] && log "[ok] LLM da san sang cho AIBOX"
    return $ok
}

show_status() {
    echo "=============== TRANG THAI LLM CHO AIBOX ==============="
    printf '%-30s ' "Tunnel laptop:$PORT:"; tunnel_up && echo "DANG MO" || echo "KHONG CO"
    printf '%-30s ' "Ollama tra loi:";      ollama_ok && echo "OK"      || echo "KHONG"
    printf '%-30s ' "AIBOX qua USB:";       device_ok && echo "OK"      || echo "KHONG THAY"
    printf '%-30s ' "adb reverse $PORT:";   reverse_ok && echo "OK"     || echo "THIEU"
    echo "-------------------------------------------------------"
    if ollama_ok; then
        echo "Model co san tren may GPU:"
        curl -s -m 5 "http://127.0.0.1:$PORT/api/tags" \
          | python3 -c "import json,sys
d=json.load(sys.stdin)
for m in d.get('models',[]):
    print('   %-28s %.1f GB' % (m.get('name'), m.get('size',0)/1e9))" 2>/dev/null
    fi
    echo "-------------------------------------------------------"
    echo "Tren AIBOX chay:  OLLAMA_URL=http://127.0.0.1:$PORT python3 -m chess_ai.main ..."
    echo "======================================================="
}

run_test() {
    echo "--- Hoi thu LLM tu LAPTOP ---"
    curl -s -m 120 "http://127.0.0.1:$PORT/api/chat" -H 'Content-Type: application/json' \
      -d '{"model":"'"${OLLAMA_MODEL:-qwen2.5:14b}"'","stream":false,"messages":[{"role":"user","content":"Chao ban, tra loi ngan gon bang tieng Viet: ban la ai?"}]}' \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('message',{}).get('content','(khong co tra loi)'))" 2>&1
    echo
    echo "--- Hoi thu TU AIBOX (qua adb reverse) ---"
    adb -s "$SERIAL" shell "curl -s -m 120 http://127.0.0.1:$PORT/api/tags | head -c 200" 2>&1
    echo
}

case "${1:-}" in
    --status|-s) show_status; exit 0 ;;
    --once|-o)   reconcile;   exit $? ;;
    --test|-t)   run_test;    exit 0 ;;
esac

exec 9>"$LOCK"
flock -n 9 || { echo "llm_tunnel da chay roi"; exit 0; }
log_always "===== llm_tunnel khoi dong (GPU=$GPU_HOST cong=$PORT) ====="
trap 'pkill -f "ssh.*-L $PORT:127.0.0.1:$PORT.*$GPU_HOST" 2>/dev/null; log_always "===== llm_tunnel dung ====="; exit 0' TERM INT

while true; do
    reconcile
    sleep "$INTERVAL"
done
