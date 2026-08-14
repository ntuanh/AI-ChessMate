#!/usr/bin/env bash
# =============================================================================
#  aibox_bridge.sh — GIU CAU NOI: Tailscale -> USB(adb) -> AIBOX ssh
#
#  Duong di:
#     Team ==Tailscale==> 100.106.160.24:2222 ==socat==> 127.0.0.1:2223
#                                             ==adb(USB)==> AIBOX:22
#
#  Script chay vong lap, tu dong dung lai moi thu khi:
#     - rut cap USB roi cam lai      (adb forward bi mat -> tao lai)
#     - adb server chet / restart    (tao lai forward)
#     - socat chet                   (bat lai)
#     - tailscaled restart           (cho co IP roi bind lai)
#     - reboot laptop                (systemd user service tu chay)
#
#  Dung:
#     ./aibox_bridge.sh            -> chay vong lap (systemd goi cai nay)
#     ./aibox_bridge.sh --status   -> xem trang thai hien tai
#     ./aibox_bridge.sh --once     -> sua 1 lan roi thoat
# =============================================================================
set -uo pipefail

# ----------------------------- CAU HINH --------------------------------------
SERIAL="${AIBOX_SERIAL:-79f3af64}"   # serial USB cua AIBOX (adb devices)
TS_PORT="${AIBOX_TS_PORT:-2222}"     # cong mo tren IP Tailscale cho team
LOCAL_PORT="${AIBOX_LOCAL_PORT:-2223}" # cong localhost do adb forward tao ra
AIBOX_SSH_PORT=22                    # cong ssh tren AIBOX
EXTRA_FORWARDS=("8090:8090")         # cac forward phu: "congLaptop:congAIBOX"
INTERVAL="${AIBOX_INTERVAL:-5}"      # giay giua 2 lan kiem tra

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}"
LOG="$STATE_DIR/aibox-bridge.log"
LOCK="$STATE_DIR/aibox-bridge.lock"
mkdir -p "$STATE_DIR"

ADB="$(command -v adb || echo /usr/bin/adb)"
SOCAT="$(command -v socat || echo /usr/bin/socat)"
TAILSCALE="$(command -v tailscale || echo /usr/bin/tailscale)"

# ----------------------------- TIEN ICH --------------------------------------
LAST_MSG=""
log() {
    # chi ghi khi thong diep DOI -> tranh log rac hang nghin dong
    local msg="$*"
    [ "$msg" = "$LAST_MSG" ] && return 0
    LAST_MSG="$msg"
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$LOG"
}

log_always() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

ts_ip() { "$TAILSCALE" ip -4 2>/dev/null | head -1; }

device_ready() {
    "$ADB" devices 2>/dev/null | awk -v s="$SERIAL" '$1==s && $2=="device"{f=1} END{exit !f}'
}

forward_ok() {   # $1 = cong localhost
    "$ADB" -s "$SERIAL" forward --list 2>/dev/null | grep -q "tcp:$1\b"
}

listener_ok() {  # $1 = ip, $2 = cong
    ss -tln 2>/dev/null | grep -q "$1:$2 "
}

# ------------------------- CAC BUOC DUNG LAI ---------------------------------
ensure_forwards() {
    if ! device_ready; then
        log "[cho] AIBOX chua san sang (chua cam cap? adb chua thay $SERIAL)"
        return 1
    fi

    if ! forward_ok "$LOCAL_PORT"; then
        "$ADB" -s "$SERIAL" forward "tcp:$LOCAL_PORT" "tcp:$AIBOX_SSH_PORT" >/dev/null 2>&1 \
            && log_always "[sua] tao lai adb forward $LOCAL_PORT -> AIBOX:$AIBOX_SSH_PORT" \
            || { log "[loi] khong tao duoc adb forward $LOCAL_PORT"; return 1; }
    fi

    # forward tien loi cho chinh laptop: 127.0.0.1:TS_PORT -> AIBOX:22
    forward_ok "$TS_PORT" || \
        "$ADB" -s "$SERIAL" forward "tcp:$TS_PORT" "tcp:$AIBOX_SSH_PORT" >/dev/null 2>&1

    # cac forward phu (vd cong 8090 cho stream camera)
    local pair lp rp
    for pair in "${EXTRA_FORWARDS[@]:-}"; do
        [ -z "$pair" ] && continue
        lp="${pair%%:*}"; rp="${pair##*:}"
        forward_ok "$lp" || \
            "$ADB" -s "$SERIAL" forward "tcp:$lp" "tcp:$rp" >/dev/null 2>&1
    done
    return 0
}

ensure_socat() {
    local ip; ip="$(ts_ip)"
    if [ -z "$ip" ]; then
        log "[cho] chua co IP Tailscale (tailscaled dang khoi dong?)"
        return 1
    fi

    if listener_ok "$ip" "$TS_PORT"; then
        return 0                       # da co nguoi lang nghe roi -> khong dung vao
    fi

    "$SOCAT" "TCP-LISTEN:$TS_PORT,bind=$ip,fork,reuseaddr" \
             "TCP:127.0.0.1:$LOCAL_PORT" >>"$LOG" 2>&1 &
    sleep 1
    if listener_ok "$ip" "$TS_PORT"; then
        log_always "[sua] bat lai socat $ip:$TS_PORT -> 127.0.0.1:$LOCAL_PORT"
        return 0
    fi
    log "[loi] socat khong bind duoc $ip:$TS_PORT"
    return 1
}

reconcile() {
    local ok=0
    ensure_forwards || ok=1
    ensure_socat    || ok=1
    [ $ok -eq 0 ] && log "[ok] cau noi dang chay binh thuong"
    return $ok
}

# ------------------------------ STATUS ---------------------------------------
show_status() {
    local ip; ip="$(ts_ip)"
    echo "================= TRANG THAI CAU NOI AIBOX ================="
    printf '%-26s %s\n' "IP Tailscale:"  "${ip:-KHONG CO}"
    printf '%-26s ' "AIBOX qua USB:"
    device_ready && echo "OK ($SERIAL)" || echo "KHONG THAY"
    printf '%-26s ' "adb forward $LOCAL_PORT:"
    forward_ok "$LOCAL_PORT" && echo "OK" || echo "THIEU"
    printf '%-26s ' "socat $ip:$TS_PORT:"
    [ -n "$ip" ] && { listener_ok "$ip" "$TS_PORT" && echo "DANG NGHE" || echo "KHONG CHAY"; } || echo "-"
    echo "------------------------------------------------------------"
    echo "Team ket noi bang:  ssh -p $TS_PORT thunopro@${ip:-<ip>}"
    echo "Ban tu test bang :  ssh -p $TS_PORT thunopro@127.0.0.1"
    echo "------------------------------------------------------------"
    echo "Dang ket noi vao cong $TS_PORT:"
    ss -tn 2>/dev/null | grep "${ip:-@@@}:$TS_PORT" | awk '{print $5}' \
        | cut -d: -f1 | sort | uniq -c | sed 's/^/   /' || echo "   (khong co ai)"
    echo "============================================================"
}

# ------------------------------- MAIN ----------------------------------------
case "${1:-}" in
    --status|-s) show_status; exit 0 ;;
    --once|-o)   reconcile;   exit $? ;;
esac

# chi cho 1 ban chay cung luc
exec 9>"$LOCK"
flock -n 9 || { echo "aibox_bridge da chay roi (lock: $LOCK)"; exit 0; }

log_always "===== aibox_bridge khoi dong (serial=$SERIAL, cong=$TS_PORT) ====="
trap 'log_always "===== aibox_bridge dung ====="; exit 0' TERM INT

while true; do
    reconcile
    sleep "$INTERVAL"
done
