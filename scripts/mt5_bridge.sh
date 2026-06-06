#!/usr/bin/env bash
# =============================================================================
# MT5 Bridge (Arch Linux + Wine) — robust, idempotent, headless
# =============================================================================
# One command brings the MT5 API up on Linux:
#
#   scripts/mt5_bridge.sh up        # ensure Xvfb + rpyc server are running,
#                                   # then verify a live MT5 login (auto)
#   scripts/mt5_bridge.sh down      # stop server + Xvfb (+ stray terminal)
#   scripts/mt5_bridge.sh status    # show component + login status
#   scripts/mt5_bridge.sh restart
#   scripts/mt5_bridge.sh login-gui # first-time GUI login (rarely needed)
#   scripts/mt5_bridge.sh install-service  # install systemd --user unit
#
# Design:
#   - The rpyc server runs under Wine Python and exposes the MetaTrader5 API.
#   - The bot (or this script's healthcheck) calls initialize(login,...) which
#     auto-launches the terminal headless under Xvfb and logs in. No GUI needed
#     after the account has been logged in once.
#   - Idempotent: re-running `up` only starts what is missing.
#   - Readiness is health-checked (socket + rpyc), not fixed sleeps.
#
# Config: auto-sourced from the project .env (MT5_LOGIN/PASSWORD/SERVER,
#         MT5_BRIDGE_PORT, MT5_WIN_PATH). Override via env vars.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Load .env (robust: only KEY=value lines, supports quoted values) --------
if [ -f "$PROJECT_DIR/.env" ]; then
    while IFS= read -r _line || [ -n "$_line" ]; do
        case "$_line" in
            ''|\#*) continue ;;                      # skip blank / comment
            [A-Za-z_]*=*) eval "export $_line" 2>/dev/null || true ;;
        esac
    done < "$PROJECT_DIR/.env"
fi

PORT="${MT5_BRIDGE_PORT:-18812}"
DISP="${MT5_BRIDGE_DISPLAY:-:99}"
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine}"
export WINEDEBUG="${WINEDEBUG:--all}"

WINPY="$WINEPREFIX/drive_c/users/$USER/AppData/Local/Programs/Python/Python39/python.exe"
MT5_TERM="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
WIN_PATH="${MT5_WIN_PATH:-C:\\Program Files\\MetaTrader 5\\terminal64.exe}"

RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/xaubot-mt5"
mkdir -p "$RUN_DIR"
XVFB_PID="$RUN_DIR/xvfb.pid"
SRV_PID="$RUN_DIR/server.pid"
LOG="$RUN_DIR/bridge.log"

log()  { echo "[mt5-bridge] $*"; }
err()  { echo "[mt5-bridge] ERROR: $*" >&2; }

_check_prereq() {
    local ok=1
    command -v wine >/dev/null || { err "wine not found"; ok=0; }
    command -v Xvfb >/dev/null || { err "Xvfb not found (pacman -S xorg-server-xvfb)"; ok=0; }
    [ -f "$WINPY" ]    || { err "Wine Python missing: $WINPY"; ok=0; }
    [ -f "$MT5_TERM" ] || { err "MT5 terminal missing: $MT5_TERM"; ok=0; }
    python3 -c "import mt5linux" 2>/dev/null || { err "linux pkg 'mt5linux' missing (pip install mt5linux)"; ok=0; }
    [ "$ok" = 1 ] || exit 1
}

_alive()      { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
_port_open()  { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && exec 3>&- ; }

# Wait until the rpyc port accepts connections (timeout seconds)
_wait_port() {
    local t="${1:-20}"
    for _ in $(seq 1 "$t"); do _port_open && return 0; sleep 1; done
    return 1
}

_ensure_xvfb() {
    if _alive "$XVFB_PID"; then return 0; fi
    # Reuse an existing Xvfb on $DISP if present
    if pgrep -f "Xvfb $DISP" >/dev/null 2>&1; then
        pgrep -f "Xvfb $DISP" | head -1 > "$XVFB_PID"
        log "Reusing Xvfb on $DISP"
        return 0
    fi
    setsid Xvfb "$DISP" -screen 0 1280x1024x24 -nolisten tcp >>"$LOG" 2>&1 &
    echo $! > "$XVFB_PID"
    sleep 1
    log "Xvfb started on $DISP (pid $(cat "$XVFB_PID"))"
}

_ensure_server() {
    if _alive "$SRV_PID" && _port_open; then return 0; fi
    if _port_open; then
        log "rpyc server already listening on :$PORT (external)"
        return 0
    fi
    log "Starting rpyc server (Wine Python) on 127.0.0.1:$PORT ..."
    setsid env DISPLAY="$DISP" WINEDEBUG="$WINEDEBUG" \
        wine "$WINPY" -m mt5linux --host 127.0.0.1 -p "$PORT" >>"$LOG" 2>&1 &
    echo $! > "$SRV_PID"
    if _wait_port 25; then
        log "rpyc server UP (pid $(cat "$SRV_PID"))"
    else
        err "rpyc server failed to open port :$PORT (see $LOG)"
        exit 1
    fi
}

# Verify a live login through the bridge (auto-launches terminal headless).
_verify_login() {
    DISPLAY="$DISP" MT5_BRIDGE_PORT="$PORT" python3 - "$WIN_PATH" <<'PY'
import os, sys
from mt5linux import MetaTrader5
port = int(os.getenv("MT5_BRIDGE_PORT", "18812"))
win_path = sys.argv[1]
login = int(os.getenv("MT5_LOGIN", "0"))
pw = os.getenv("MT5_PASSWORD", "")
srv = os.getenv("MT5_SERVER", "")
try:
    m = MetaTrader5(host="127.0.0.1", port=port, timeout=120)
except Exception as e:
    print(f"CONNECT_FAIL {e}"); sys.exit(3)
# attach first, then explicit login (auto-launches terminal via path)
ok = m.initialize()
if not ok:
    ok = m.initialize(login=login, password=pw, server=srv, path=win_path)
if not ok:
    print(f"LOGIN_FAIL {m.last_error()}"); sys.exit(1)
ai = m.account_info()
if ai:
    print(f"OK login={ai.login} balance={ai.balance} {ai.currency} server={ai.server}")
ti = m.terminal_info()
if ti:
    print(f"TERMINAL connected={ti.connected} trade_allowed={ti.trade_allowed}")
sys.exit(0)
PY
}

up() {
    _check_prereq
    _ensure_xvfb
    _ensure_server
    log "Verifying MT5 login ..."
    if out="$(_verify_login)"; then
        echo "$out" | sed 's/^/[mt5-bridge] /'
        log "Bridge READY (port=$PORT display=$DISP)"
        echo "$out" | grep -q "trade_allowed=False" && \
            log "WARNING: AlgoTrading OFF in terminal -> bot cannot place orders. Enable 'Algo Trading' in MT5."
        return 0
    else
        err "Login verification failed:"
        echo "$out" | sed 's/^/[mt5-bridge]   /'
        err "If LOGIN_FAIL (-6): run 'scripts/mt5_bridge.sh login-gui' once."
        return 1
    fi
}

down() {
    for p in "$SRV_PID" "$XVFB_PID"; do
        if _alive "$p"; then kill "$(cat "$p")" 2>/dev/null || true; log "stopped pid $(cat "$p")"; fi
        rm -f "$p"
    done
    pkill -f "mt5linux" 2>/dev/null || true
    pkill -f "terminal64.exe" 2>/dev/null || true
    log "Bridge down."
}

status() {
    _alive "$XVFB_PID"  && log "Xvfb    : UP ($(cat "$XVFB_PID"))"  || log "Xvfb    : DOWN"
    if _port_open; then log "rpyc    : UP (:$PORT listening)"; else log "rpyc    : DOWN"; fi
    if _port_open; then
        log "Login check:"
        _verify_login 2>/dev/null | sed 's/^/[mt5-bridge]   /' || log "  (login check failed)"
    fi
    log "port=$PORT display=$DISP run_dir=$RUN_DIR"
}

login_gui() {
    _check_prereq
    local d="${DISPLAY:-:0}"
    log "Opening MT5 on DISPLAY=$d. In MT5: File -> Login to Trade Account"
    log "  Login=${MT5_LOGIN:-?}  Server=${MT5_SERVER:-?}"
    log "Close the terminal after 'authorized'. Then run: scripts/mt5_bridge.sh up"
    DISPLAY="$d" WINEDEBUG=-all wine "$MT5_TERM"
}

install_service() {
    local unit_dir="$HOME/.config/systemd/user"
    mkdir -p "$unit_dir"
    cat > "$unit_dir/mt5-bridge.service" <<EOF
[Unit]
Description=XauBot MT5 Bridge (Wine + mt5linux)
After=graphical-session.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_DIR
ExecStart=$SCRIPT_DIR/mt5_bridge.sh up
ExecStop=$SCRIPT_DIR/mt5_bridge.sh down
TimeoutStartSec=120

[Install]
WantedBy=default.target
EOF
    log "Installed: $unit_dir/mt5-bridge.service"
    log "Enable on login: systemctl --user enable --now mt5-bridge.service"
    log "Logs:            journalctl --user -u mt5-bridge -f"
}

case "${1:-}" in
    up)              up ;;
    down|stop)       down ;;
    restart)         down; sleep 2; up ;;
    status)          status ;;
    login-gui|login_gui) login_gui ;;
    install-service) install_service ;;
    *) echo "Usage: $0 {up|down|restart|status|login-gui|install-service}"; exit 1 ;;
esac
