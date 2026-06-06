# MT5 on Arch Linux (Wine + mt5linux bridge)

The `MetaTrader5` Python package is Windows-only. On Arch Linux we run the MT5
terminal under **Wine** plus a **Windows Python** that exposes the MT5 API over
an `rpyc` socket (`mt5linux`). The Linux-side bot connects to that socket.

```
Linux Python (bot)  --rpyc-->  Wine Python (mt5linux server)  -->  MT5 terminal  -->  broker
   mt5linux client                MetaTrader5 pkg                 terminal64.exe
```

## One-time setup (already done on this machine)

- Wine: `wine-11.10 (Staging)`  ✅
- Headless display: `Xvfb` (pkg `xorg-server-xvfb`)  ✅
- MT5 terminal installed in Wine: `~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe`  ✅
- Windows Python 3.9 in Wine: `~/.wine/drive_c/users/$USER/AppData/Local/Programs/Python/Python39/python.exe`  ✅
- Wine Python packages: `MetaTrader5`, `mt5linux`, `rpyc`  ✅
- Linux Python packages: `mt5linux`, `rpyc`  ✅

To reproduce the Wine Python install:

```bash
curl -sL -o /tmp/py39.exe https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe
DISPLAY=:1 WINEDEBUG=-all wine /tmp/py39.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
WINPY="$HOME/.wine/drive_c/users/$USER/AppData/Local/Programs/Python/Python39/python.exe"
wine "$WINPY" -m pip install MetaTrader5 mt5linux
```

## Configuration (`.env`)

```env
MT5_LOGIN=...
MT5_PASSWORD=...
MT5_SERVER=...                 # exact broker server name (see terminal)
MT5_BRIDGE_HOST=127.0.0.1
MT5_BRIDGE_PORT=18812
MT5_WIN_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

## Run (robust, one command)

```bash
# Bring the whole bridge up (idempotent): Xvfb + rpyc server + verified login.
# The terminal is auto-launched headless and logged in from .env credentials.
scripts/mt5_bridge.sh up

# Other commands
scripts/mt5_bridge.sh status     # components + live login + trade_allowed
scripts/mt5_bridge.sh restart
scripts/mt5_bridge.sh down
```

`up` is safe to re-run: it only starts what is missing and health-checks the
rpyc port (no blind sleeps). Credentials and port are read from `.env`
(`MT5_LOGIN/PASSWORD/SERVER`, `MT5_BRIDGE_PORT`, `MT5_WIN_PATH`).

### Auto-start on login (systemd --user)

```bash
scripts/mt5_bridge.sh install-service
systemctl --user enable --now mt5-bridge.service
journalctl --user -u mt5-bridge -f      # logs
```

After this the bridge comes up automatically; you never repeat the manual
steps. Then just run the bot:

```bash
python main_live.py
```

### First-time only

The account must be logged into the terminal once so the profile knows it.
Normally `up` does this automatically via explicit login. If you ever get
`LOGIN_FAIL (-6)`, do a one-time GUI login:

```bash
scripts/mt5_bridge.sh login-gui      # File -> Login to Trade Account
```

The bot (`src/mt5_connector.py`) auto-detects the bridge: if the native
`MetaTrader5` import fails but `mt5linux` is present, it connects lazily to
`MT5_BRIDGE_HOST:MT5_BRIDGE_PORT` on the first `connect()` call, trying
`initialize()` attach-mode first then explicit login.

### Zero-step auto-connect

You do **not** need to start anything by hand. When the bot calls `connect()`
and the bridge port is closed, it auto-runs `scripts/mt5_bridge.sh up`
(Xvfb + rpyc server + login) and waits for it to become ready. So on Linux:

```bash
python main_live.py        # bridge is started automatically if needed
```

Disable autostart (e.g. when using the systemd service) with:

```bash
export MT5_BRIDGE_AUTOSTART=0
```

### Broker symbol note

XM Global names spot gold **`GOLD`** (not `XAUUSD`). `.env` sets `SYMBOL=GOLD`.
The connector pre-selects whatever `SYMBOL` is configured.

## Root cause of the `-6 Authorization failed` we hit

Per the official mt5linux docs, the intended flow is:
**(1) open MT5 and log in via the GUI → (2) start the server → (3) call
`initialize()` with no credentials to attach to the running terminal.**

Our first attempt launched the terminal with `/portable` and passed
`login/password/server` to `initialize()`. A fresh portable profile has **no
saved account**, so the broker rejected the fresh login attempt with
`(-6, 'Authorization failed')`. Confirmed: no `accounts.dat`/`servers.dat`
account was present in the Wine prefix.

Fixes applied:
- launcher no longer uses `/portable` (uses the persistent profile)
- added `scripts/mt5_bridge.sh login_gui` for the one-time GUI login
- the connector + test now try `initialize()` attach-mode first

> So the credentials were almost certainly fine — the terminal simply had never
> logged into the account. Do the one-time `login_gui` step, then the bridge
> attaches cleanly.

## QA status (verified)

- rpyc server starts and listens on `127.0.0.1:18812`  ✅
- Linux client connects through to the broker  ✅
- `mt5.initialize(...)` reaches the broker server  ✅
- `src/mt5_connector.py` imports without blocking; backend detected as `bridge` ✅

> NOTE: With the supplied XM credentials the broker returned
> `(-6, 'Authorization failed')`. The transport works end-to-end — this is a
> credential/server-name issue, not an infrastructure problem. Confirm in the
> MT5 terminal:
> - exact **server name** (XM has several, e.g. `XMGlobal-MT5`, `XMGlobal-MT5 2` … `10`)
> - **login number** and **password** (investor vs master password)
> - account is active and AlgoTrading is enabled

## Troubleshooting

- `Authorization failed (-6)` → wrong server name / login / password, or
  expired account. Log into the terminal GUI once to confirm the exact server.
- Client `result expired` / timeout → the Wine MT5 terminal hung; run
  `scripts/mt5_bridge.sh restart`.
- Port already in use → another server instance is running; `mt5_bridge.sh stop`.
