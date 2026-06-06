#!/usr/bin/env python3
"""
MT5 Bridge Connection Test (Arch Linux + Wine)
==============================================
Connects to the mt5linux rpyc server (started by scripts/mt5_bridge.sh),
initializes the MetaTrader5 terminal with .env credentials, and prints
account + market-data sanity checks.

Prereq:
    scripts/mt5_bridge.sh start

Usage:
    python scripts/test_mt5_bridge.py
"""
import os
import sys

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HOST = os.getenv("MT5_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.getenv("MT5_BRIDGE_PORT", "18812"))
LOGIN = int(os.getenv("MT5_LOGIN", "0"))
PASSWORD = os.getenv("MT5_PASSWORD", "")
SERVER = os.getenv("MT5_SERVER", "")
# Wine-side (Windows) terminal path used by the rpyc server
WIN_PATH = os.getenv("MT5_WIN_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")


def main() -> int:
    try:
        from mt5linux import MetaTrader5
    except ImportError:
        print("ERROR: mt5linux not installed (pip install mt5linux)")
        return 2

    print(f"Connecting to bridge {HOST}:{PORT} ...")
    try:
        mt5 = MetaTrader5(host=HOST, port=PORT)
    except Exception as e:
        print(f"ERROR: cannot reach rpyc server: {e}")
        print("Hint: start it first -> scripts/mt5_bridge.sh start")
        return 3

    # Preferred (official mt5linux) flow: the terminal is already logged in via
    # GUI, so initialize() attaches to it without credentials.
    print("initialize() [attach to already-logged-in terminal] ...")
    ok = mt5.initialize()
    if not ok:
        print(f"  attach failed -> {mt5.last_error()}")
        # Fallback: explicit login (works only if account/server are valid AND
        # the terminal already knows this account).
        print(f"initialize(login={LOGIN}, server={SERVER!r}) [explicit login] ...")
        ok = mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER, path=WIN_PATH)
    if not ok:
        print(f"FAILED: initialize -> {mt5.last_error()}")
        print("Fix: log in once via GUI so the terminal remembers the account:")
        print("     scripts/mt5_bridge.sh login_gui")
        print("Also verify server name, login, and MASTER password (not investor).")
        return 1

    print("initialize OK")
    ai = mt5.account_info()
    if ai is not None:
        print(f"  account : {ai.login} | balance={ai.balance} {ai.currency} | server={ai.server}")
    ti = mt5.terminal_info()
    if ti is not None:
        print(f"  terminal: connected={ti.connected} trade_allowed={ti.trade_allowed}")

    symbol = os.getenv("SYMBOL", "XAUUSD")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 5)
    if rates is not None and len(rates) > 0:
        print(f"  rates   : fetched {len(rates)} {symbol} M15 bars OK")
    else:
        print(f"  rates   : WARN no rates for {symbol} -> {mt5.last_error()}")

    mt5.shutdown()
    print("DONE: bridge fully functional")
    return 0


if __name__ == "__main__":
    sys.exit(main())
