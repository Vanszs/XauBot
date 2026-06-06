#!/usr/bin/env python3
"""
Raw Data Collector (M1 + M15 GOLD)
==================================
Collects RAW OHLCV bars from MT5 (via the Linux Wine bridge) and saves them
UNPROCESSED to data/raw/. Keeping raw data separate from features means we can
re-run preprocessing/labeling experiments without re-downloading.

Pulls the maximum the broker provides (paginated where possible).

Prereq: bridge up -> scripts/mt5_bridge.sh up

Usage:
    python scripts/collect_data.py [--symbol GOLD] [--m1 99999] [--m15 99999]
"""
import argparse, os, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import polars as pl
from loguru import logger


TF_MAP = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 16385}


def _rates_to_df(r):
    return pl.DataFrame({
        "time":  [datetime.utcfromtimestamp(int(x[0])) for x in r],
        "open":  [float(x[1]) for x in r],
        "high":  [float(x[2]) for x in r],
        "low":   [float(x[3]) for x in r],
        "close": [float(x[4]) for x in r],
        "volume":[float(x[5]) for x in r],
    })


def collect(m, symbol, tf_name, want):
    """Fetch up to `want` bars, paginating backwards past the per-call cap."""
    tf = getattr(m, f"TIMEFRAME_{tf_name}")
    frames = []
    r = m.copy_rates_from_pos(symbol, tf, 0, min(want, 99999))
    if r is None or len(r) == 0:
        logger.error(f"{tf_name}: no data ({m.last_error()})")
        return None
    df = _rates_to_df(r)
    frames.append(df)
    got = df.height
    oldest = df["time"].min()

    # paginate backwards
    while got < want:
        import datetime as dt
        r = m.copy_rates_from(symbol, tf, oldest - dt.timedelta(minutes=TF_MAP[tf_name]),
                              min(want - got, 99999))
        if r is None or len(r) <= 1:
            break
        prev = _rates_to_df(r).filter(pl.col("time") < oldest)
        if prev.height == 0:
            break
        frames.append(prev)
        got += prev.height
        new_oldest = prev["time"].min()
        if new_oldest >= oldest:
            break
        oldest = new_oldest

    out = pl.concat(frames).unique(subset=["time"]).sort("time")
    logger.info(f"{tf_name}: {out.height} bars | {out['time'].min()} -> {out['time'].max()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=os.getenv("SYMBOL", "GOLD"))
    ap.add_argument("--m1", type=int, default=99999)
    ap.add_argument("--m15", type=int, default=99999)
    ap.add_argument("--host", default=os.getenv("MT5_BRIDGE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("MT5_BRIDGE_PORT", "18812")))
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    from mt5linux import MetaTrader5
    m = MetaTrader5(host=args.host, port=args.port, timeout=240)
    if not m.initialize():
        m.initialize(login=int(os.getenv("MT5_LOGIN", "0")),
                     password=os.getenv("MT5_PASSWORD", ""),
                     server=os.getenv("MT5_SERVER", ""),
                     path=os.getenv("MT5_WIN_PATH", ""))
    m.symbol_select(args.symbol, True)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for tf, want in (("M1", args.m1), ("M15", args.m15)):
        df = collect(m, args.symbol, tf, want)
        if df is not None:
            p = f"{args.outdir}/{args.symbol}_{tf}.parquet"
            df.write_parquet(p)
            logger.info(f"saved -> {p}")
    m.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
