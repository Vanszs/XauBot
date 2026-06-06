#!/usr/bin/env python3
"""
Download ~1 year of training data from MT5 (via the Linux Wine bridge) and
build the full feature set (technical + SMC + news calendar). Saves to
data/training_data.parquet.

Prereq: bridge up  ->  scripts/mt5_bridge.sh up

Usage:
    python scripts/download_training_data.py [--bars 35000] [--symbol GOLD]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import polars as pl
from loguru import logger

from src.mt5_connector import MT5Connector
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=int(os.getenv("TRAIN_BARS", "35000")),
                    help="Max M15 bars to request (~25k = 1 year). Broker returns up to its limit.")
    ap.add_argument("--symbol", default=os.getenv("SYMBOL", "GOLD"))
    ap.add_argument("--timeframe", default=os.getenv("EXECUTION_TIMEFRAME", "M15"))
    ap.add_argument("--out", default="data/training_data.parquet")
    args = ap.parse_args()

    conn = MT5Connector(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
        path=os.getenv("MT5_WIN_PATH") or os.getenv("MT5_PATH"),
    )
    if not conn.connect():
        logger.error("Could not connect to MT5. Is the bridge up? (scripts/mt5_bridge.sh up)")
        return 1

    logger.info(f"Requesting {args.bars} bars of {args.symbol} {args.timeframe} ...")
    df = conn.get_market_data(args.symbol, args.timeframe, args.bars)
    conn.disconnect()

    if df is None or len(df) == 0:
        logger.error("No data returned. Check the symbol name (XM uses 'GOLD').")
        return 1

    n = len(df)
    span = df["time"].max() - df["time"].min()
    logger.info(f"Received {n} bars | {df['time'].min()} -> {df['time'].max()} ({span})")
    if n < args.bars:
        logger.warning(f"Broker returned fewer bars than requested ({n} < {args.bars}) — "
                       "this is the broker's max available history.")

    # Build features (technical + SMC + time + NEWS calendar)
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=True)
    smc = SMCAnalyzer(swing_length=5)
    df = smc.calculate_all(df)
    df = fe.create_target(df, lookahead=1)

    news_cols = [c for c in ("news_high_impact_today", "news_window",
                             "hours_to_news", "news_risk") if c in df.columns]
    logger.info(f"News features present: {news_cols}")
    logger.info(f"Total columns: {len(df.columns)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    logger.info(f"Saved -> {args.out}  ({n} rows, {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
