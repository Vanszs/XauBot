#!/usr/bin/env python3
"""
Fast Vectorized Backtest (validation)
=====================================
Answers ONE question quickly: does the model's signal produce a positive
expectancy / win-rate under realistic TP/SL — or was the headline win-rate an
artifact of look-ahead leakage?

Speed: features are already in data/training_data.parquet, the model predicts
the WHOLE set in one batched GPU call, and trade outcomes are evaluated with a
single vectorized forward scan (no O(n^2) per-bar recompute).

Trade model:
  - Enter when model prob crosses the confidence threshold (long if p>=thr,
    short if p<=1-thr).
  - Exit via triple barrier: TP = tp_atr*ATR, SL = sl_atr*ATR, else time limit.
  - Apply spread cost (points) per round trip.

Usage:
  python scripts/fast_backtest.py [--model models/xgboost_model.pkl]
      [--data data/training_data.parquet] [--tp-atr 2 --sl-atr 1 --max-hold 24]
      [--thr 0.6] [--spread 20] [--device cuda]
"""
import argparse, pickle, warnings, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/xgboost_model.pkl")
    ap.add_argument("--data", default="data/training_data.parquet")
    ap.add_argument("--tp-atr", type=float, default=2.0)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--spread", type=float, default=20.0, help="spread cost in points")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    m = pickle.load(open(args.model, "rb"))
    feats = m["feature_names"]
    thr = args.thr if args.thr is not None else m.get("confidence_threshold", 0.6)
    booster = m["model"]

    df = pl.read_parquet(args.data)

    d = df
    # Required price cols must exist; feature cols may be missing (e.g. regime
    # added later by HMM) -> fill with 0 to match live fallback.
    price_need = {"high", "low", "close", "atr"}
    pmiss = price_need - set(df.columns)
    if pmiss:
        logger.error(f"data missing price cols: {pmiss}")
        return 1
    missing_feats = [f for f in feats if f not in d.columns]
    if missing_feats:
        logger.warning(f"filling {len(missing_feats)} missing feature(s) with 0: {missing_feats}")
        d = d.with_columns([pl.lit(0.0).alias(f) for f in missing_feats])

    d = d.drop_nulls(subset=list(price_need))
    n = d.height
    X = np.nan_to_num(d.select(feats).to_numpy().astype(np.float32))
    close = d["close"].to_numpy().astype(np.float64)
    high = d["high"].to_numpy().astype(np.float64)
    low = d["low"].to_numpy().astype(np.float64)
    atr = d["atr"].to_numpy().astype(np.float64)

    # --- batched GPU prediction (whole dataset at once) ---
    dm = xgb.DMatrix(X, feature_names=feats)
    try:
        booster.set_param({"device": args.device})
    except Exception:
        pass
    prob = booster.predict(dm)
    logger.info(f"predicted {n} bars | prob mean={prob.mean():.3f}")

    # --- signal: long p>=thr, short p<=1-thr ---
    sig = np.zeros(n, dtype=np.int8)
    sig[prob >= thr] = 1
    sig[prob <= (1 - thr)] = -1

    # --- vectorized triple-barrier outcome per entry bar ---
    H = args.max_hold
    wins = losses = flat = 0
    pnl_points = 0.0
    rets = []
    n_trades = 0
    last_exit = -1  # simple non-overlap: no new entry until prior trade exits

    for i in range(n - H):
        if sig[i] == 0 or i <= last_exit:
            continue
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = close[i]
        direction = sig[i]
        if direction == 1:
            tp = entry + args.tp_atr * a
            sl = entry - args.sl_atr * a
        else:
            tp = entry - args.tp_atr * a
            sl = entry + args.sl_atr * a

        outcome = None
        for j in range(1, H + 1):
            hi, lo = high[i + j], low[i + j]
            if direction == 1:
                if lo <= sl:  # SL first (conservative)
                    outcome = ("loss", sl); break
                if hi >= tp:
                    outcome = ("win", tp); break
            else:
                if hi >= sl:
                    outcome = ("loss", sl); break
                if lo <= tp:
                    outcome = ("win", tp); break
        if outcome is None:
            exit_px = close[i + H]
            r = (exit_px - entry) * direction
            outcome = ("win" if r > 0 else "loss", exit_px)
            last_exit = i + H
        else:
            last_exit = i + j

        label, exit_px = outcome
        gross = (exit_px - entry) * direction
        net = gross - args.spread  # spread cost per round trip (points)
        pnl_points += net
        rets.append(net)
        n_trades += 1
        if net > 0:
            wins += 1
        else:
            losses += 1

    rets = np.array(rets) if rets else np.array([0.0])
    win_rate = wins / n_trades * 100 if n_trades else 0
    gross_win = rets[rets > 0].sum()
    gross_loss = -rets[rets < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    expectancy = rets.mean()
    sharpe = rets.mean() / rets.std() * np.sqrt(len(rets)) if rets.std() > 0 else 0

    print("\n" + "=" * 50)
    print("FAST VECTORIZED BACKTEST (validation)")
    print("=" * 50)
    print(f"Bars            : {n}")
    print(f"Threshold       : {thr}")
    print(f"TP/SL/hold      : {args.tp_atr}/{args.sl_atr} ATR, {H} bars")
    print(f"Spread cost     : {args.spread} pts/trade")
    print(f"Total trades    : {n_trades}")
    print(f"Win rate        : {win_rate:.1f}%")
    print(f"Profit factor   : {pf:.2f}")
    print(f"Expectancy      : {expectancy:.2f} pts/trade")
    print(f"Net P/L (points): {pnl_points:.0f}")
    print(f"Sharpe (approx) : {sharpe:.2f}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
