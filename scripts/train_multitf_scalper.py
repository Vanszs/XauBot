#!/usr/bin/env python3
"""
Multi-Timeframe SMC Scalping Trainer (GPU)
==========================================
End-to-end pipeline:
  1. Download M1 + M15 GOLD history from MT5 (via Linux Wine bridge)
  2. Build the multi-TF dataset (M1 features + M15 HTF context + SMC + news)
  3. Label with the triple-barrier method (TP/SL/time)
  4. Train XGBoost on GPU (device=cuda) with a train/test gap (no leakage)
  5. Walk-forward validation vs a naive baseline

Prereq: bridge up  ->  scripts/mt5_bridge.sh up

Usage:
    python scripts/train_multitf_scalper.py \
        [--m1-bars 99999] [--m15-bars 99999] \
        [--tp-atr 2.0] [--sl-atr 1.0] [--max-hold 24] \
        [--device cuda] [--out models/xgb_scalper_m1m15.pkl]
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger

from src.mt5_connector import MT5Connector
from src.multi_tf_dataset import build_multitf_dataset, get_htf_feature_columns
from src.ml_model import get_default_feature_columns


def fetch(conn, symbol, timeframe, bars):
    logger.info(f"Fetching {bars} {symbol} {timeframe} ...")
    df = conn.get_market_data(symbol, timeframe, bars)
    if df is None or df.height == 0:
        raise RuntimeError(f"No {timeframe} data for {symbol}")
    logger.info(f"  {df.height} bars | {df['time'].min()} -> {df['time'].max()}")
    return df


def feature_list(df: pl.DataFrame):
    base = [f for f in get_default_feature_columns() if f in df.columns and f != "regime"]
    htf = get_htf_feature_columns(df)
    feats = sorted(set(base + htf))
    return feats


def train_gpu(df, feats, device="cuda", train_ratio=0.7, gap=200, rounds=400,
              warmup=50, embargo=24):
    d = df.filter(pl.col("target") >= 0).drop_nulls(subset=feats + ["target"])
    X = d.select(feats).to_numpy().astype(np.float32)
    y = d["target"].to_numpy().astype(np.int32)
    # Drop warmup rows where rolling indicators are still NaN->0 (artificial).
    if warmup > 0 and len(X) > warmup:
        X, y = X[warmup:], y[warmup:]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(X)
    split = int(n * train_ratio)
    # Gap must cover BOTH autocorrelation AND the triple-barrier label horizon
    # (labels peek up to max_holding bars ahead -> embargo to prevent leakage).
    te_start = min(split + max(gap, embargo), n - 1)
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[te_start:], y[te_start:]
    logger.info(f"train={len(Xtr)} test={len(Xte)} gap={te_start-split} "
                f"(embargo={embargo}) warmup={warmup} feats={len(feats)}")

    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
    dtest = xgb.DMatrix(Xte, label=yte, feature_names=feats)
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "device": device, "tree_method": "hist",
        "max_depth": 4, "learning_rate": 0.03,
        "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 1.0, "reg_lambda": 5.0, "gamma": 1.0,
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=rounds,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=20, verbose_eval=50,
    )
    tr_auc = booster.eval(dtrain).split("auc:")[-1]
    te_auc = booster.eval(dtest).split("auc:")[-1]
    logger.info(f"Train AUC={tr_auc} Test AUC={te_auc}")
    return booster, params


def walk_forward(df, feats, device, window=20000, test=4000, step=4000,
                 embargo=24, warmup=50, start_frac=0.0):
    d = df.filter(pl.col("target") >= 0).drop_nulls(subset=feats + ["target"])
    X = np.nan_to_num(d.select(feats).to_numpy().astype(np.float32))
    y = d["target"].to_numpy().astype(np.int32)
    if warmup > 0 and len(X) > warmup:
        X, y = X[warmup:], y[warmup:]
    from sklearn.metrics import roc_auc_score, accuracy_score
    aucs, accs = [], []
    i = int(len(X) * start_frac)
    # Embargo between train and test so triple-barrier labels (look max_holding
    # bars ahead) cannot leak across the boundary.
    while i + window + embargo + test <= len(X):
        Xtr, ytr = X[i:i+window], y[i:i+window]
        ts = i + window + embargo
        Xte, yte = X[ts:ts+test], y[ts:ts+test]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            i += step; continue
        dtr = xgb.DMatrix(Xtr, label=ytr); dte = xgb.DMatrix(Xte, label=yte)
        p = {"objective":"binary:logistic","eval_metric":"auc","device":device,
             "tree_method":"hist","max_depth":4,"learning_rate":0.03,
             "min_child_weight":10,"subsample":0.8,"colsample_bytree":0.7,
             "reg_alpha":1.0,"reg_lambda":5.0,"gamma":1.0}
        b = xgb.train(p, dtr, num_boost_round=200, verbose_eval=False)
        pred = b.predict(dte)
        aucs.append(roc_auc_score(yte, pred))
        accs.append(accuracy_score(yte, (pred > 0.5).astype(int)))
        i += step
    if aucs:
        logger.info(f"Walk-forward folds={len(aucs)} avg_AUC={np.mean(aucs):.4f} "
                    f"avg_ACC={np.mean(accs):.4f}")
    # baseline: majority class accuracy
    base_acc = max(np.mean(y), 1 - np.mean(y))
    logger.info(f"Baseline (majority) ACC={base_acc:.4f}")
    return {"wf_auc": float(np.mean(aucs)) if aucs else None,
            "wf_acc": float(np.mean(accs)) if accs else None,
            "baseline_acc": float(base_acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-bars", type=int, default=99999)
    ap.add_argument("--m15-bars", type=int, default=99999)
    ap.add_argument("--symbol", default=os.getenv("SYMBOL", "GOLD"))
    ap.add_argument("--tp-atr", type=float, default=2.0)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="models/xgb_scalper_m1m15.pkl")
    ap.add_argument("--cache", default="data/multitf_dataset.parquet")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    if args.use_cache and Path(args.cache).exists():
        logger.info(f"Loading cached dataset {args.cache}")
        ds = pl.read_parquet(args.cache)
    else:
        conn = MT5Connector(
            login=int(os.getenv("MT5_LOGIN", "0")),
            password=os.getenv("MT5_PASSWORD", ""),
            server=os.getenv("MT5_SERVER", ""),
            path=os.getenv("MT5_WIN_PATH") or os.getenv("MT5_PATH"),
        )
        if not conn.connect():
            logger.error("MT5 connect failed. Start bridge: scripts/mt5_bridge.sh up")
            return 1
        m1 = fetch(conn, args.symbol, "M1", args.m1_bars)
        m15 = fetch(conn, args.symbol, "M15", args.m15_bars)
        conn.disconnect()
        ds = build_multitf_dataset(m1, m15, tp_atr=args.tp_atr,
                                   sl_atr=args.sl_atr, max_holding=args.max_hold)
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        ds.write_parquet(args.cache)
        logger.info(f"Cached dataset -> {args.cache}")

    feats = feature_list(ds)
    logger.info(f"Using {len(feats)} features")

    booster, params = train_gpu(ds, feats, device=args.device)
    wf = walk_forward(ds, feats, device=args.device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"booster": booster, "features": feats,
                     "params": params, "walk_forward": wf,
                     "tp_atr": args.tp_atr, "sl_atr": args.sl_atr,
                     "max_hold": args.max_hold, "symbol": args.symbol}, f)
    logger.info(f"Saved model -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
