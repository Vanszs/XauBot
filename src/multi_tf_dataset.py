"""
Multi-Timeframe Dataset Builder (M1 + M15) for SMC Scalping
===========================================================
Builds a training dataset on the **M1 timeframe** (execution / entry) enriched
with **M15 higher-timeframe (HTF) context** (bias, regime, SMC structure,
premium/discount). This encodes the SMC scalping methodology:

    HTF bias (M15)  ->  LTF entry timing (M1)

No look-ahead leakage
---------------------
Each M1 bar is joined to the **most recent CLOSED M15 bar** via
`join_asof(strategy="backward")` on a *shifted* M15 timestamp. We shift the M15
release time forward by one M15 interval so an M1 bar at time t only sees M15
data whose candle has fully closed at or before t. This matches live conditions
exactly (you never know the current, still-forming M15 candle).

Labeling uses the triple-barrier method on M1.

Output columns:
  - all M1 features (technical + SMC + news/calendar from FeatureEngineer/SMC)
  - htf_* columns: M15 context broadcast to M1
  - target: triple-barrier label
"""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

import polars as pl
from loguru import logger

from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from src.triple_barrier import TripleBarrierLabeler


# M15 context columns to broadcast onto M1 (HTF bias / structure / regime).
# Only scale-invariant features (no raw ema/atr/macd which are price-level).
HTF_SOURCE_COLS = [
    "rsi", "atr_percent",
    "ema9_dist_atr", "ema21_dist_atr", "ema_spread_atr",
    "macd_hist_bps",
    "market_structure", "bos", "choch",
    "range_position", "premium_zone", "discount_zone", "equilibrium_zone",
    "displacement", "displacement_strength",
    "ob", "fvg_signal",
]


def _prefix_htf(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """Select + rename the chosen columns with an htf_ prefix (+ keep time)."""
    have = [c for c in cols if c in df.columns]
    return df.select(["time"] + have).rename({c: f"htf_{c}" for c in have})


def build_features_single_tf(
    df: pl.DataFrame,
    swing_length: int = 5,
    include_ml_features: bool = True,
) -> pl.DataFrame:
    """Run the full feature stack (technical + news/calendar + SMC) on one TF."""
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=include_ml_features)
    smc = SMCAnalyzer(swing_length=swing_length)
    df = smc.calculate_all(df)
    return df


def build_multitf_dataset(
    m1: pl.DataFrame,
    m15: pl.DataFrame,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
    max_holding: int = 24,
    m15_interval_min: int = 15,
) -> pl.DataFrame:
    """Assemble the M1+M15 SMC scalping training dataset.

    Args:
        m1, m15: OHLCV Polars frames with a 'time' Datetime column.
        tp_atr/sl_atr/max_holding: triple-barrier params (on M1).
        m15_interval_min: minutes per HTF bar (for the close-time shift).

    Returns:
        M1 dataframe with M1 features + htf_* M15 context + triple-barrier target.
    """
    if "time" not in m1.columns or "time" not in m15.columns:
        raise ValueError("both m1 and m15 require a 'time' column")

    logger.info(f"Building features: M1={m1.height} bars, M15={m15.height} bars")
    m1f = build_features_single_tf(m1).sort("time")
    m15f = build_features_single_tf(m15).sort("time")

    # HTF context: shift M15 timestamp forward by one interval so only CLOSED
    # M15 candles are visible to an M1 bar (no look-ahead).
    htf = _prefix_htf(m15f, HTF_SOURCE_COLS).with_columns(
        (pl.col("time") + pl.duration(minutes=m15_interval_min)).alias("_htf_avail")
    ).sort("_htf_avail")

    merged = m1f.join_asof(
        htf.drop("time"),
        left_on="time",
        right_on="_htf_avail",
        strategy="backward",
    )

    # HTF-derived convenience feature: bias from the (stationary) EMA spread.
    if "htf_ema_spread_atr" in merged.columns:
        merged = merged.with_columns(
            pl.when(pl.col("htf_ema_spread_atr") > 0).then(1)
              .when(pl.col("htf_ema_spread_atr") < 0).then(-1)
              .otherwise(0).cast(pl.Int8).alias("htf_bias")
        )

    # Triple-barrier labels on M1
    labeler = TripleBarrierLabeler(tp_atr=tp_atr, sl_atr=sl_atr, max_holding=max_holding)
    merged = labeler.label(merged)

    # Drop helper col
    merged = merged.drop(["_htf_avail"], strict=False)

    logger.info(f"Multi-TF dataset: {merged.height} rows, {len(merged.columns)} cols")
    return merged


def get_htf_feature_columns(df: pl.DataFrame) -> List[str]:
    """Return the htf_* feature columns present in df (for model features)."""
    return [c for c in df.columns if c.startswith("htf_")]
