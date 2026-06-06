"""
Triple-Barrier Labeling (production)
====================================
Labels each bar by simulating a trade with a take-profit, stop-loss, and time
limit -- "whichever barrier is hit first" (Lopez de Prado, *Advances in
Financial ML*). This mirrors real SMC trading mechanics (TP/SL/max-hold) and
produces far better labels than naive next-bar direction.

Design
------
- Barriers are **ATR-scaled** (adapt to volatility / regime).
- **Asymmetric RR** supported, e.g. tp=2.0*ATR, sl=1.0*ATR (RR 2:1), matching
  the SMC scalping rule "tight stop, ≥2R target".
- Long-side simulation by default; the label answers: *if we entered long here,
  would TP or SL be hit first?* -> BUY(1) if TP-first, SELL(0) if SL-first.
- `allow_hold`: when True, time-barrier exits with |return| below a small
  threshold are labelled HOLD(2). For binary models keep allow_hold=False.
- No look-ahead in **features**; labels intentionally use future bars (that is
  what a label is). Last `max_holding` bars are marked -1 (unlabelable).

Usage
-----
    from src.triple_barrier import TripleBarrierLabeler
    labeler = TripleBarrierLabeler(tp_atr=2.0, sl_atr=1.0, max_holding=24)
    df = labeler.label(df)   # df needs columns: high, low, close, atr
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class TripleBarrierLabeler:
    tp_atr: float = 2.0          # take-profit = tp_atr * ATR (asymmetric RR)
    sl_atr: float = 1.0          # stop-loss   = sl_atr * ATR
    max_holding: int = 24        # vertical barrier (bars). M15:24=6h, M1:24=24m
    allow_hold: bool = False     # emit HOLD(2) for inconclusive time exits
    hold_eps_atr: float = 0.25   # |ret| < eps*ATR at time-exit -> HOLD

    def label(self, df: pl.DataFrame, atr_col: str = "atr") -> pl.DataFrame:
        """Apply triple-barrier labeling. Returns df with added columns:
        target (1 BUY / 0 SELL / 2 HOLD / -1 unlabeled), barrier_hit,
        bars_to_barrier, ret_atr (ATR-normalised return at exit).
        """
        required = {"high", "low", "close", atr_col}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"TripleBarrier: missing columns {missing}")
            return df

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        atr = df[atr_col].to_numpy().astype(np.float64)
        n = len(df)

        target = np.full(n, -1, dtype=np.int8)
        barrier = np.full(n, "no_data", dtype="U12")
        bars_to = np.zeros(n, dtype=np.int32)
        ret_atr = np.zeros(n, dtype=np.float32)

        H = self.max_holding
        for i in range(n - H):
            entry = close[i]
            a = atr[i]
            if a <= 0 or np.isnan(a):
                barrier[i] = "no_atr"
                continue

            up = entry + self.tp_atr * a      # take-profit (long)
            dn = entry - self.sl_atr * a      # stop-loss   (long)

            hit = False
            for j in range(1, H + 1):
                k = i + j
                # Conservative: if both barriers in same bar, assume SL first
                # (worst case) to avoid optimistic labels.
                if low[k] <= dn:
                    target[i] = 0            # SELL / stopped out
                    barrier[i] = "sl"
                    bars_to[i] = j
                    ret_atr[i] = (dn - entry) / a
                    hit = True
                    break
                if high[k] >= up:
                    target[i] = 1            # BUY / take-profit
                    barrier[i] = "tp"
                    bars_to[i] = j
                    ret_atr[i] = (up - entry) / a
                    hit = True
                    break

            if not hit:
                final = close[min(i + H, n - 1)]
                r = (final - entry) / a
                bars_to[i] = H
                ret_atr[i] = r
                if self.allow_hold and abs(r) < self.hold_eps_atr:
                    target[i] = 2            # HOLD (inconclusive, flat)
                    barrier[i] = "time_hold"
                else:
                    target[i] = 1 if r >= 0 else 0
                    barrier[i] = "time_up" if r >= 0 else "time_down"

        out = df.with_columns([
            pl.Series("target", target),
            pl.Series("barrier_hit", barrier),
            pl.Series("bars_to_barrier", bars_to),
            pl.Series("ret_atr", ret_atr),
        ])

        self._log_stats(target)
        return out

    @staticmethod
    def _log_stats(target: np.ndarray) -> None:
        labeled = target[target >= 0]
        if labeled.size == 0:
            logger.warning("TripleBarrier: no labeled rows")
            return
        n = labeled.size
        n_buy = int((labeled == 1).sum())
        n_sell = int((labeled == 0).sum())
        n_hold = int((labeled == 2).sum())
        n_unl = int((target == -1).sum())
        logger.info(
            f"TripleBarrier labels: BUY={n_buy} ({n_buy/n*100:.1f}%) "
            f"SELL={n_sell} ({n_sell/n*100:.1f}%) "
            f"HOLD={n_hold} ({n_hold/n*100:.1f}%) unlabeled={n_unl}"
        )
