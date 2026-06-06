"""
Economic Calendar Data Source
=============================
Provides historical US high-impact economic events with point-in-time fields
(release_time, forecast, previous, actual) for use as ML features.

Design goals
------------
- **No look-ahead leakage**: `actual`/`surprise` are only valid for bars whose
  timestamp is at/after the event's `release_time`. `forecast`/`previous` are
  known in advance and may be used before the release.
- **Pluggable providers**:
    * RecurringCalendarProvider  - deterministic monthly schedule (NFP, CPI,
      PPI, FOMC). Used when no external feed is configured. Provides realistic
      release_times; forecast/previous/actual are filled from a CSV if present,
      otherwise left as NaN (so only timing-based features are produced).
    * CsvCalendarProvider        - loads a real ForexFactory/Investing-style CSV
      with columns: datetime,currency,event,importance,actual,forecast,previous
- Output is a Polars DataFrame, one row per event, UTC timestamps.

CSV path can be set via env CALENDAR_CSV (default data/economic_calendar.csv).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import List, Optional

import polars as pl
from loguru import logger


# Event schema produced by every provider
CALENDAR_SCHEMA = {
    "release_time": pl.Datetime,   # UTC release timestamp
    "event": pl.Utf8,             # e.g. "NFP", "CPI", "FOMC"
    "importance": pl.Int8,        # 1=low 2=med 3=high
    "forecast": pl.Float64,       # known before release (may be NaN)
    "previous": pl.Float64,       # known before release (may be NaN)
    "actual": pl.Float64,         # known only at/after release (may be NaN)
}


@dataclass
class _EventDef:
    name: str
    importance: int
    release_utc_hour: float  # e.g. 13.5 = 13:30 UTC


# Recurring US high-impact events (UTC release times)
_NFP = _EventDef("NFP", 3, 13.5)     # first Friday 13:30 UTC
_CPI = _EventDef("CPI", 3, 13.5)     # ~12th 13:30 UTC
_PPI = _EventDef("PPI", 2, 13.5)     # ~13th 13:30 UTC
_FOMC = _EventDef("FOMC", 3, 19.0)   # ~8 meetings, ~18th 19:00 UTC
_FOMC_MONTHS = {1, 3, 5, 6, 7, 9, 10, 12}


def _hour_to_time(h: float) -> dtime:
    hh = int(h)
    mm = int(round((h - hh) * 60))
    return dtime(hh, mm)


def _first_friday(year: int, month: int) -> datetime:
    d = datetime(year, month, 1)
    # Monday=0..Sunday=6 ; Friday=4
    offset = (4 - d.weekday()) % 7
    return d + timedelta(days=offset)


class RecurringCalendarProvider:
    """Deterministic recurring US calendar.

    Generates events between two dates with realistic release timestamps.
    forecast/previous/actual are left NaN unless a CSV overlay is provided.
    """

    def events(self, start: datetime, end: datetime) -> pl.DataFrame:
        rows = []
        y, m = start.year, start.month
        while datetime(y, m, 1) <= end:
            # NFP: first Friday
            nfp = _first_friday(y, m).replace(
                hour=_hour_to_time(_NFP.release_utc_hour).hour,
                minute=_hour_to_time(_NFP.release_utc_hour).minute,
            )
            rows.append((nfp, _NFP.name, _NFP.importance))
            # CPI ~12th, PPI ~13th
            for day, ev in ((12, _CPI), (13, _PPI)):
                t = _hour_to_time(ev.release_utc_hour)
                rows.append((datetime(y, m, day, t.hour, t.minute), ev.name, ev.importance))
            # FOMC ~18th in FOMC months
            if m in _FOMC_MONTHS:
                t = _hour_to_time(_FOMC.release_utc_hour)
                rows.append((datetime(y, m, 18, t.hour, t.minute), _FOMC.name, _FOMC.importance))
            # next month
            m += 1
            if m > 12:
                m = 1
                y += 1

        rows = [r for r in rows if start <= r[0] <= end]
        if not rows:
            return pl.DataFrame(schema=CALENDAR_SCHEMA)

        return pl.DataFrame(
            {
                "release_time": [r[0] for r in rows],
                "event": [r[1] for r in rows],
                "importance": [r[2] for r in rows],
                "forecast": [float("nan")] * len(rows),
                "previous": [float("nan")] * len(rows),
                "actual": [float("nan")] * len(rows),
            },
            schema=CALENDAR_SCHEMA,
        ).sort("release_time")


class CsvCalendarProvider:
    """Loads a real economic calendar CSV.

    Expected columns (case-insensitive), extra columns ignored:
        datetime/time, event/title, currency, importance/impact,
        actual, forecast, previous

    Only USD high/medium-impact rows are kept. Numeric fields are parsed
    leniently (strings like "3.2%", "250K" -> 3.2, 250000).
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    @staticmethod
    def _to_float(v) -> float:
        if v is None:
            return float("nan")
        s = str(v).strip().replace(",", "")
        if s in ("", "-", "n/a", "N/A", "None"):
            return float("nan")
        mult = 1.0
        if s.endswith("%"):
            s = s[:-1]
        elif s[-1:].upper() == "K":
            mult, s = 1e3, s[:-1]
        elif s[-1:].upper() == "M":
            mult, s = 1e6, s[:-1]
        elif s[-1:].upper() == "B":
            mult, s = 1e9, s[:-1]
        try:
            return float(s) * mult
        except ValueError:
            return float("nan")

    def events(self, start: datetime, end: datetime) -> pl.DataFrame:
        path = Path(self.csv_path)
        if not path.exists():
            logger.warning(f"Calendar CSV not found: {path}")
            return pl.DataFrame(schema=CALENDAR_SCHEMA)

        raw = pl.read_csv(path, infer_schema_length=0)  # all as str, parse manually
        cols = {c.lower(): c for c in raw.columns}

        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_dt = col("datetime", "date", "time", "timestamp")
        c_ev = col("event", "title", "name")
        c_cur = col("currency", "country")
        c_imp = col("importance", "impact")
        c_act = col("actual")
        c_fc = col("forecast")
        c_prev = col("previous")
        if c_dt is None or c_ev is None:
            logger.error("Calendar CSV missing datetime/event columns")
            return pl.DataFrame(schema=CALENDAR_SCHEMA)

        rows = []
        for r in raw.iter_rows(named=True):
            cur = (str(r.get(c_cur, "")).upper() if c_cur else "USD")
            if c_cur and "USD" not in cur and "US" != cur:
                continue
            imp_raw = str(r.get(c_imp, "")).lower() if c_imp else "high"
            if "low" in imp_raw or imp_raw == "1":
                continue  # keep med/high only
            importance = 3 if ("high" in imp_raw or imp_raw == "3") else 2
            try:
                ts = _parse_dt(str(r[c_dt]))
            except Exception:
                continue
            if ts is None or not (start <= ts <= end):
                continue
            rows.append((
                ts, str(r[c_ev]), importance,
                self._to_float(r.get(c_fc)) if c_fc else float("nan"),
                self._to_float(r.get(c_prev)) if c_prev else float("nan"),
                self._to_float(r.get(c_act)) if c_act else float("nan"),
            ))

        if not rows:
            return pl.DataFrame(schema=CALENDAR_SCHEMA)
        return pl.DataFrame(
            {
                "release_time": [r[0] for r in rows],
                "event": [r[1] for r in rows],
                "importance": [r[2] for r in rows],
                "forecast": [r[3] for r in rows],
                "previous": [r[4] for r in rows],
                "actual": [r[5] for r in rows],
            },
            schema=CALENDAR_SCHEMA,
        ).sort("release_time")


def _parse_dt(s: str) -> Optional[datetime]:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%m/%d/%Y %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def get_calendar_provider():
    """Return the configured provider: CSV if present, else recurring."""
    csv_path = os.getenv("CALENDAR_CSV", "data/economic_calendar.csv")
    if Path(csv_path).exists():
        logger.info(f"Economic calendar: CSV provider ({csv_path})")
        return CsvCalendarProvider(csv_path)
    logger.info("Economic calendar: recurring provider (no CSV found)")
    return RecurringCalendarProvider()


def get_events(start: datetime, end: datetime) -> pl.DataFrame:
    """Convenience: events from the configured provider within [start, end]."""
    return get_calendar_provider().events(start, end)
