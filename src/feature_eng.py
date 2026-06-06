"""
Feature Engineering Module - Pure Polars
=========================================
Technical indicators and ML features using Polars expressions.

NO PANDAS. NO TA-Lib.

Implements:
- RSI (Wilder's Smoothing)
- ATR (Average True Range)
- MACD
- Bollinger Bands
- EMA/SMA
- Volume Profile
- ML-ready features
"""

import polars as pl
import numpy as np
from typing import List, Optional
from loguru import logger


class FeatureEngineer:
    """
    Feature engineering using pure Polars expressions.
    
    All calculations are vectorized for maximum performance.
    No loops, no external TA libraries.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def calculate_all(
        self,
        df: pl.DataFrame,
        include_ml_features: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate all technical indicators and features.
        
        Args:
            df: Polars DataFrame with OHLCV data
            include_ml_features: Include ML-specific features
            
        Returns:
            DataFrame with all features added
        """
        df = self.calculate_rsi(df)
        df = self.calculate_atr(df)
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_ema_crossover(df)
        df = self.calculate_volume_features(df)
        
        if include_ml_features:
            df = self.calculate_ml_features(df)
        
        return df
    
    def calculate_rsi(
        self,
        df: pl.DataFrame,
        period: int = 14,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate RSI using Wilder's Smoothing.
        
        Wilder's Smoothing uses alpha = 1/n for ewm_mean.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Args:
            df: DataFrame with price data
            period: RSI period (default 14)
            column: Price column to use
            
        Returns:
            DataFrame with RSI column added
        """
        # Wilder's smoothing alpha
        alpha = 1.0 / period
        
        df = df.with_columns([
            # Calculate price changes
            pl.col(column).diff().alias("_delta"),
        ])
        
        df = df.with_columns([
            # Separate gains (positive changes) and losses (negative changes)
            pl.when(pl.col("_delta") > 0)
                .then(pl.col("_delta"))
                .otherwise(0.0)
                .alias("_gains"),
            
            pl.when(pl.col("_delta") < 0)
                .then(-pl.col("_delta"))
                .otherwise(0.0)
                .alias("_losses"),
        ])
        
        df = df.with_columns([
            # Apply Wilder's smoothing (EWM with alpha=1/period, adjust=False)
            pl.col("_gains")
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("_avg_gain"),
            
            pl.col("_losses")
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("_avg_loss"),
        ])
        
        df = df.with_columns([
            # Calculate RSI
            pl.when(pl.col("_avg_loss") == 0)
                .then(100.0)
                .otherwise(
                    100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
                )
                .alias("rsi"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_delta", "_gains", "_losses", "_avg_gain", "_avg_loss"])
        
        logger.debug(f"RSI calculated (period={period})")
        return df
    
    def calculate_atr(
        self,
        df: pl.DataFrame,
        period: int = 14,
    ) -> pl.DataFrame:
        """
        Calculate ATR (Average True Range) using Wilder's Smoothing.
        
        True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
        ATR = Wilder's smoothing of True Range
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period (default 14)
            
        Returns:
            DataFrame with ATR column added
        """
        alpha = 1.0 / period
        
        df = df.with_columns([
            # Previous close for True Range calculation
            pl.col("close").shift(1).alias("_prev_close"),
        ])
        
        df = df.with_columns([
            # Three components of True Range
            (pl.col("high") - pl.col("low")).alias("_hl"),
            (pl.col("high") - pl.col("_prev_close")).abs().alias("_hpc"),
            (pl.col("low") - pl.col("_prev_close")).abs().alias("_lpc"),
        ])
        
        df = df.with_columns([
            # True Range = maximum of three components
            pl.max_horizontal("_hl", "_hpc", "_lpc").alias("_tr"),
        ])
        
        df = df.with_columns([
            # Apply Wilder's smoothing to get ATR
            pl.col("_tr")
                .ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                .alias("atr"),
        ])
        
        # Calculate ATR percentage (ATR / Close)
        df = df.with_columns([
            (pl.col("atr") / pl.col("close") * 100).alias("atr_percent"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_prev_close", "_hl", "_hpc", "_lpc", "_tr"])
        
        logger.debug(f"ATR calculated (period={period})")
        return df
    
    def calculate_macd(
        self,
        df: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line)
        Histogram = MACD Line - Signal Line
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)
            column: Price column to use
            
        Returns:
            DataFrame with MACD columns added
        """
        df = df.with_columns([
            # Calculate EMAs
            pl.col(column)
                .ewm_mean(span=fast_period, adjust=False)
                .alias("_ema_fast"),
            pl.col(column)
                .ewm_mean(span=slow_period, adjust=False)
                .alias("_ema_slow"),
        ])
        
        df = df.with_columns([
            # MACD line
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"),
        ])
        
        df = df.with_columns([
            # Signal line
            pl.col("macd")
                .ewm_mean(span=signal_period, adjust=False)
                .alias("macd_signal"),
        ])
        
        df = df.with_columns([
            # Histogram
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_ema_fast", "_ema_slow"])
        
        logger.debug(f"MACD calculated ({fast_period}/{slow_period}/{signal_period})")
        return df
    
    def calculate_bollinger_bands(
        self,
        df: pl.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(period)
        Upper Band = Middle + (std_dev * StdDev)
        Lower Band = Middle - (std_dev * StdDev)
        
        Args:
            df: DataFrame with price data
            period: SMA period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            column: Price column to use
            
        Returns:
            DataFrame with Bollinger Band columns added
        """
        df = df.with_columns([
            # Middle band (SMA)
            pl.col(column)
                .rolling_mean(window_size=period)
                .alias("bb_middle"),
            
            # Rolling standard deviation
            pl.col(column)
                .rolling_std(window_size=period)
                .alias("_bb_std"),
        ])
        
        df = df.with_columns([
            # Upper and lower bands
            (pl.col("bb_middle") + std_dev * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - std_dev * pl.col("_bb_std")).alias("bb_lower"),
        ])
        
        df = df.with_columns([
            # Bollinger Band Width (volatility indicator)
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle"))
                .alias("bb_width"),
            
            # %B (position within bands, 0-1 normally)
            ((pl.col(column) - pl.col("bb_lower")) / 
             (pl.col("bb_upper") - pl.col("bb_lower")))
                .alias("bb_percent_b"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_bb_std"])
        
        logger.debug(f"Bollinger Bands calculated (period={period}, std={std_dev})")
        return df
    
    def calculate_ema_crossover(
        self,
        df: pl.DataFrame,
        fast_period: int = 9,
        slow_period: int = 21,
        column: str = "close",
    ) -> pl.DataFrame:
        """
        Calculate EMA crossover signals.
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            column: Price column
            
        Returns:
            DataFrame with EMA and crossover columns
        """
        df = df.with_columns([
            pl.col(column)
                .ewm_mean(span=fast_period, adjust=False)
                .alias(f"ema_{fast_period}"),
            pl.col(column)
                .ewm_mean(span=slow_period, adjust=False)
                .alias(f"ema_{slow_period}"),
        ])
        
        # EMA crossover detection
        df = df.with_columns([
            (pl.col(f"ema_{fast_period}") > pl.col(f"ema_{slow_period}"))
                .alias("_ema_above"),
        ])
        
        df = df.with_columns([
            pl.col("_ema_above").shift(1).alias("_ema_above_prev"),
        ])
        
        df = df.with_columns([
            # Bullish crossover: fast crosses above slow
            (pl.col("_ema_above") & ~pl.col("_ema_above_prev").fill_null(False))
                .cast(pl.Int8)
                .alias("ema_cross_bull"),
            
            # Bearish crossover: fast crosses below slow
            (~pl.col("_ema_above") & pl.col("_ema_above_prev").fill_null(False))
                .cast(pl.Int8)
                .alias("ema_cross_bear"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_ema_above", "_ema_above_prev"])
        
        logger.debug(f"EMA crossover calculated ({fast_period}/{slow_period})")
        return df
    
    def calculate_volume_features(
        self,
        df: pl.DataFrame,
        period: int = 20,
    ) -> pl.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with volume data
            period: Period for volume analysis
            
        Returns:
            DataFrame with volume features
        """
        if "volume" not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df
        
        df = df.with_columns([
            # Volume SMA
            pl.col("volume")
                .rolling_mean(window_size=period)
                .alias("volume_sma"),
        ])
        
        df = df.with_columns([
            # Volume ratio (current / average)
            (pl.col("volume") / pl.col("volume_sma")).alias("volume_ratio"),
            
            # Volume trend (increasing or decreasing)
            (pl.col("volume") > pl.col("volume").shift(1))
                .cast(pl.Int8)
                .alias("volume_increasing"),
        ])
        
        # High volume bars (> 1.5x average)
        df = df.with_columns([
            (pl.col("volume_ratio") > 1.5)
                .cast(pl.Int8)
                .alias("high_volume"),
        ])

        # === ADVANCED: Order Flow Imbalance (Pseudo-OFI) ===
        # Phase 4 - Advanced Exit Strategies
        # Directional volume classification
        df = df.with_columns([
            # Buy volume: close > open (bullish candle)
            pl.when(pl.col("close") > pl.col("open"))
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("buy_volume"),

            # Sell volume: close < open (bearish candle)
            pl.when(pl.col("close") < pl.col("open"))
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("sell_volume"),
        ])

        # Pseudo-OFI calculation
        df = df.with_columns([
            (
                (pl.col("buy_volume") - pl.col("sell_volume")) /
                (pl.col("buy_volume") + pl.col("sell_volume") + 1e-9)
            )
            .alias("ofi_pseudo")
            .fill_nan(0)
            .fill_null(0)
        ])

        # OFI trend and divergence
        df = df.with_columns([
            # Rolling OFI mean (20 bars)
            pl.col("ofi_pseudo").rolling_mean(20).alias("ofi_trend"),

            # Rolling OFI std (for normalization)
            pl.col("ofi_pseudo").rolling_std(20).alias("ofi_std"),
        ])

        df = df.with_columns([
            # OFI divergence (current vs trend)
            (pl.col("ofi_pseudo") - pl.col("ofi_trend"))
            .alias("ofi_divergence")
            .fill_nan(0)
            .fill_null(0)
        ])

        # Volume momentum (acceleration)
        df = df.with_columns([
            # Volume ratio change (1st derivative)
            (pl.col("volume_ratio") / pl.col("volume_ratio").shift(1) - 1)
            .alias("volume_momentum")
            .fill_nan(0)
            .fill_null(0),
        ])

        # Volume toxicity metric
        # Combines: volume acceleration + OFI divergence + spread expansion
        if "spread" in df.columns:
            df = df.with_columns([
                # Toxicity score (0-5+)
                (
                    pl.col("volume_momentum").abs() +
                    pl.col("ofi_divergence").abs() * 2 +
                    (pl.col("spread") / pl.col("spread").rolling_mean(20) - 1).abs()
                )
                .alias("toxicity")
                .fill_nan(0)
                .fill_null(0)
            ])
        else:
            # Simplified toxicity without spread
            df = df.with_columns([
                (
                    pl.col("volume_momentum").abs() +
                    pl.col("ofi_divergence").abs() * 2
                )
                .alias("toxicity")
                .fill_nan(0)
                .fill_null(0)
            ])

        logger.debug(f"Volume features calculated (period={period}, includes OFI & toxicity)")
        return df
    
    def calculate_ml_features(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate ML-specific features for XGBoost.
        
        Includes:
        - Returns and momentum
        - Price position features
        - Volatility features
        - Lag features
        - Time-based features
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            DataFrame with ML features
        """
        # Returns and momentum
        df = df.with_columns([
            # Simple returns
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("returns_5"),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("returns_20"),
            
            # Log returns
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns"),
        ])
        
        # Price position features
        df = df.with_columns([
            # Price position within day's range
            ((pl.col("close") - pl.col("low")) / 
             (pl.col("high") - pl.col("low")))
                .alias("price_position"),
            
            # Distance from SMA
            pl.col("close")
                .rolling_mean(window_size=20)
                .alias("_sma_20"),
        ])
        
        df = df.with_columns([
            (pl.col("close") / pl.col("_sma_20") - 1).alias("dist_from_sma_20"),
        ])
        
        # Volatility features
        df = df.with_columns([
            # Realized volatility (rolling std of returns)
            pl.col("log_returns")
                .rolling_std(window_size=20)
                .alias("volatility_20"),
            
            # Normalized range
            ((pl.col("high") - pl.col("low")) / pl.col("close"))
                .alias("normalized_range"),
            
            # Average normalized range
            ((pl.col("high") - pl.col("low")) / pl.col("close"))
                .rolling_mean(window_size=14)
                .alias("avg_normalized_range"),
        ])
        
        # Lag features
        df = df.with_columns([
            pl.col("close").shift(1).alias("close_lag_1"),
            pl.col("close").shift(2).alias("close_lag_2"),
            pl.col("close").shift(3).alias("close_lag_3"),
            pl.col("close").shift(5).alias("close_lag_5"),
        ])
        
        # Trend features
        df = df.with_columns([
            # Higher high / lower low sequences
            (pl.col("high") > pl.col("high").shift(1))
                .cast(pl.Int8)
                .alias("higher_high"),
            (pl.col("low") < pl.col("low").shift(1))
                .cast(pl.Int8)
                .alias("lower_low"),
        ])
        
        # Rolling trend strength
        df = df.with_columns([
            pl.col("higher_high")
                .rolling_sum(window_size=5)
                .alias("hh_count_5"),
            pl.col("lower_low")
                .rolling_sum(window_size=5)
                .alias("ll_count_5"),
        ])
        
        # Time-based features (if datetime column exists)
        if "time" in df.columns and df["time"].dtype == pl.Datetime:
            df = df.with_columns([
                pl.col("time").dt.hour().alias("hour"),
                pl.col("time").dt.weekday().alias("weekday"),
                
                # Trading session indicators
                ((pl.col("time").dt.hour() >= 8) & (pl.col("time").dt.hour() < 16))
                    .cast(pl.Int8)
                    .alias("london_session"),
                ((pl.col("time").dt.hour() >= 13) & (pl.col("time").dt.hour() < 21))
                    .cast(pl.Int8)
                    .alias("ny_session"),
            ])

            # Economic-news calendar features (recurring US high-impact events).
            # Deterministic from the bar timestamp, so they work for both
            # historical training data and live inference.
            df = self.calculate_news_features(df)

        # Stationary, scale-invariant versions of price-level indicators.
        # Absolute EMA/MACD scale with price and become out-of-distribution when
        # the live price differs from the training range (e.g. GOLD trained at
        # ~2000-3000 but trading at ~4300). These normalized forms stay valid.
        if "ema_9" in df.columns and "ema_21" in df.columns and "atr" in df.columns:
            df = df.with_columns([
                # EMA distance in ATR units (how far price is from each EMA)
                pl.when(pl.col("atr") > 0)
                  .then((pl.col("close") - pl.col("ema_9")) / pl.col("atr"))
                  .otherwise(0.0).alias("ema9_dist_atr"),
                pl.when(pl.col("atr") > 0)
                  .then((pl.col("close") - pl.col("ema_21")) / pl.col("atr"))
                  .otherwise(0.0).alias("ema21_dist_atr"),
                # EMA spread normalized (trend strength, scale-free)
                pl.when(pl.col("atr") > 0)
                  .then((pl.col("ema_9") - pl.col("ema_21")) / pl.col("atr"))
                  .otherwise(0.0).alias("ema_spread_atr"),
            ])
        if "macd" in df.columns:
            df = df.with_columns([
                # MACD family in basis points of price (scale-free)
                (pl.col("macd") / pl.col("close") * 10000).alias("macd_bps"),
                (pl.col("macd_signal") / pl.col("close") * 10000).alias("macd_signal_bps"),
                (pl.col("macd_histogram") / pl.col("close") * 10000).alias("macd_hist_bps"),
            ])

        # Drop temporary columns
        df = df.drop(["_sma_20"])
        
        logger.debug("ML features calculated")
        return df

    def calculate_news_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add economic-news calendar features derived from the bar timestamp.

        US high-impact events that move XAUUSD follow a recurring monthly
        schedule:
          - NFP  : first Friday of the month (13:30 UTC)
          - CPI  : ~day 12 of the month (13:30 UTC)
          - PPI  : ~day 13 of the month (13:30 UTC)
          - FOMC : ~8 meetings/yr; approximated as ~day 18 of Jan/Mar/May/Jun/
                   Jul/Sep/Oct/Dec (19:00 UTC)

        These approximations let the model learn news-driven regimes without an
        external calendar feed. Features produced:
          - news_high_impact_today : 1 if a high-impact event falls on this date
          - news_window            : 1 if within +/- 2h of a high-impact release
          - hours_to_news          : hours until the nearest high-impact release
                                     today (clipped to [-12, 12]; 99 if none)
          - news_risk              : 0..1 proximity risk (1 = at release time)
        """
        if "time" not in df.columns or df["time"].dtype != pl.Datetime:
            return df

        day = pl.col("time").dt.day()
        wd = pl.col("time").dt.weekday()        # Mon=1 .. Sun=7 (polars)
        month = pl.col("time").dt.month()
        hour = pl.col("time").dt.hour()

        # First Friday of month -> NFP. First Friday day-of-month is in 1..7
        # and is a Friday (weekday == 5).
        is_nfp_day = (wd == 5) & (day <= 7)
        # CPI ~12th, PPI ~13th (use a small window to absorb scheduling drift)
        is_cpi_day = (day >= 11) & (day <= 13)
        is_ppi_day = (day >= 12) & (day <= 14)
        # FOMC months (approx) and ~18th
        fomc_months = [1, 3, 5, 6, 7, 9, 10, 12]
        is_fomc_day = month.is_in(fomc_months) & (day >= 17) & (day <= 19)

        high_impact_day = (is_nfp_day | is_cpi_day | is_ppi_day | is_fomc_day)

        # Release hours: data releases 13:30 UTC, FOMC 19:00 UTC.
        # Use the closest release hour active on the day for proximity.
        release_hour = (
            pl.when(is_fomc_day).then(pl.lit(19))
            .otherwise(pl.lit(13))
        )

        hours_to = (release_hour - hour).cast(pl.Float64)
        hours_to_clipped = (
            pl.when(~high_impact_day).then(pl.lit(99.0))
            .otherwise(hours_to.clip(-12.0, 12.0))
        )

        # Proximity risk: 1 at release, decaying over +/- 4h window.
        risk = (
            pl.when(~high_impact_day).then(pl.lit(0.0))
            .otherwise((1.0 - (hours_to.abs() / 4.0)).clip(0.0, 1.0))
        )

        df = df.with_columns([
            high_impact_day.cast(pl.Int8).alias("news_high_impact_today"),
            (high_impact_day & (hours_to.abs() <= 2))
                .cast(pl.Int8).alias("news_window"),
            hours_to_clipped.alias("hours_to_news"),
            risk.alias("news_risk"),
        ])

        # Point-in-time economic calendar values (forecast/previous always,
        # actual/surprise only after each event's release_time -> no leakage).
        df = self.merge_economic_calendar(df)
        return df

    def merge_economic_calendar(self, df: pl.DataFrame) -> pl.DataFrame:
        """Merge forecast/previous/actual/surprise from the economic calendar,
        respecting point-in-time availability (no look-ahead leakage).

        Rules per bar (by timestamp t):
          - forecast / previous : known in advance -> taken from the most recent
            event whose release_time <= t OR the next upcoming event same day
            (forecast is published before the release). We expose the *upcoming
            or just-released* event's forecast/previous.
          - actual / surprise   : only set when t >= release_time of that event;
            before the release they are 0 (matches live conditions).

        Produces columns:
          cal_forecast, cal_previous, cal_actual, cal_surprise, cal_surprise_abs
        All default to 0.0 when no nearby event or values are unknown (NaN->0).
        """
        if "time" not in df.columns or df["time"].dtype != pl.Datetime:
            return df

        try:
            from src.economic_calendar import get_events
        except Exception as e:  # noqa: BLE001
            logger.debug(f"economic_calendar unavailable: {e}")
            return df

        tmin = df["time"].min()
        tmax = df["time"].max()
        if tmin is None or tmax is None:
            return df

        from datetime import timedelta
        events = get_events(tmin - timedelta(days=2), tmax + timedelta(days=2))
        if events is None or events.height == 0:
            # still emit zero columns for schema stability
            return df.with_columns([
                pl.lit(0.0).alias(c) for c in
                ("cal_forecast", "cal_previous", "cal_actual",
                 "cal_surprise", "cal_surprise_abs")
            ])

        # surprise computed at source (may be NaN if forecast/actual unknown)
        events = events.with_columns(
            (pl.col("actual") - pl.col("forecast")).alias("_surprise")
        ).sort("release_time")

        df = df.sort("time")

        # join_asof backward: attach the most recent event at/<= bar time.
        # This gives forecast/previous/actual of the last released event.
        past = df.join_asof(
            events.select([
                pl.col("release_time"),
                pl.col("forecast").alias("_fc_past"),
                pl.col("previous").alias("_prev_past"),
                pl.col("actual").alias("_act_past"),
                pl.col("_surprise").alias("_surp_past"),
            ]),
            left_on="time", right_on="release_time", strategy="backward",
        )

        # join_asof forward: attach the next upcoming event (>= bar time) so we
        # can expose its forecast/previous BEFORE the release (no actual).
        nxt = df.join_asof(
            events.select([
                pl.col("release_time").alias("_next_release"),
                pl.col("forecast").alias("_fc_next"),
                pl.col("previous").alias("_prev_next"),
            ]),
            left_on="time", right_on="_next_release", strategy="forward",
        ).select(["_next_release", "_fc_next", "_prev_next"])

        merged = pl.concat([past, nxt], how="horizontal")

        # forecast/previous: prefer the upcoming event's values when an event is
        # near in the future (its forecast is already public); otherwise fall
        # back to the last released event's values.
        cal_forecast = pl.coalesce([pl.col("_fc_next"), pl.col("_fc_past")])
        cal_previous = pl.coalesce([pl.col("_prev_next"), pl.col("_prev_past")])
        # actual/surprise only from PAST (already released) events.
        cal_actual = pl.col("_act_past")
        cal_surprise = pl.col("_surp_past")

        merged = merged.with_columns([
            cal_forecast.fill_nan(0.0).fill_null(0.0).alias("cal_forecast"),
            cal_previous.fill_nan(0.0).fill_null(0.0).alias("cal_previous"),
            cal_actual.fill_nan(0.0).fill_null(0.0).alias("cal_actual"),
            cal_surprise.fill_nan(0.0).fill_null(0.0).alias("cal_surprise"),
        ]).with_columns(
            pl.col("cal_surprise").abs().alias("cal_surprise_abs")
        )

        return merged.drop([
            "_fc_past", "_prev_past", "_act_past", "_surp_past",
            "_next_release", "_fc_next", "_prev_next", "release_time",
        ], strict=False)
    
    def create_target(
        self,
        df: pl.DataFrame,
        lookahead: int = 1,
        threshold: float = 0.0,
    ) -> pl.DataFrame:
        """
        Create target variable for ML training.
        
        Args:
            df: DataFrame with price data
            lookahead: Bars to look ahead for target
            threshold: Minimum return threshold for positive target
            
        Returns:
            DataFrame with target column
        """
        df = df.with_columns([
            # Future close
            pl.col("close").shift(-lookahead).alias("_future_close"),
        ])
        
        df = df.with_columns([
            # Binary target: 1 if price goes up, 0 otherwise
            ((pl.col("_future_close") / pl.col("close") - 1) > threshold)
                .cast(pl.Int32)
                .alias("target"),
            
            # Return target (for regression)
            (pl.col("_future_close") / pl.col("close") - 1)
                .alias("target_return"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_future_close"])
        
        logger.debug(f"Target created (lookahead={lookahead}, threshold={threshold})")
        return df
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Get list of feature columns for ML training.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns
        exclude_cols = {
            "time", "open", "high", "low", "close", "volume",
            "spread", "real_volume", "target", "target_return",
            # SMC columns that are signals, not features
            "swing_high_level", "swing_low_level",
            "fvg_top", "fvg_bottom", "fvg_mid",
            "ob_top", "ob_bottom",
            "bos_level", "choch_level",
            "bsl_level", "ssl_level",
            "last_swing_high", "last_swing_low",
        }
        
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith("_")  # Temporary columns
        ]
        
        return feature_cols


def get_default_feature_engineer() -> FeatureEngineer:
    """Get default configured feature engineer."""
    return FeatureEngineer()


if __name__ == "__main__":
    # Test feature engineering with synthetic data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 500
    
    base_price = 2000.0
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
        "open": prices,
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.001),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.001),
        "close": prices * (1 + np.random.randn(n) * 0.0005),
        "volume": np.random.randint(1000, 10000, n),
    })
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Calculate all features
    df = fe.calculate_all(df, include_ml_features=True)
    
    # Create target
    df = fe.create_target(df, lookahead=1)
    
    # Get feature columns
    feature_cols = fe.get_feature_columns(df)
    
    print("\n=== Feature Engineering Test ===")
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"\nFeatures: {feature_cols}")
    
    # Show sample with key indicators
    print("\n=== Sample Data (Last 5 Rows) ===")
    display_cols = ["time", "close", "rsi", "atr", "macd", "bb_percent_b", "returns_1", "target"]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df.select(available_cols).tail(5))
    
    # Stats for key indicators
    print("\n=== Indicator Statistics ===")
    for col in ["rsi", "atr", "macd", "bb_percent_b"]:
        if col in df.columns:
            stats = df[col].describe()
            print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
