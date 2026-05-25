"""
signal_engine.py — Institutional Signal Factory (v3)

Unified engine for Regime Detection, Trend Bias, and Entry Triggers.
SIGNAL STAGE of the Pipeline.
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange
from config import REGIME_ADX_THRESHOLD, REGIME_ATR_PCT_THRESHOLD, N_PERIOD_BREAKOUT, RSI_OVERSOLD, RSI_OVERBOUGHT
from utils.safe_core import SafeCore
from utils.schemas import TradeSignal
from logger import trades_logger

class SignalEngine:
    def __init__(self):
        pass

    def compute_all_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Enforces a single-pass computation of all required signals.
        1. Indicators
        2. Regime (TREND, RANGE, LOW_VOL)
        3. Trend Bias (LONG, SHORT)
        4. Triggers (Breakout + RSI)
        """
        if len(df) < 200:
            trades_logger.warning(f"[{symbol}] Insufficient data for SignalEngine (len < 200).")
            return df

        df = df.copy()

        # 1. Base Indicators
        # ADX
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
        df['adx'] = adx_ind.adx()
        
        # ATR
        atr_ind = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
        df['atr'] = atr_ind.average_true_range()
        
        # ATR Percentile
        df['atr_percentile'] = df['atr'].rolling(window=100).apply(
            lambda x: SafeCore.float(pd.Series(x).rank(pct=True).iloc[-1] * 100) if len(x) > 0 else 50.0, raw=False
        ).fillna(50.0)

        # EMA
        df['ema50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
        df['ema200'] = EMAIndicator(close=df['close'], window=200, fillna=True).ema_indicator()

        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=True).rsi()

        # Breakouts
        df['highest_high'] = df['high'].rolling(window=N_PERIOD_BREAKOUT).max()
        df['lowest_low'] = df['low'].rolling(window=N_PERIOD_BREAKOUT).min()

        # Volume SMA for expansion
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # 2. Regime Detection
        conditions = [
            (df['adx'] > REGIME_ADX_THRESHOLD), 
            (df['atr_percentile'] < REGIME_ATR_PCT_THRESHOLD)
        ]
        choices = ['TREND', 'LOW_VOL']
        df['regime'] = np.select(conditions, choices, default='RANGE')

        # 3. Trend Bias
        df['trend_bias'] = np.where(df['ema50'] > df['ema200'], 'LONG', 'SHORT')

        # 4. Triggers (Deconflicted)
        df['trigger'] = None
        
        # TREND: Breakout Only (0.1% relaxed micro-breakout)
        trend_long = (df['regime'] == 'TREND') & (df['close'] > df['highest_high'].shift(1) * 0.999)
        trend_short = (df['regime'] == 'TREND') & (df['close'] < df['lowest_low'].shift(1) * 1.001)
        
        # RANGE: RSI Only (Relaxed to 40/60)
        range_long = (df['regime'] == 'RANGE') & (df['rsi'] < 40)
        range_short = (df['regime'] == 'RANGE') & (df['rsi'] > 60)
        
        # LOW_VOL: Volatility Breakout (Micro-breakout + ATR Expansion or Delta)
        atr_delta = (df['atr_percentile'] - df['atr_percentile'].shift(1)).abs()
        dynamic_threshold_lv = np.where(df['regime'] == 'LOW_VOL', 5, 40)
        atr_pass = (df['atr_percentile'] > dynamic_threshold_lv) | (atr_delta >= 1.0)
        
        low_vol_long = (df['regime'] == 'LOW_VOL') & (df['close'] > df['highest_high'].shift(1) * 0.999) & atr_pass
        low_vol_short = (df['regime'] == 'LOW_VOL') & (df['close'] < df['lowest_low'].shift(1) * 1.001) & atr_pass

        df.loc[trend_long | range_long | low_vol_long, 'trigger'] = 'LONG'
        df.loc[trend_short | range_short | low_vol_short, 'trigger'] = 'SHORT'

        # 5. Dynamic Scoring
        df['score'] = 0
        
        is_long = df['trigger'] == 'LONG'
        is_short = df['trigger'] == 'SHORT'
        is_triggered = is_long | is_short

        # 5a. Trend Alignment
        trend_align_long = is_long & (df['trend_bias'] == 'LONG')
        trend_align_short = is_short & (df['trend_bias'] == 'SHORT')
        df.loc[trend_align_long | trend_align_short, 'score'] += 1

        # 5b. Volume Expansion
        vol_expand = is_triggered & (df['volume'] > df['volume_sma'])
        df.loc[vol_expand, 'score'] += 1

        # 5c. Momentum Strength
        mom_strong_long = is_long & (df['rsi'] > 50) & (df['adx'] > 20)
        mom_strong_short = is_short & (df['rsi'] < 50) & (df['adx'] > 20)
        df.loc[mom_strong_long | mom_strong_short, 'score'] += 1

        # 5d. Volatility Expansion (Dynamic Threshold)
        is_low_vol = df['regime'] == 'LOW_VOL'
        dynamic_threshold = np.where(is_low_vol, 10, 40)
        volatility_expand = is_triggered & (df['atr_percentile'] > dynamic_threshold)
        df.loc[volatility_expand, 'score'] += 1

        # Calculate Confidence
        df['confidence'] = df['score'] / 4.0

        return df

    def get_last_signal(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        """
        Extracts the final signal schema from the computed dataframe.
        """
        last = SafeCore.last_row(df)
        if last is None:
            return TradeSignal(symbol=symbol)

        return TradeSignal(
            symbol=symbol,
            trigger=last.get('trigger'),
            regime=last.get('regime', 'RANGE'),
            trend_bias=last.get('trend_bias', 'NEUTRAL'),
            confidence=SafeCore.float(last.get('confidence', 0.0)),
            score=int(SafeCore.float(last.get('score', 0)))
        )
