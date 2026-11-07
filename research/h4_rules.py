"""The five registered rules. Parameters are fixed by docs/research/4h_preregistration.md; do not tune them.

Each array entry describes what is known at that bar's close and acts at the next bar's open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Rule:
    name: str
    stop_atr: float
    trailing: bool = False
    max_bars: int | None = None


@dataclass
class Signals:
    atr: np.ndarray
    enter_long: np.ndarray
    enter_short: np.ndarray
    exit_long: np.ndarray
    exit_short: np.ndarray


RULES = {
    "A": Rule("A_breakout", 3.0, trailing=True),
    "B": Rule("B_momentum", 3.0),
    "C": Rule("C_ema_cross", 3.0),
    "D": Rule("D_pullback", 2.0, max_bars=30),
    "E": Rule("E_breakout_long", 3.0, trailing=True),
}


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def signals(key: str, bars: np.ndarray) -> Signals:
    high, low, close = (pd.Series(bars[:, i]) for i in (2, 3, 4))
    never = pd.Series(False, index=close.index)
    enter_long, enter_short, exit_long, exit_short = never, never, never, never
    if key in ("A", "E"):
        enter_long = close > high.rolling(55).max().shift(1)
        if key == "A":
            enter_short = close < low.rolling(55).min().shift(1)
    elif key == "B":
        ret = close / close.shift(180) - 1
        enter_long, enter_short = ret > 0.10, ret < -0.10
        exit_long, exit_short = ret <= 0, ret >= 0
    elif key == "C":
        fast, slow = _ema(close, 50), _ema(close, 200)
        above = fast > slow
        below = fast < slow
        enter_long = above & below.shift(1, fill_value=False)
        enter_short = below & above.shift(1, fill_value=False)
        exit_long, exit_short = enter_short, enter_long
    elif key == "D":
        trend, mean, dev = _ema(close, 200), close.rolling(20).mean(), close.rolling(20).std(ddof=0)
        enter_long = (close > trend) & (close < mean - 2.5 * dev)
        enter_short = (close < trend) & (close > mean + 2.5 * dev)
        exit_long, exit_short = close > mean, close < mean
    else:
        raise KeyError(key)
    return Signals(_atr(high, low, close).to_numpy(), *(s.fillna(False).to_numpy(dtype=bool) for s in (enter_long, enter_short, exit_long, exit_short)))
