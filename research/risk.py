"""Quant architecture, risk layer: how much each coin moves and how much it moves with the market.

See docs/QUANT_ARCHITECTURE_KO.md. Everything here at a close uses returns up to and including that close only.

  sigma   exponentially weighted standard deviation of the 4h log return (half-life HALF_LIFE bars)
  beta    the coin's beta to the equal-weight mean return of the universe over the last BETA_WINDOW bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = 84          # bars: two weeks
VOL_MIN_BARS = 42       # a week of returns before a coin gets a sigma
BETA_WINDOW = 360       # bars: 60 days
BETA_MIN_BARS = 180
RESIDUAL_FLOOR = 0.25   # a coin's own variance is at least this share of its total variance


def returns(closes: pd.DataFrame) -> pd.DataFrame:
    """4h log returns; a missing close leaves the returns on both sides of it missing."""
    return np.log(closes).diff()


def market_return(rets: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.Series:
    """The equal-weight mean return of the coins that are members at each close (all coins when members is None)."""
    r = rets if members is None else rets.where(members.reindex_like(rets).fillna(False).astype(bool))
    return r.mean(axis=1)


def sigma(rets: pd.DataFrame) -> pd.DataFrame:
    """Per-bar volatility from an exponentially weighted mean of squared returns (the mean return is taken as zero)."""
    return np.sqrt((rets ** 2).ewm(halflife=HALF_LIFE, min_periods=VOL_MIN_BARS, ignore_na=True).mean())


def beta(rets: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Rolling beta of every coin to `market` over BETA_WINDOW bars."""
    out = {}
    m = market.reindex(rets.index)
    for s in rets.columns:
        both = rets[s].notna() & m.notna()
        x, y = m.where(both), rets[s].where(both)
        cov = (x * y).rolling(BETA_WINDOW, min_periods=BETA_MIN_BARS).mean() - \
            x.rolling(BETA_WINDOW, min_periods=BETA_MIN_BARS).mean() * y.rolling(BETA_WINDOW, min_periods=BETA_MIN_BARS).mean()
        var = (x * x).rolling(BETA_WINDOW, min_periods=BETA_MIN_BARS).mean() - x.rolling(BETA_WINDOW, min_periods=BETA_MIN_BARS).mean() ** 2
        out[s] = cov / var.where(var > 0)
    return pd.DataFrame(out, index=rets.index)


def portfolio_vol(w: np.ndarray, sig: np.ndarray, b: np.ndarray, market_sigma: float) -> float:
    """Per-bar volatility of weights `w` under a one-factor model: the market part plus each coin's own part."""
    own = np.maximum(sig ** 2 - (b * market_sigma) ** 2, RESIDUAL_FLOOR * sig ** 2)
    return float(np.sqrt((w @ b) ** 2 * market_sigma ** 2 + (w ** 2 * own).sum()))
