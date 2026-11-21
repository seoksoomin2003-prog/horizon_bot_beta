"""v17: the top coin by an absolute score, mu = beta * (market forecast) + alpha, long (or short) or flat.

See docs/research/direction_preregistration_v17.md.

Market layer: the trend of the universe's equal-weight index over 28 and 84 days, each divided by the index's
volatility over that span, averaged into z. Every month a regression through the origin of the next 7 days' index
return (over volatility * sqrt(7 days)) on z, fitted on the closes answered by the month's first day, gives the slope b;
the forecast is m_hat = SHRINK * b * z * volatility * sqrt(7 days).

Simulator: [time x coin] arrays on the 4h close grid as in research/topcoin.py. Decisions at close t trade at close[t];
the position then earns bar t + 1 and pays funding[t].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import risk
from research.direction import month_starts
from research.h4_data import BAR_MS
from research.topcoin import TAKER_COST, Panel, Result

LOOKBACKS = (168, 504)                 # bars: 28 and 84 days
HORIZON = 42                           # bars: 7 days
SHRINK = 0.5
MIN_ROWS = 360                         # answered closes before the slope is used (60 days); fewer: m_hat = 0
ROUND_TRIP = 2 * TAKER_COST            # c in the score
TARGET_BAR_VOL = 0.5 / np.sqrt(6 * 365)    # 50% a year, per 4h bar
SWITCH_K = 2.0


def index_sigma(market: pd.Series) -> pd.Series:
    """Per-bar volatility of the index, the same way as risk.sigma."""
    return np.sqrt((market ** 2).ewm(halflife=risk.HALF_LIFE, min_periods=risk.VOL_MIN_BARS, ignore_na=True).mean())


def trend(market: pd.Series) -> pd.Series:
    """z at every close: the mean over LOOKBACKS of (index return over L bars) / (index volatility * sqrt(L))."""
    s = index_sigma(market)
    return sum(market.rolling(n, min_periods=n).sum() / (s * np.sqrt(n)) for n in LOOKBACKS) / len(LOOKBACKS)


def forward(market: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """The index return over bars t + 1 .. t + horizon, at close t."""
    return market.rolling(horizon, min_periods=horizon).sum().shift(-horizon)


def forecast(market: pd.Series, score_start_ms: int, end_ms: int, z: pd.Series | None = None) -> tuple[pd.Series, dict]:
    """m_hat at every close from score_start_ms to end_ms (NaN before), refit monthly; and the slope used each month.
    `z` replaces the trend signal (the smoke run's random numbers)."""
    z = trend(market) if z is None else z.reindex(market.index)
    scale = index_sigma(market) * np.sqrt(HORIZON)
    y = forward(market) / scale
    times = market.index.to_numpy()
    out = pd.Series(np.nan, index=market.index)
    slopes = {}
    starts = month_starts(score_start_ms, end_ms)
    for m, nxt in zip(starts, starts[1:] + [end_ms]):
        train = (times + HORIZON * BAR_MS <= m) & z.notna().to_numpy() & y.notna().to_numpy()
        zz, yy = z.to_numpy()[train], y.to_numpy()[train]
        b = SHRINK * float((zz * yy).sum() / (zz ** 2).sum()) if train.sum() >= MIN_ROWS and (zz ** 2).sum() > 0 else 0.0
        slopes[int(m)] = b
        rows = (times >= m) & (times < nxt)
        out[rows] = (b * z * scale)[rows]
    return out, slopes


def mu(alpha: np.ndarray, beta: np.ndarray, m_hat: np.ndarray) -> np.ndarray:
    """mu = beta * m_hat + alpha; NaN where alpha or beta is missing (out of the universe)."""
    return beta * m_hat[:, None] + alpha


@dataclass(frozen=True)
class TimedRules:
    sides: tuple[int, ...] = (1,)          # (1,): rule C, long or flat; (1, -1): rule D, long, short or flat
    switch_k: float = SWITCH_K
    target_vol: float = TARGET_BAR_VOL


RULES_C = TimedRules()
RULES_D = TimedRules(sides=(1, -1))


def best(mu_row: np.ndarray, sigma_row: np.ndarray, close_row: np.ndarray, sides: tuple[int, ...]) -> tuple[int, int, float] | None:
    """(coin, side, S) with the highest S = (side * mu - c) / sigma among the tradable coins, or None."""
    ok = np.isfinite(mu_row) & np.isfinite(sigma_row) & (sigma_row > 0) & np.isfinite(close_row)
    top = None
    for side in sides:
        s = np.where(ok, (side * np.where(ok, mu_row, 0.0) - ROUND_TRIP) / np.where(ok, sigma_row, 1.0), -np.inf)
        k = int(np.argmax(s))
        if np.isfinite(s[k]) and (top is None or s[k] > top[2]):
            top = (k, side, float(s[k]))
    return top


def simulate(panel: Panel, MU: np.ndarray, rules: TimedRules, start: int, end: int, cost: float = 1.0,
             initial: float = 1000.0) -> Result:
    """Rule C or D over closes start .. end. `cost` scales the cost charged; decisions always use ROUND_TRIP."""
    c = TAKER_COST * cost
    C, S, F = panel.close, panel.sigma, panel.funding
    equity = initial
    curve = [equity]
    res = Result(equity=np.empty(0))
    pos = None                 # dict(j, side, q (signed units), entry, last, opened, start_equity)

    def close(t: int, price: float, why: str) -> None:
        nonlocal equity, pos
        equity -= abs(pos["q"]) * price * c
        res.trades.append({"symbol": panel.symbols[pos["j"]], "side": pos["side"], "entry_ms": int(panel.times[pos["opened"]]),
                           "exit_ms": int(panel.times[t]), "entry": pos["entry"], "exit": price, "why": why,
                           "net": equity / pos["start_equity"] - 1, "bars": t - pos["opened"]})
        pos = None

    for t in range(start, end):
        cand = best(MU[t], S[t], C[t], rules.sides)
        if pos is not None:
            j, side = pos["j"], pos["side"]
            if not np.isfinite(MU[t, j]):
                close(t, C[t, j] if np.isfinite(C[t, j]) else pos["last"], "universe")
            elif side * MU[t, j] < 0:
                close(t, C[t, j], "signal")
            elif cand is not None and (cand[0], cand[1]) != (j, side) and \
                    cand[1] * MU[t, cand[0]] - side * MU[t, j] > rules.switch_k * ROUND_TRIP:
                close(t, C[t, j], "switch")
        if pos is None and equity > 0 and cand is not None and cand[2] > 0:
            k, side, _ = cand
            notional = equity * min(1.0, rules.target_vol / S[t, k])
            res.orders += 1
            res.fills += 1
            start_equity = equity
            equity -= notional * c
            pos = {"j": k, "side": side, "q": side * notional / C[t, k], "entry": C[t, k], "last": C[t, k],
                   "opened": t, "start_equity": start_equity}
        u = t + 1
        if pos is not None and np.isfinite(C[u, pos["j"]]):
            j = pos["j"]
            res.bars_in += 1
            f = F[t, j] if np.isfinite(F[t, j]) else 0.0
            equity -= pos["side"] * f * abs(pos["q"]) * pos["last"]
            equity += pos["q"] * (C[u, j] - pos["last"])
            pos["last"] = C[u, j]
            if equity <= 0:
                equity = 0.0
                res.trades.append({"symbol": panel.symbols[j], "side": pos["side"], "entry_ms": int(panel.times[pos["opened"]]),
                                   "exit_ms": int(panel.times[u]), "entry": pos["entry"], "exit": C[u, j], "why": "liquidated",
                                   "net": -1.0, "bars": u - pos["opened"]})
                pos = None
        curve.append(equity)
    if pos is not None:
        res.trades.append({"symbol": panel.symbols[pos["j"]], "side": pos["side"], "entry_ms": int(panel.times[pos["opened"]]),
                           "exit_ms": int(panel.times[end]), "entry": pos["entry"], "exit": pos["last"], "why": "open",
                           "net": equity / pos["start_equity"] - 1, "bars": end - pos["opened"]})
    res.equity = np.array(curve)
    return res


def shifted(m_hat: np.ndarray, start: int, end: int, seed: int, min_shift: int = 180 * 6) -> np.ndarray:
    """The null: m_hat over closes start .. end rolled by a random offset of at least min_shift bars (wrapping)."""
    n = end + 1 - start
    rng = np.random.default_rng(seed)
    out = m_hat.copy()
    out[start:end + 1] = np.roll(m_hat[start:end + 1], int(rng.integers(min_shift, n - min_shift)))
    return out
