"""Quant architecture, portfolio layer: turns expected returns, risk and the trading cost into target weights.

See docs/QUANT_ARCHITECTURE_KO.md. For every coin the weight w (a share of equity, negative = short) maximises

    alpha * w  -  lam * var * w^2  -  cost * |w - w_now|

where alpha is the expected relative return over the horizon, var the coin's variance over the horizon and cost the
one-side trading cost. The answer is closed form:

    alpha - 2 lam var w_now >  cost   ->  w = (alpha - cost) / (2 lam var)
    alpha - 2 lam var w_now < -cost   ->  w = (alpha + cost) / (2 lam var)
    otherwise                         ->  w = w_now            (the gain would not pay for the trade)

so the band in which a coin is left alone follows from the cost instead of a chosen rank. The book is kept near zero
beta by subtracting mu * beta from alpha, with mu found by bisection; each weight is capped; a book larger than the
gross limit gets a higher risk penalty until it fits; a target smaller than MIN_WEIGHT becomes 0. A coin outside the
universe is closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.ls_sim import SIDE_COST

LAMBDA = 10.0
CAP = 0.10          # at most this share of equity in one coin, either side
GROSS = 1.0         # longs plus shorts at most this multiple of equity
MIN_WEIGHT = 0.005  # a target below half a percent of equity is not worth an order: it becomes 0


def solve(alpha: np.ndarray, var: np.ndarray, w0: np.ndarray, cost: float, lam: float) -> np.ndarray:
    """The closed-form weight per coin for given alphas."""
    pull = alpha - 2 * lam * var * w0
    up, down = (alpha - cost) / (2 * lam * var), (alpha + cost) / (2 * lam * var)
    return np.where(pull > cost, up, np.where(pull < -cost, down, w0))


def _neutral(a: np.ndarray, v: np.ndarray, b: np.ndarray, held: np.ndarray, cost: float, lam: float, cap: float,
             neutral: bool) -> np.ndarray:
    """The capped closed-form weights with mu * beta taken from alpha so that the beta of the book is near zero."""
    weights = lambda mu: np.clip(solve(a - mu * b, v, held, cost, lam), -cap, cap)
    if not neutral or not np.any(b != 0):
        return weights(0.0)
    exposure = lambda mu: float(b @ weights(mu))
    low, high = -1.0, 1.0
    while exposure(low) < 0 and low > -1e6:
        low *= 2
    while exposure(high) > 0 and high < 1e6:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        if exposure(mid) > 0:
            low = mid
        else:
            high = mid
    return weights((low + high) / 2)


def optimize(alpha: np.ndarray, var: np.ndarray, beta: np.ndarray, w0: np.ndarray, cost: float = SIDE_COST,
             lam: float = LAMBDA, cap: float = CAP, gross: float = GROSS, neutral: bool = True,
             min_weight: float = MIN_WEIGHT) -> np.ndarray:
    """Target weights. alpha NaN = outside the universe: closed. var or beta NaN: the coin is left as it is.

    When the book would be larger than `gross`, the risk penalty is raised until it fits, which keeps the cost band
    (scaling every weight down instead would trade every coin at every close). Only if raising it cannot get there,
    because coins inside their band already hold too much, are the weights scaled down."""
    w0 = np.asarray(w0, float)
    out = np.where(np.isnan(alpha), 0.0, w0)
    live = ~np.isnan(alpha) & np.isfinite(var) & (var > 0) & np.isfinite(beta)
    if live.any():
        a, v, b, held = alpha[live], var[live], beta[live], w0[live]
        rest = float(np.abs(out[~live]).sum())
        size = lambda w: rest + float(np.abs(w).sum())
        w = _neutral(a, v, b, held, cost, lam, cap, neutral)
        if size(w) > gross:
            low, high = lam, lam * 2
            while size(_neutral(a, v, b, held, cost, high, cap, neutral)) > gross and high < lam * 1e6:
                low, high = high, high * 2
            for _ in range(25):
                mid = np.sqrt(low * high)
                if size(_neutral(a, v, b, held, cost, mid, cap, neutral)) > gross:
                    low = mid
                else:
                    high = mid
            w = _neutral(a, v, b, held, cost, high, cap, neutral)
        out[live] = w
    total = np.abs(out).sum()
    out = out * (gross / total) if total > gross else out
    return np.where(np.abs(out) < min_weight, 0.0, out)


def mark(w: np.ndarray, move: np.ndarray, rate: np.ndarray) -> tuple[dict, np.ndarray]:
    """One bar held: the return parts (shares of equity at the bar's start) and the weights after the prices moved.
    move: simple price change per coin; rate: funding rate summed over the bar (a long pays a positive rate)."""
    move, rate = np.nan_to_num(move), np.nan_to_num(rate)
    long_gain, short_gain = float((w * move)[w > 0].sum()), float((w * move)[w < 0].sum())
    funding = -float(w @ rate)
    ret = long_gain + short_gain + funding
    drifted = w * (1 + move) / (1 + ret) if ret > -1 else np.zeros_like(w)
    return {"long": long_gain, "short": short_gain, "funding": funding, "ret": ret}, drifted


def run(alpha: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, sigma: pd.DataFrame, beta: pd.DataFrame,
        horizon: int, lam: float = LAMBDA, side_cost: float = SIDE_COST, cap: float = CAP, gross: float = GROSS,
        initial: float = 1000.0) -> pd.DataFrame:
    """Rebalances at every 4h close that has alphas and returns one row per bar, in the columns ls_sim.summary reads.

    alpha: symbol, close_ms, alpha. closes, funding, sigma, beta: index close_ms, one column per symbol (funding
    summed over (close, next close], sigma per bar). The variance over the horizon is sigma^2 * horizon. Trades are
    at the close's price; the last bar also pays for closing everything.
    """
    times, coins = closes.index.to_numpy(), list(closes.columns)
    a = alpha.pivot_table(index="close_ms", columns="symbol", values="alpha").reindex(index=closes.index, columns=coins)
    has = a.notna().any(axis=1).to_numpy()
    raw, marks = closes.to_numpy(float), closes.ffill().to_numpy(float)
    A, R = a.to_numpy(float), funding.reindex_like(closes).to_numpy(float)
    V = (sigma.reindex_like(closes) ** 2 * horizon).to_numpy(float)
    B = beta.reindex_like(closes).to_numpy(float)
    w, equity, rows = np.zeros(len(coins)), initial, []
    for n in range(len(times) - 1):
        traded = 0.0
        if has[n]:
            priced = np.isfinite(raw[n])
            want = optimize(np.where(priced, A[n], np.nan), np.where(priced, V[n], np.nan), B[n], w, side_cost, lam, cap, gross)
            want = np.where(priced, want, w)
            traded, w = float(np.abs(want - w).sum()), want
        parts, w = mark(w, marks[n + 1] / marks[n] - 1, R[n])
        if n == len(times) - 2:
            traded += float(np.abs(w).sum()) * (1 + parts["ret"])
        cost = traded * side_cost
        gross_now = float(np.abs(w).sum())
        start = equity
        equity *= 1 + parts["ret"] - cost
        both = np.isfinite(raw[n]) & np.isfinite(raw[n + 1])
        rows.append({"open_ms": int(times[n]), "close_ms": int(times[n + 1]), "equity": equity, "ret": equity / start - 1,
                     "long": parts["long"], "short": parts["short"], "funding": parts["funding"], "cost": -cost,
                     "traded": traded, "gross": gross_now, "net": float(w.sum()), "positions": int((np.abs(w) > 1e-9).sum()),
                     "market": float((raw[n + 1, both] / raw[n, both] - 1).mean()) if both.any() else 0.0})
        if equity <= 0:
            break
    return pd.DataFrame(rows)
