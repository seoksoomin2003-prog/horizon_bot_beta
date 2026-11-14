"""Long-short basket simulator for docs/research/direction_preregistration_v12.md. Parameters are fixed there; do not tune."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.direction import month_interval

SIDE_COST = 0.0010
MIN_COINS = 20          # a basket stays empty at a close with fewer predicted coins
LEG = 0.5               # longs and shorts each take half of the basket's capital
DAY_BARS = 6


def targets(p: pd.Series, k: int | None, min_coins: int = MIN_COINS) -> dict[str, float]:
    """Weights of one basket from p by symbol: the k highest long, the k lowest short, equal money per coin.
    k=None means a tenth of the predicted coins. Equal p is ordered by symbol name."""
    if len(p) < min_coins:
        return {}
    order = sorted(p.index, key=lambda s: (p[s], s))
    k = int(len(order) * 0.1 + 0.5) if k is None else k
    return {s: -LEG / k for s in order[:k]} | {s: LEG / k for s in order[-k:]}


def run(pred: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, horizon: int, k: int | None,
        side_cost: float = SIDE_COST, initial: float = 1000.0, min_coins: int = MIN_COINS) -> pd.DataFrame:
    """Trades the baskets and returns one row per 4h bar.

    pred: symbol, close_ms, p. closes: price at each 4h close (index close_ms, one column per symbol, NaN when missing).
    funding: same shape, the funding rates summed over (close, next close].
    `horizon` baskets share the capital equally; each is rebuilt every `horizon` bars at the close's price and left
    alone in between. Only the difference to the new target is traded, at `side_cost` per unit of notional.
    """
    times = closes.index.to_numpy()
    column = {s: i for i, s in enumerate(closes.columns)}
    raw, marks, rates = closes.to_numpy(float), closes.ffill().to_numpy(float), funding.to_numpy(float)
    by_time = {t: g.set_index("symbol")["p"] for t, g in pred.groupby("close_ms")}
    equity, baskets, rows = initial, [{} for _ in range(horizon)], []
    for n in range(len(times) - 1):
        t, basket = times[n], baskets[n % horizon]
        p = by_time.get(t)
        want = {s: w * equity / horizon for s, w in (targets(p, k, min_coins) if p is not None else {}).items()}
        traded = sum(abs(want.get(s, 0.0) - basket.get(s, 0.0)) for s in set(want) | set(basket))
        basket.clear()
        basket.update(want)
        if n == len(times) - 2:         # the last bar also pays for closing everything at its end
            last = sum(abs(v) * marks[n + 1, column[s]] / marks[n, column[s]] for b in baskets for s, v in b.items())
        else:
            last = 0.0
        long_gain = short_gain = paid = gross = 0.0
        for b in baskets:
            for s, held in b.items():
                i = column[s]
                move = marks[n + 1, i] / marks[n, i] - 1
                if held > 0:
                    long_gain += held * move
                else:
                    short_gain += held * move
                paid += held * rates[n, i]
                gross += abs(held)
                b[s] = held * (1 + move)
        cost = (traded + last) * side_cost
        start = equity
        equity += long_gain + short_gain - paid - cost
        both = np.isfinite(raw[n]) & np.isfinite(raw[n + 1])
        rows.append({"open_ms": int(t), "close_ms": int(times[n + 1]), "equity": equity, "ret": equity / start - 1,
                     "long": long_gain / start, "short": short_gain / start, "funding": -paid / start, "cost": -cost / start,
                     "traded": (traded + last) / start, "gross": gross / start,
                     "market": float((raw[n + 1, both] / raw[n, both] - 1).mean()) if both.any() else 0.0})
        if equity <= 0:
            break
    return pd.DataFrame(rows)


def summary(bars: pd.DataFrame, recent_ms: int, initial: float = 1000.0) -> dict:
    """What the v12 gates and the reference lines read from one run."""
    when = pd.to_datetime(bars["open_ms"], unit="ms")
    log = np.log1p(bars["ret"])
    monthly = log.groupby(when.dt.to_period("M")).sum()
    low, high = month_interval(monthly, pd.Series(1.0, index=monthly.index))
    equity = np.concatenate([[initial], bars["equity"].to_numpy()])
    recent = bars["open_ms"] >= recent_ms
    market = bars["market"]
    before_cost = float((bars["ret"] - bars["cost"]).sum())
    top3 = float(monthly.sort_values().iloc[-3:].sum() / monthly.sum()) if monthly.sum() > 0 else None
    return {"total": float(equity[-1] / initial - 1), "months": int(len(monthly)),
            "month_mean": float(monthly.mean()), "month_ci": [low, high],
            "month_sharpe": float(monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() > 0 else 0.0,
            "by_year": {int(y): float(np.expm1(v)) for y, v in log.groupby(when.dt.year).sum().items()},
            "recent": float(np.expm1(log[recent].sum())),
            "max_drawdown": float((1 - equity / np.maximum.accumulate(equity)).max()),
            "beta": float(np.cov(bars["ret"], market)[0, 1] / market.var()) if market.var() > 0 else 0.0,
            "turnover_per_day": float(bars["traded"].mean() * DAY_BARS), "gross": float(bars["gross"].mean()),
            "parts": {k: float(bars[k].sum()) for k in ("long", "short", "funding", "cost")},
            "break_even_side_cost": before_cost / float(bars["traded"].sum()) if bars["traded"].sum() > 0 else None,
            "top3_month_share": top3, "positive_months": float((monthly > 0).mean())}
