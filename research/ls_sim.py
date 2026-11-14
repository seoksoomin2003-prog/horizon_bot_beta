"""Long-short basket simulator for docs/research/direction_preregistration_v12.md. Parameters are fixed there; do not tune."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.direction import month_interval
from research.h4_data import BAR_MS

SIDE_COST = 0.0010
MIN_COINS = 20          # a basket stays empty at a close with fewer predicted coins
LEG = 0.5               # longs and shorts each take half of the basket's capital
DAY_BARS = 6
MAKER_COST = 0.0002     # v13: the exchange's base limit-order fee per side
GAP = 0.0005            # v13: the price must pass a limit order by this fraction before it counts as filled (as in v8)


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


def choose(p: pd.Series | None, held: dict[str, float], k: int | None, keep: float,
           min_coins: int = MIN_COINS) -> tuple[list[str], dict[str, float], int]:
    """v13: what one basket does at its turn: (coins to close, side of the coins to open, coins a side).

    A held long stays while its p is within the top `keep` share of the predicted coins, a held short while within the
    bottom share; at most k stay, the surest first. The empty places go to the highest (lowest) p among the coins that
    do not stay. Fewer than `min_coins` predictions close everything."""
    if p is None or len(p) < min_coins:
        return list(held), {}, 0
    order = sorted(p.index, key=lambda s: (p[s], s))
    k = int(len(order) * 0.1 + 0.5) if k is None else k
    band = max(k, int(len(order) * keep + 0.5))
    low, high = set(order[:band]), set(order[-band:])
    longs = [s for s in reversed(order) if held.get(s, 0.0) > 0 and s in high][:k]
    shorts = [s for s in order if held.get(s, 0.0) < 0 and s in low][:k]
    stay = set(longs) | set(shorts)
    new_longs = [s for s in reversed(order) if s not in stay][:k - len(longs)]
    new_shorts = [s for s in order if s not in stay and s not in new_longs][:k - len(shorts)]
    return [s for s in held if s not in stay], {s: 1.0 for s in new_longs} | {s: -1.0 for s in new_shorts}, k


def run_held(pred: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, horizon: int, k: int | None, keep: float,
             side_cost: float = SIDE_COST, windows: dict[str, pd.DataFrame] | None = None, maker_cost: float = MAKER_COST,
             initial: float = 1000.0, min_coins: int = MIN_COINS) -> pd.DataFrame:
    """v13: like `run`, but a basket follows `choose` and never re-sizes a coin that stays.

    Without `windows` every order trades at the close for `side_cost`. With them (frames shaped like `closes`:
    `limit` the order price five minutes after the close, `low`/`high`/`last` of the next 30 minutes) orders are limit
    orders: one fills at `limit` for `maker_cost` only when the price passes it by GAP. An unfilled entry is dropped
    until the basket's next turn; an unfilled exit goes to market at `last` for `side_cost`. Without 5m bars at that
    close nothing is entered and exits trade at the close for `side_cost`.
    """
    times = closes.index.to_numpy()
    column = {s: i for i, s in enumerate(closes.columns)}
    raw, marks, rates = closes.to_numpy(float), closes.ffill().to_numpy(float), funding.to_numpy(float)
    w = None if windows is None else {name: windows[name].to_numpy(float) for name in ("limit", "low", "high", "last")}
    by_time = {t: g.set_index("symbol")["p"] for t, g in pred.groupby("close_ms")}

    def order(n: int, i: int, buy: bool, entry: bool) -> tuple[bool, float, float]:
        """(done, price paid over the close, cost per unit of notional)."""
        if w is None:
            return True, 1.0, side_cost
        limit = w["limit"][n, i]
        if not np.isfinite(limit):
            return (False, 1.0, 0.0) if entry else (True, 1.0, side_cost)
        if (w["low"][n, i] < limit * (1 - GAP)) if buy else (w["high"][n, i] > limit * (1 + GAP)):
            return True, limit / marks[n, i], maker_cost
        return (False, 1.0, 0.0) if entry else (True, w["last"][n, i] / marks[n, i], side_cost)

    equity, baskets, rows = initial, [{} for _ in range(horizon)], []
    for n in range(len(times) - 1):
        t, basket = times[n], baskets[n % horizon]
        closing, opening, places = choose(by_time.get(t), basket, k, keep, min_coins)
        long_gain = short_gain = traded = cost = paid = gross = 0.0
        entries = filled = exits = exits_limit = 0
        for s in closing:
            held = basket.pop(s)
            _, price, rate = order(n, column[s], held < 0, False)
            if held > 0:
                long_gain += held * (price - 1)
            else:
                short_gain += held * (price - 1)
            traded += abs(held) * price
            cost += abs(held) * price * rate
            exits += 1
            exits_limit += w is not None and rate == maker_cost
        for s, side in opening.items():
            done, price, rate = order(n, column[s], side > 0, True)
            entries += 1
            if not done:
                continue
            filled += 1
            want = side * LEG * equity / horizon / places
            if side > 0:
                long_gain += want * (1 / price - 1)
            else:
                short_gain += want * (1 / price - 1)
            basket[s] = want / price            # its worth at the close's price
            traded += abs(want)
            cost += abs(want) * rate
        timing = long_gain + short_gain
        if n == len(times) - 2:         # the last bar also pays for closing everything at its end, at market
            last = sum(abs(v) * marks[n + 1, column[s]] / marks[n, column[s]] for b in baskets for s, v in b.items())
            traded, cost = traded + last, cost + last * side_cost
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
        start = equity
        equity += long_gain + short_gain - paid - cost
        both = np.isfinite(raw[n]) & np.isfinite(raw[n + 1])
        rows.append({"open_ms": int(t), "close_ms": int(times[n + 1]), "equity": equity, "ret": equity / start - 1,
                     "long": long_gain / start, "short": short_gain / start, "funding": -paid / start, "cost": -cost / start,
                     "traded": traded / start, "gross": gross / start, "timing": timing / start,
                     "entries": entries, "filled": filled, "exits": exits, "exits_limit": int(exits_limit),
                     "market": float((raw[n + 1, both] / raw[n, both] - 1).mean()) if both.any() else 0.0})
        if equity <= 0:
            break
    return pd.DataFrame(rows)


def plan(now: pd.DataFrame | None, held: dict[str, tuple[float, int]], t: int, keep: float,
         min_coins: int = MIN_COINS) -> tuple[list[str], dict[str, int], dict[str, tuple[float, int]], int]:
    """v14: what the book does at close t: (coins to close, new exit time of coins kept past their exit,
    side and exit time of the coins to open, places a side).

    `now` has one row per predicted coin (index symbol): up, up_bars, down, down_bars. `held` maps a coin to its signed
    money and planned exit time. A coin is left alone until its exit time. Then it stays, with a new exit time from its
    current period, only if it is within the first `keep` share of its side's order; otherwise it is closed. Empty
    places go to the first coins of each order that are not held; longs are decided first. A coin that would be closed
    and opened again on the same side at once simply stays.
    """
    if now is None or len(now) < min_coins:
        return list(held), {}, {}, 0
    k = int(len(now) * 0.1 + 0.5)
    band = max(k, int(len(now) * keep + 0.5))
    up, down = now["up"].to_dict(), now["down"].to_dict()
    longs, shorts = sorted(up, key=lambda s: (-up[s], s)), sorted(down, key=lambda s: (down[s], s))
    top, bottom = set(longs[:band]), set(shorts[:band])
    closing, extend = [], {}
    for s, (money, exit_ms) in held.items():
        if exit_ms > t:
            continue
        if s in (top if money > 0 else bottom):
            extend[s] = t + int(now.at[s, "up_bars" if money > 0 else "down_bars"]) * BAR_MS
        else:
            closing.append(s)
    staying = {s: money for s, (money, _) in held.items() if s not in closing}
    opening: dict[str, tuple[float, int]] = {}
    for order, side, bars in ((longs, 1.0, "up_bars"), (shorts, -1.0, "down_bars")):
        free = k - sum(1 for money in staying.values() if money * side > 0)
        for s in order:
            if free <= 0:
                break
            if s not in staying and s not in opening:
                opening[s] = (side, t + int(now.at[s, bars]) * BAR_MS)
                free -= 1
    for s in [s for s in closing if s in opening and opening[s][0] * held[s][0] > 0]:
        extend[s] = opening.pop(s)[1]       # it would be closed and bought back at once: keep it without trading
        closing.remove(s)
    return closing, extend, opening, k


def run_planned(pred: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, keep: float, side_cost: float = SIDE_COST,
                initial: float = 1000.0, min_coins: int = MIN_COINS) -> pd.DataFrame:
    """v14: one book that follows `plan` at every 4h close, trading at the close for `side_cost`.

    pred: symbol, close_ms, up, up_bars, down, down_bars. A new coin gets half the equity divided by the places a side.
    """
    times = closes.index.to_numpy()
    column = {s: i for i, s in enumerate(closes.columns)}
    raw, marks, rates = closes.to_numpy(float), closes.ffill().to_numpy(float), funding.to_numpy(float)
    by_time = {t: g.set_index("symbol") for t, g in pred.groupby("close_ms")}
    equity, held, rows = initial, {}, []
    for n in range(len(times) - 1):
        t = int(times[n])
        closing, extend, opening, places = plan(by_time.get(t), {s: (v[0], v[1]) for s, v in held.items()}, t, keep, min_coins)
        traded, planned = 0.0, 0
        for s in closing:
            traded += abs(held.pop(s)[0])
        for s, exit_ms in extend.items():
            held[s][1] = exit_ms
        for s, (side, exit_ms) in opening.items():
            held[s] = [side * LEG * equity / places, exit_ms]
            traded += abs(held[s][0])
            planned += (exit_ms - t) // BAR_MS
        if n == len(times) - 2:         # the last bar also pays for closing everything at its end
            traded += sum(abs(v[0]) * marks[n + 1, column[s]] / marks[n, column[s]] for s, v in held.items())
        long_gain = short_gain = paid = gross = 0.0
        for s, v in held.items():
            i = column[s]
            move = marks[n + 1, i] / marks[n, i] - 1
            if v[0] > 0:
                long_gain += v[0] * move
            else:
                short_gain += v[0] * move
            paid += v[0] * rates[n, i]
            gross += abs(v[0])
            v[0] *= 1 + move
        cost = traded * side_cost
        start = equity
        equity += long_gain + short_gain - paid - cost
        both = np.isfinite(raw[n]) & np.isfinite(raw[n + 1])
        rows.append({"open_ms": t, "close_ms": int(times[n + 1]), "equity": equity, "ret": equity / start - 1,
                     "long": long_gain / start, "short": short_gain / start, "funding": -paid / start, "cost": -cost / start,
                     "traded": traded / start, "gross": gross / start, "entries": len(opening), "planned_bars": int(planned),
                     "exits": len(closing), "kept": len(extend),
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
