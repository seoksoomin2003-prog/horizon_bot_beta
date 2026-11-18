"""v16: trade one coin at a time, the one the score ranks first, with buy and sell timing rules.

See docs/research/direction_preregistration_v16.md. Everything works on [time x coin] arrays on the 4h close grid:
row t of open/high/low/close is the bar that ends at times[t]; sigma[t] and alpha[t] are known at that close;
funding[t] is the funding paid over (times[t], times[t] + 4h], i.e. while holding through bar t + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAKER_COST = 0.0002        # limit order, the exchange's base fee per side
TAKER_COST = 0.0010        # market order, fee and slippage per side
FILL_GAP = 0.0005          # v7 stricter fill: the low must pass the limit by this much


@dataclass(frozen=True)
class Rules:
    limit_k: float | None = 0.25        # buy limit at close * (1 - limit_k * sigma); None buys at the close (market)
    wait_bars: int = 2                  # an unfilled limit is cancelled after this many bars
    hold_bars: int = 42                 # sell at the close this many bars after the entry ...
    stop_k: float | None = 2.0          # ... or at entry * (1 - stop_k * sigma * sqrt(hold_bars)); None: no stop
    score_exit: bool = True             # ... or at a close where the coin is in the bottom half of the scores
    min_edge: float | None = MAKER_COST + TAKER_COST   # no trade when the best alpha at the close is below this


RULES_A = Rules()
RULES_B = Rules(limit_k=None, stop_k=None, score_exit=False, min_edge=None)


@dataclass
class Panel:
    times: np.ndarray
    symbols: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    sigma: np.ndarray
    funding: np.ndarray
    alpha: np.ndarray          # NaN outside the universe or before a score exists


@dataclass
class Result:
    equity: np.ndarray                      # at every close from start to end
    trades: list[dict] = field(default_factory=list)
    orders: int = 0
    fills: int = 0
    bars_in: int = 0


def top_picker(panel: Panel):
    def pick(t: int) -> int | None:
        row = panel.alpha[t]
        return None if np.isnan(row).all() else int(np.nanargmax(row))
    return pick


def bottom_picker(panel: Panel):
    def pick(t: int) -> int | None:
        row = panel.alpha[t]
        return None if np.isnan(row).all() else int(np.nanargmin(row))
    return pick


def random_picker(panel: Panel, seed: int):
    rng = np.random.default_rng(seed)

    def pick(t: int) -> int | None:
        ok = np.flatnonzero(np.isfinite(panel.alpha[t]))
        return None if not len(ok) else int(rng.choice(ok))
    return pick


def permuted(panel: Panel, months: np.ndarray, seed: int) -> Panel:
    """The null: every month, the scores are handed to the month's universe coins in a random order. The rules, the
    scores' spread and their persistence within the month are kept; only the link between a score and its coin is cut."""
    rng = np.random.default_rng(seed)
    a = np.full_like(panel.alpha, np.nan)
    for m in np.unique(months):
        rows = np.flatnonzero(months == m)
        coins = np.flatnonzero(np.isfinite(panel.alpha[rows]).any(axis=0))
        if len(coins):
            a[np.ix_(rows, rng.permutation(coins))] = panel.alpha[np.ix_(rows, coins)]
    return Panel(panel.times, panel.symbols, panel.open, panel.high, panel.low, panel.close, panel.sigma, panel.funding, a)


def bottom_half(panel: Panel, t: int, j: int) -> bool:
    """True when coin j is out of the universe at close t or its score is in the lower half."""
    row = panel.alpha[t]
    if not np.isfinite(row[j]):
        return True
    n = int(np.isfinite(row).sum())
    rank = int((row[np.isfinite(row)] < row[j]).sum()) + 1          # 1 = lowest
    return rank <= n / 2


def simulate(panel: Panel, pick, rules: Rules, start: int, end: int, cost: float = 1.0, initial: float = 1000.0) -> Result:
    """Runs the rules over closes start .. end (indices into panel.times). One coin at a time, long only, 1x."""
    maker, taker = MAKER_COST * cost, TAKER_COST * cost
    O, H, L, C, S, F, A = panel.open, panel.high, panel.low, panel.close, panel.sigma, panel.funding, panel.alpha
    equity = initial
    curve = [equity]
    res = Result(equity=np.empty(0))
    j = None                  # coin held
    entry = last = stop = 0.0
    held = 0
    opened = start
    pending = None            # (coin, limit price, expiry index, decision index)
    start_equity = equity

    def close_trade(t: int, price: float, why: str) -> None:
        nonlocal equity, j
        equity *= price / last
        equity *= 1 - taker
        res.trades.append({"symbol": panel.symbols[j], "entry_ms": int(panel.times[opened]), "exit_ms": int(panel.times[t]),
                           "entry": entry, "exit": price, "why": why, "net": equity / start_equity - 1, "bars": held})
        j = None

    for t in range(start, end):
        # decisions at close t
        if j is not None:
            if rules.score_exit and bottom_half(panel, t, j):
                close_trade(t, C[t, j], "score")
            elif held >= rules.hold_bars:
                if pick(t) == j:
                    held = 0
                else:
                    close_trade(t, C[t, j], "time")
        if j is None and pending is None:
            c = pick(t)
            best = np.nanmax(A[t]) if np.isfinite(A[t]).any() else np.nan
            if c is not None and (rules.min_edge is None or best >= rules.min_edge) and np.isfinite(C[t, c]):
                res.orders += 1
                if rules.limit_k is None:
                    res.fills += 1
                    start_equity = equity
                    equity *= 1 - taker
                    j, entry, last, held, opened = c, C[t, c], C[t, c], 0, t
                    stop = entry * (1 - rules.stop_k * S[t, c] * np.sqrt(rules.hold_bars)) if rules.stop_k is not None else -np.inf
                else:
                    pending = (c, C[t, c] * (1 - rules.limit_k * S[t, c]), t + rules.wait_bars, t)
        # the next bar
        u = t + 1
        if pending is not None:
            c, limit, expiry, decided = pending
            if np.isfinite(L[u, c]) and L[u, c] <= limit * (1 - FILL_GAP):
                res.fills += 1
                pending = None
                start_equity = equity
                equity *= 1 - maker
                j, entry, last, held, opened = c, limit, limit, 1, u
                stop = entry * (1 - rules.stop_k * S[decided, c] * np.sqrt(rules.hold_bars)) if rules.stop_k is not None else -np.inf
                equity *= 1 - (F[t, c] if np.isfinite(F[t, c]) else 0.0)
                if L[u, c] <= stop:                       # the order of the bar's moves is unknown: assume the worse
                    close_trade(u, stop, "stop")
                else:
                    equity *= C[u, c] / last
                    last = C[u, c]
            elif u >= expiry:
                pending = None
        elif j is not None:
            res.bars_in += 1
            held += 1
            if np.isfinite(C[u, j]):
                equity *= 1 - (F[t, j] if np.isfinite(F[t, j]) else 0.0)
                if L[u, j] <= stop:
                    close_trade(u, min(O[u, j], stop), "stop")
                else:
                    equity *= C[u, j] / last
                    last = C[u, j]
        curve.append(equity)
    if j is not None:                                     # still holding at the end: marked, not sold
        res.trades.append({"symbol": panel.symbols[j], "entry_ms": int(panel.times[opened]), "exit_ms": int(panel.times[end]),
                           "entry": entry, "exit": last, "why": "open", "net": equity / start_equity - 1, "bars": held})
    res.equity = np.array(curve)
    return res


def summary(panel: Panel, res: Result, start: int, end: int) -> dict:
    eq = res.equity
    times = panel.times[start:end + 1]
    years = {}
    year = np.array([np.datetime64(int(t), "ms").astype("datetime64[Y]").astype(int) + 1970 for t in times])
    for y in np.unique(year):
        idx = np.flatnonzero(year == y)
        first = max(idx[0] - 1, 0)
        years[int(y)] = float(eq[idx[-1]] / eq[first] - 1)
    nets = np.array([tr["net"] for tr in res.trades if tr["why"] != "open"])
    why = {}
    for tr in res.trades:
        why[tr["why"]] = why.get(tr["why"], 0) + 1
    return {"total": float(eq[-1] / eq[0] - 1), "by_year": years,
            "max_drawdown": float((1 - eq / np.maximum.accumulate(eq)).max()),
            "trades": len(nets), "orders": res.orders, "fill_rate": res.fills / res.orders if res.orders else None,
            "exits": why, "trade_mean": float(nets.mean()) if len(nets) else None,
            "trade_ci": [float(nets.mean() - 1.96 * nets.std(ddof=1) / np.sqrt(len(nets))),
                         float(nets.mean() + 1.96 * nets.std(ddof=1) / np.sqrt(len(nets)))] if len(nets) > 1 else None,
            "win_rate": float((nets > 0).mean()) if len(nets) else None,
            "time_in_market": res.bars_in / max(end - start, 1)}


def hold(panel: Panel, start: int, end: int, coin: str | None = None) -> np.ndarray:
    """Equity of holding one coin (or the universe, equal weight, rebalanced every bar), no costs."""
    if coin is not None:
        j = panel.symbols.index(coin)
        c = panel.close[start:end + 1, j]
        return c / c[0]
    curve = [1.0]
    for t in range(start, end):
        inside = np.isfinite(panel.alpha[t]) & np.isfinite(panel.close[t]) & np.isfinite(panel.close[t + 1])
        r = panel.close[t + 1, inside] / panel.close[t, inside] - 1
        curve.append(curve[-1] * (1 + (r.mean() if len(r) else 0.0)))
    return np.array(curve)
