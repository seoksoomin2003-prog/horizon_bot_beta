"""Weekly long-short simulator for docs/research/xs_preregistration_v2.md. Parameters are fixed there; do not tune."""

from __future__ import annotations

import numpy as np

from research.h4_sim import BAR_MS

WEEK_MS = 7 * 86_400_000
WEEK_BARS = 42
UNIVERSE = ["SOLUSDT", "XRPUSDT", "AVAXUSDT", "DOGEUSDT", "BNBUSDT", "SUIUSDT", "ADAUSDT"]
LEGS = 2
LEG_WEIGHT = 0.25
MIN_AGE_BARS = 168          # 28 days listed before a coin is ranked
SIDE_COST = 0.0010
RULES = {"X1": "momentum_28d", "X2": "reversal_7d", "X3": "funding_carry_7d"}


def mondays(start_ms: int, end_ms: int) -> list[int]:
    """Mondays 00:00 UTC whose whole week ends by end_ms. 1970-01-05 was a Monday."""
    first = start_ms + (-(start_ms - 4 * 86_400_000)) % WEEK_MS
    return list(range(first, end_ms - WEEK_MS + 1, WEEK_MS))


class Coin:
    def __init__(self, symbol: str, bars: np.ndarray, funding: np.ndarray):
        self.symbol, self.bars, self.funding = symbol, bars, funding
        self.index = {int(t): i for i, t in enumerate(bars[:, 0])}

    def week(self, ws: int):
        """Bar index of the Monday open, if the coin has the whole week and enough history."""
        i = self.index.get(ws)
        if i is None or i < MIN_AGE_BARS + 1 or i + WEEK_BARS > len(self.bars):
            return None
        if int(self.bars[i + WEEK_BARS - 1, 0]) != ws + WEEK_MS - BAR_MS:
            return None
        return i

    def score(self, key: str, i: int, ws: int) -> float:
        close = self.bars[:, 4]
        if key == "X1":
            return close[i - 1] / close[i - 1 - 168] - 1
        if key == "X2":
            return close[i - 1] / close[i - 1 - 42] - 1
        rates = self.funding[(self.funding[:, 0] >= ws - 7 * 86_400_000) & (self.funding[:, 0] < ws), 1]
        return float(rates.mean()) if len(rates) else float("nan")

    def funding_in(self, ws: int) -> float:
        f = self.funding
        return float(f[(f[:, 0] >= ws) & (f[:, 0] < ws + WEEK_MS), 1].sum())


def pick(key: str, scores: dict) -> tuple[list, list]:
    """(longs, shorts). Ties keep universe order, so the result never depends on dict ordering."""
    ranked = sorted((s for s in scores if scores[s] == scores[s]), key=lambda s: (scores[s], UNIVERSE.index(s)))
    low, high = ranked[:LEGS], ranked[-LEGS:]
    if key == "X1":
        return high, low
    return low, high         # X2 buys the weakest week, X3 buys the lowest funding


def run(key: str, coins: list[Coin], start_ms: int, end_ms: int, side_cost: float = SIDE_COST, initial: float = 1000.0) -> dict:
    equity = initial
    held: dict[str, float] = {}          # signed notional at the last close
    curve, weeks, bench, hits, traded_weeks, ruined_ms = [], [], [], [], 0, None
    for ws in mondays(start_ms, end_ms):
        live = {c.symbol: (c, i) for c in coins if (i := c.week(ws)) is not None}
        scores = {s: c.score(key, i, ws) for s, (c, i) in live.items()}
        start_equity = equity
        target: dict[str, float] = {}
        if len([v for v in scores.values() if v == v]) >= 2 * LEGS:
            longs, shorts = pick(key, scores)
            target = {s: LEG_WEIGHT * equity for s in longs} | {s: -LEG_WEIGHT * equity for s in shorts}
            traded_weeks += 1
        turnover = sum(abs(target.get(s, 0.0) - held.get(s, 0.0)) for s in set(target) | set(held))
        equity -= turnover * side_cost
        paid = sum(n * live[s][0].funding_in(ws) for s, n in target.items())
        rets = {s: c.bars[i + WEEK_BARS - 1, 4] / c.bars[i, 1] - 1 for s, (c, i) in live.items()}
        for k in range(WEEK_BARS):
            mark = sum(n * (live[s][0].bars[live[s][1] + k, 4] / live[s][0].bars[live[s][1], 1] - 1) for s, n in target.items())
            curve.append((ws + (k + 1) * BAR_MS, equity + mark - (paid if k == WEEK_BARS - 1 else 0.0)))
        equity = curve[-1][1]
        ruin = next((t for t, v in curve[-WEEK_BARS:] if v <= 0), None)
        if ruin is not None:
            # The account is gone (a real exchange liquidates first); nothing after this point can be simulated.
            ruined_ms = ruin
            curve = [(t, max(v, 0.0)) for t, v in curve if t <= ruin]
            weeks.append(-1.0)
            equity, held = 0.0, {}
            break
        held = {s: n * (1 + rets[s]) for s, n in target.items()}
        weeks.append(equity / start_equity - 1)
        bench.append(float(np.mean(list(rets.values()))) if rets else 0.0)
        if target:
            long_ret = np.mean([rets[s] for s, n in target.items() if n > 0])
            short_ret = np.mean([rets[s] for s, n in target.items() if n < 0])
            hits.append(long_ret > short_ret)
    # Close everything at the end; the closing cost belongs to the last week.
    equity -= sum(abs(n) for n in held.values()) * side_cost
    if curve and ruined_ms is None:
        curve[-1] = (curve[-1][0], equity)
        weeks[-1] = equity / start_equity - 1
    return {"curve": curve, "weeks": weeks, "bench": bench, "hit_rate": float(np.mean(hits)) if hits else None,
            "traded_weeks": traded_weeks, "cash": equity, "initial": initial, "ruined_ms": ruined_ms}
