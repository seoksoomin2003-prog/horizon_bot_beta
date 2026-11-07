"""Portfolio simulator for the registered 4h rules. Order within a bar: funding, stops gapped at the open, signal exits,
entries, stops inside the bar, then the close (trailing update and new signals, which act at the next open)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from research.h4_rules import Rule, Signals

BAR_MS = 4 * 3_600_000
WEEK_BARS = 42
RISK = 0.0075
MAX_POSITIONS = 3
MAX_NOTIONAL = 3.0
SIDE_COST = 0.0010


@dataclass
class Market:
    symbol: str
    bars: np.ndarray
    funding: np.ndarray
    signals: Signals
    index: dict = field(init=False)
    funding_by_bar: dict = field(init=False)

    def __post_init__(self):
        self.index = {int(t): i for i, t in enumerate(self.bars[:, 0])}
        self.funding_by_bar = {}
        for ts, rate in self.funding:
            key = int(ts) // BAR_MS * BAR_MS
            self.funding_by_bar[key] = self.funding_by_bar.get(key, 0.0) + float(rate)


@dataclass
class Position:
    symbol: str
    side: int
    qty: float
    entry: float
    stop: float
    entry_ms: int
    entry_bar: int
    pnl: float


def run(rule: Rule, markets: list[Market], start_ms: int, end_ms: int, side_cost: float = SIDE_COST, initial: float = 1000.0) -> dict:
    times = sorted({t for m in markets for t in m.index if start_ms <= t < end_ms})
    cash = initial
    held: dict[str, Position] = {}
    pending_exit: set[str] = set()
    pending_entry: dict[str, tuple] = {}
    closed, curve = [], []
    skipped = 0
    prices: dict[str, float] = {}

    def close_position(pos: Position, price: float, ts: int, reason: str) -> None:
        nonlocal cash
        cost = pos.qty * price * side_cost
        gross = pos.side * pos.qty * (price - pos.entry)
        cash += gross - cost
        pos.pnl += gross - cost
        closed.append({"symbol": pos.symbol, "side": pos.side, "entry_ms": pos.entry_ms, "exit_ms": ts, "entry": pos.entry,
                       "exit": price, "qty": pos.qty, "pnl": pos.pnl, "reason": reason})
        del held[pos.symbol]

    def equity() -> float:
        return cash + sum(p.side * p.qty * (prices[s] - p.entry) for s, p in held.items())

    for ts in times:
        rows = {m.symbol: (m, m.index[ts]) for m in markets if ts in m.index}
        for symbol, (m, i) in rows.items():
            prices[symbol] = m.bars[i, 1]
        # Funding belongs to whoever held through the settlement at the start of this bar.
        for symbol, pos in held.items():
            if symbol in rows:
                rate = rows[symbol][0].funding_by_bar.get(ts, 0.0)
                paid = pos.side * pos.qty * prices[symbol] * rate
                cash -= paid
                pos.pnl -= paid
        for symbol, pos in list(held.items()):
            if symbol in rows:
                open_px = prices[symbol]
                if (pos.side == 1 and open_px <= pos.stop) or (pos.side == -1 and open_px >= pos.stop):
                    close_position(pos, open_px, ts, "stop_gap")
                elif symbol in pending_exit:
                    close_position(pos, open_px, ts, "signal")
        pending_exit -= set(rows)
        for m in markets:
            symbol = m.symbol
            if symbol not in rows or symbol not in pending_entry:
                continue
            side, atr = pending_entry.pop(symbol)
            if symbol in held:
                continue
            open_px = prices[symbol]
            distance = rule.stop_atr * atr
            eq = equity()
            qty = RISK * eq / distance if distance > 0 else 0.0
            exposure = sum(p.qty * prices[s] for s, p in held.items())
            if qty <= 0 or len(held) >= MAX_POSITIONS or exposure + qty * open_px > MAX_NOTIONAL * eq:
                skipped += 1
                continue
            cost = qty * open_px * side_cost
            cash -= cost
            held[symbol] = Position(symbol, side, qty, open_px, open_px - side * distance, ts, rows[symbol][1], -cost)
        pending_entry = {s: v for s, v in pending_entry.items() if s not in rows}
        for symbol, pos in list(held.items()):
            if symbol not in rows:
                continue
            m, i = rows[symbol]
            if pos.side == 1 and m.bars[i, 3] <= pos.stop:
                close_position(pos, pos.stop, ts, "stop")
            elif pos.side == -1 and m.bars[i, 2] >= pos.stop:
                close_position(pos, pos.stop, ts, "stop")
        for m in markets:
            symbol = m.symbol
            if symbol not in rows:
                continue
            i = rows[symbol][1]
            prices[symbol] = m.bars[i, 4]
            s = m.signals
            pos = held.get(symbol)
            leaving = False
            if pos is not None:
                if rule.trailing and not math.isnan(s.atr[i]):
                    level = m.bars[i, 4] - pos.side * rule.stop_atr * s.atr[i]
                    pos.stop = max(pos.stop, level) if pos.side == 1 else min(pos.stop, level)
                aged = rule.max_bars is not None and i - pos.entry_bar + 1 >= rule.max_bars
                leaving = aged or (s.exit_long[i] if pos.side == 1 else s.exit_short[i])
                if leaving:
                    pending_exit.add(symbol)
            if (pos is None or leaving) and not math.isnan(s.atr[i]):
                if s.enter_long[i]:
                    pending_entry[symbol] = (1, s.atr[i])
                elif s.enter_short[i]:
                    pending_entry[symbol] = (-1, s.atr[i])
        curve.append((ts + BAR_MS, equity()))
    for symbol, pos in list(held.items()):
        close_position(pos, prices[symbol], times[-1] + BAR_MS, "end")
    if curve:
        curve[-1] = (curve[-1][0], cash)
    return {"curve": curve, "positions": closed, "skipped": skipped, "cash": cash, "initial": initial}


def weekly_returns(curve: list[tuple], initial: float) -> list[float]:
    values = [initial] + [v for _, v in curve]
    points = values[::WEEK_BARS]
    if points[-1] != values[-1]:
        points.append(values[-1])
    return [b / a - 1 for a, b in zip(points, points[1:])]


def sharpe(weeks: list[float]) -> float:
    if len(weeks) < 2 or np.std(weeks, ddof=1) == 0:
        return 0.0
    return float(np.mean(weeks) / np.std(weeks, ddof=1) * math.sqrt(52))


def block_interval(weeks: list[float], block: int = 4, rounds: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% interval of the mean weekly return, resampling runs of 4 weeks so that nearby weeks stay together."""
    rng = random.Random(seed)
    n, means = len(weeks), []
    for _ in range(rounds):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(0, n - block + 1)
            sample.extend(weeks[start:start + block])
        means.append(sum(sample[:n]) / n)
    means.sort()
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def max_drawdown(curve: list[tuple], initial: float) -> float:
    peak, worst = initial, 0.0
    for _, v in curve:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst


def year_log_returns(curve: list[tuple], initial: float) -> dict:
    """Log return per calendar year and the months of data in it. Log returns add up, so a year can be removed from the total."""
    import time as _time
    out: dict = {}
    previous = initial
    for ts, v in curve:
        year = _time.gmtime((ts - 1) / 1000).tm_year
        row = out.setdefault(year, {"log": 0.0, "bars": 0})
        row["log"] += math.log(v / previous)
        row["bars"] += 1
        previous = v
    for row in out.values():
        row["months"] = row["bars"] * 4 / 24 / 30.44
    return out


def benchmark(markets: list[Market], start_ms: int, end_ms: int, side_cost: float = SIDE_COST, initial: float = 1000.0) -> list[tuple]:
    """Equal weight in every listed coin, rebalanced weekly at the open. Funding left out, which favors the benchmark."""
    times = sorted({t for m in markets for t in m.index if start_ms <= t < end_ms})
    value, curve, weights = initial, [], {}
    for n, ts in enumerate(times):
        live = [m for m in markets if ts in m.index]
        if n % WEEK_BARS == 0:
            target = {m.symbol: 1 / len(live) for m in live}
            traded = sum(abs(target.get(s, 0.0) - weights.get(s, 0.0)) for s in set(target) | set(weights))
            value *= 1 - traded * side_cost
            weights = target
        grown = 0.0
        for m in live:
            if m.symbol in weights:
                i = m.index[ts]
                grown += weights[m.symbol] * m.bars[i, 4] / m.bars[i, 1]
        missing = sum(w for s, w in weights.items() if s not in {m.symbol for m in live})
        total = grown + missing
        weights = {m.symbol: weights[m.symbol] * m.bars[m.index[ts], 4] / m.bars[m.index[ts], 1] / total for m in live if m.symbol in weights} | \
                  {s: w / total for s, w in weights.items() if s not in {m.symbol for m in live}}
        value *= total
        curve.append((ts + BAR_MS, value))
    return curve
