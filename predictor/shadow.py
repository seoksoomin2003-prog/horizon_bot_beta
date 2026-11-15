"""v15 shadow book: the quant architecture run forward on live prices, with no orders.

See docs/research/direction_preregistration_v15.md and docs/QUANT_ARCHITECTURE_KO.md.

At every 4h close the predictor calls cycle(). It
  1. marks the book held since the last recorded close to this close's prices and charges the funding paid,
  2. on time (within MAX_LATE_MS of the close) and with the model ready, computes the alpha of every universe coin
     (research/alpha.py), its risk (research/risk.py) and new target weights (research/portfolio.py), and charges the
     cost of the trades,
  3. scores the alphas recorded HORIZON bars ago against what followed (rank IC).

Files in runtime/predictor/: shadow_state.json (the book now), shadow_book.jsonl (one row per close), shadow_alpha.jsonl
(the alphas behind each rebalance) and shadow_ic.jsonl (their scores). Money here is a virtual 1,000 units.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import alpha, portfolio, risk
from research.h4_data import BAR_MS, DEV_START_MS

VERSION = 15
NAMES = ["carry", "low_vol", "low_beta"]     # kept by the exploration rule (docs/research/quant_explore.md)
HORIZON = 42                                  # 7 days
COUNT = 30
LAMBDA = 1.0                                  # chosen by the exploration's volatility rule
SIDE_COST = 0.0010
START_EQUITY = 1000.0
MAX_LATE_MS = 15 * 60_000
TRAIN_START = DEV_START_MS
NOT_TRADED = {"BTCDOMUSDT"}


def append(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def month_start(ms: int) -> int:
    return int(pd.Timestamp(ms, unit="ms").to_period("M").start_time.value // 1_000_000)


def inputs(market, latest: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The universe rows with scores, and the close, sigma and beta panels, from the market's completed bars."""
    coins = [s for s in market.symbols if s not in NOT_TRADED and len(market.bars.get(s, []))]
    bars = {s: market.bars[s] for s in coins}
    member_rows = alpha.members(bars, TRAIN_START, latest + BAR_MS, COUNT)
    first = int(min(b[0, 0] for b in bars.values())) + BAR_MS
    grid = np.arange(first, latest + 1, BAR_MS)
    closes = pd.DataFrame({s: pd.Series(b[:, 4], index=b[:, 0].astype(np.int64) + BAR_MS).reindex(grid) for s, b in bars.items()},
                          index=grid)
    rets = risk.returns(closes)
    sig = risk.sigma(rets)
    bet = risk.beta(rets, risk.market_return(rets, alpha.membership(member_rows, grid, coins)))
    frames = market.frames(HORIZON)
    data = alpha.prepare(frames[frames["symbol"].isin(coins) & (frames["close_ms"] <= latest)], member_rows, sig, bet)
    return alpha.add_scores(data, NAMES), closes, sig, bet


def funding_between(rates: np.ndarray, start: int, end: int) -> float:
    """Funding rates with start < time <= end, summed."""
    if not len(rates):
        return 0.0
    t = rates[:, 0]
    return float(rates[(t > start) & (t <= end), 1].sum())


class ShadowBook:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.model = alpha.AlphaModel(NAMES, HORIZON)

    def state(self) -> dict | None:
        try:
            return json.loads((self.dir / "shadow_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, state: dict) -> None:
        tmp = self.dir / "shadow_state.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.dir / "shadow_state.json")

    def cycle(self, market, now_ms: int, clock=time.time) -> str:
        latest = now_ms // BAR_MS * BAR_MS
        state = self.state()
        if state and state["last_close_ms"] >= latest:
            return "shadow book already at this close"
        data, closes, sig, bet = inputs(market, latest)
        if latest not in closes.index:
            return "shadow book: no completed bar for this close yet"
        prices = closes.loc[latest]
        made = int(clock() * 1000)
        self.score(data, latest)

        equity = START_EQUITY if state is None else state["equity"]
        held = {} if state is None else dict(state["weights"])
        parts = {"long": 0.0, "short": 0.0, "funding": 0.0, "ret": 0.0}
        if held:
            coins = list(held)
            w = np.array([held[s] for s in coins])
            last = np.array([state["prices"][s] for s in coins])
            now = np.array([prices.get(s, np.nan) for s in coins], dtype=float)
            move = np.where(np.isfinite(now), now / last - 1, 0.0)
            rate = np.array([funding_between(market.funding.get(s, np.empty((0, 2))), state["last_close_ms"], latest) for s in coins])
            parts, w = portfolio.mark(w, move, rate)
            equity *= 1 + parts["ret"]
            held = dict(zip(coins, w))
            last_price = {s: (float(p) if np.isfinite(p) else state["prices"][s]) for s, p in zip(coins, now)}
        else:
            last_price = {}

        traded, rebalanced, why = 0.0, False, ""
        if made - latest > MAX_LATE_MS:
            why = "late"
        else:
            if self.model.month != month_start(latest):
                try:
                    self.model.fit(data, month_start(latest))
                except ValueError as e:
                    why = f"model not ready: {e}"
            if self.model.month is not None and not why:
                rows = data[data["close_ms"] == latest].dropna(subset=self.model.columns + ["sigma"])
                rows = rows[rows["symbol"].map(lambda s: np.isfinite(prices.get(s, np.nan)))]
                if len(rows) < alpha.MIN_COINS:
                    why = f"only {len(rows)} universe coins at this close"
                else:
                    a = pd.Series(self.model.predict(rows), index=rows["symbol"].to_numpy())
                    coins = sorted(set(a.index) | set(held))
                    w0 = np.array([held.get(s, 0.0) for s in coins])
                    s_now, b_now = sig.loc[latest], bet.loc[latest]
                    want = portfolio.optimize(np.array([a.get(s, np.nan) for s in coins]),
                                              np.array([s_now.get(s, np.nan) ** 2 * HORIZON for s in coins]),
                                              np.array([b_now.get(s, np.nan) for s in coins]), w0, SIDE_COST, LAMBDA)
                    priced = np.array([np.isfinite(prices.get(s, np.nan)) for s in coins])
                    want = np.where(priced, want, w0)          # a coin without a price now cannot be traded
                    traded = float(np.abs(want - w0).sum())
                    equity *= 1 - traded * SIDE_COST
                    held = {s: float(v) for s, v in zip(coins, want)}
                    last_price.update({s: float(prices[s]) for s in coins if np.isfinite(prices.get(s, np.nan))})
                    append(self.dir / "shadow_alpha.jsonl", [{"close_ms": latest, "symbol": s, "alpha": float(v),
                                                               "weight": held.get(s, 0.0), "version": VERSION} for s, v in a.items()])
                    rebalanced = True
        held = {s: v for s, v in held.items() if abs(v) > 1e-12}
        b_now = bet.loc[latest]
        market_move = None                  # the universe's equal-weight move since the last close, for the realized beta (S4)
        if state is not None and state["last_close_ms"] in closes.index:
            universe_now = data.loc[data["close_ms"] == latest, "symbol"]
            moves = (prices / closes.loc[state["last_close_ms"]] - 1)[universe_now]
            market_move = float(moves.mean()) if moves.notna().any() else None
        row = {"close_ms": latest, "equity": equity, "ret": equity / (START_EQUITY if state is None else state["equity"]) - 1,
               "long": parts["long"], "short": parts["short"], "funding": parts["funding"], "cost": -traded * SIDE_COST,
               "traded": traded, "gross": float(sum(abs(v) for v in held.values())), "net": float(sum(held.values())),
               "positions": len(held), "beta": float(sum(v * b_now.get(s, 0.0) for s, v in held.items() if np.isfinite(b_now.get(s, np.nan)))),
               "market": market_move, "rebalanced": rebalanced, "note": why, "model_month": self.model.month, "version": VERSION, "made_ms": made}
        append(self.dir / "shadow_book.jsonl", [row])
        self.save({"equity": equity, "weights": held, "prices": {s: last_price[s] for s in held},
                   "last_close_ms": latest, "started_ms": latest if state is None else state["started_ms"], "version": VERSION})
        return (f"shadow book {'rebalanced' if rebalanced else 'marked'}: equity {equity:.2f}, {len(held)} positions"
                + (f" ({why})" if why else ""))

    def score(self, data: pd.DataFrame, latest: int) -> None:
        """Rank IC of the alphas of every rebalance whose HORIZON has ended, against the target that followed."""
        done = {r["close_ms"] for r in read_jsonl(self.dir / "shadow_ic.jsonl")}
        recorded = pd.DataFrame(read_jsonl(self.dir / "shadow_alpha.jsonl"))
        if recorded.empty:
            return
        ready = recorded[(recorded["close_ms"] + HORIZON * BAR_MS <= latest) & ~recorded["close_ms"].isin(done)]
        if ready.empty:
            return
        answers = data[["symbol", "close_ms", "target"]].dropna()
        both = ready.merge(answers, on=["symbol", "close_ms"])
        rows = []
        for t, g in both.groupby("close_ms"):
            if len(g) < alpha.MIN_COINS:
                continue
            k = max(int(len(g) * 0.1 + 0.5), 1)
            order = g.sort_values("alpha")
            rows.append({"close_ms": int(t), "ic": float(g["alpha"].rank().corr(g["target"].rank())), "coins": len(g),
                         "spread": float(order["target"].iloc[-k:].mean() - order["target"].iloc[:k].mean()), "version": VERSION})
        append(self.dir / "shadow_ic.jsonl", rows)
