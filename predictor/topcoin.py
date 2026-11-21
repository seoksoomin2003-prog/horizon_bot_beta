"""v18 live book: the top coin by the v15 score, long or flat, on live prices, with no orders.

See docs/research/direction_preregistration_v18.md. The rule is research/timed.py's RULES_C with no market forecast
(mu = alpha): at every 4h close
  1. mark the coin held since the last recorded close to this close's price and charge the funding paid,
  2. on time (within MAX_LATE_MS of the close) and with the model ready, score every universe coin,
     S = (alpha - ROUND_TRIP) / sigma, then sell the coin held if its alpha is below 0 or it left the universe, switch
     to the top coin if its alpha beats the held one's by more than SWITCH_GAP, or, holding nothing, buy the top coin
     when its S is above 0, with equity * min(1, TARGET_BAR_VOL / sigma),
  3. record the book, the trades and every coin's score and status (for the dashboard's per-coin table).

Files in runtime/predictor/: topcoin_state.json, topcoin_book.jsonl, topcoin_trades.jsonl, topcoin_scores.jsonl.
Money here is a virtual 1,000 units.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import shadow
from predictor.shadow import append, funding_between, month_start
from research import alpha
from research.h4_data import BAR_MS

VERSION = 18
SIDE_COST = 0.0010
ROUND_TRIP = 2 * SIDE_COST
SWITCH_GAP = 2 * ROUND_TRIP
TARGET_BAR_VOL = 0.5 / np.sqrt(6 * 365)      # 50% a year, per 4h bar
START_EQUITY = 1000.0
MAX_LATE_MS = shadow.MAX_LATE_MS
BENCHMARK = "BTCUSDT"


class TopCoinBook:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.model = alpha.AlphaModel(shadow.NAMES, shadow.HORIZON)

    def state(self) -> dict | None:
        try:
            return json.loads((self.dir / "topcoin_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, state: dict) -> None:
        tmp = self.dir / "topcoin_state.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.dir / "topcoin_state.json")

    def cycle(self, market, now_ms: int, clock=time.time) -> str:
        latest = now_ms // BAR_MS * BAR_MS
        state = self.state()
        if state and state["last_close_ms"] >= latest:
            return "top-coin book already at this close"
        data, closes, sig, _ = shadow.inputs(market, latest)
        if latest not in closes.index:
            return "top-coin book: no completed bar for this close yet"
        prices = closes.loc[latest]
        made = int(clock() * 1000)
        equity = START_EQUITY if state is None else state["equity"]
        pos = None if state is None else state.get("position")

        # 1. mark
        exposure = abs(pos["q"]) * pos["last"] / equity if pos and equity > 0 else 0.0     # over the bar just ended
        funding = 0.0
        if pos:
            now = prices.get(pos["symbol"], np.nan)
            rate = funding_between(market.funding.get(pos["symbol"], np.empty((0, 2))), state["last_close_ms"], latest)
            funding = -rate * abs(pos["q"]) * pos["last"]
            equity += funding
            if np.isfinite(now):
                equity += pos["q"] * (now - pos["last"])
                pos["last"] = float(now)
        universe_now = data.loc[data["close_ms"] == latest, "symbol"]
        market_move = None
        if state is not None and state["last_close_ms"] in closes.index:
            moves = (prices / closes.loc[state["last_close_ms"]] - 1)[universe_now]
            market_move = float(moves.mean()) if moves.notna().any() else None

        # 2. decide
        why, trades, scores = "", [], []
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
                rows = rows[rows["symbol"].map(lambda s: np.isfinite(prices.get(s, np.nan))) & (rows["sigma"] > 0)]
                if len(rows) < alpha.MIN_COINS:
                    why = f"only {len(rows)} universe coins at this close"
                else:
                    a = pd.Series(self.model.predict(rows), index=rows["symbol"].to_numpy())
                    s = pd.Series(rows["sigma"].to_numpy(float), index=a.index)
                    score = (a - ROUND_TRIP) / s
                    top = str(score.idxmax())

                    def sell(reason: str) -> None:
                        nonlocal equity, pos
                        price = prices.get(pos["symbol"], np.nan)
                        price = float(price) if np.isfinite(price) else pos["last"]
                        equity -= abs(pos["q"]) * price * SIDE_COST
                        trades.append({"close_ms": latest, "action": "sell", "symbol": pos["symbol"], "price": price, "why": reason,
                                       "entry_ms": pos["entry_ms"], "entry": pos["entry"], "net": equity / pos["start_equity"] - 1,
                                       "equity": equity, "version": VERSION})
                        pos = None

                    if pos:
                        held = pos["symbol"]
                        if held not in a.index:
                            sell("universe")
                        elif a[held] < 0:
                            sell("signal")
                        elif top != held and a[top] - a[held] > SWITCH_GAP:
                            sell("switch")
                    if pos is None and score[top] > 0 and equity > 0:
                        notional = equity * min(1.0, TARGET_BAR_VOL / s[top])
                        start_equity = equity
                        equity -= notional * SIDE_COST
                        price = float(prices[top])
                        pos = {"symbol": top, "q": notional / price, "entry": price, "last": price, "entry_ms": latest,
                               "start_equity": start_equity}
                        trades.append({"close_ms": latest, "action": "buy", "symbol": top, "price": price, "size": notional / start_equity,
                                       "alpha": float(a[top]), "score": float(score[top]), "equity": equity, "version": VERSION})
                    held = pos["symbol"] if pos else None
                    order = score.sort_values(ascending=False)
                    for rank, (sym, sc) in enumerate(order.items(), 1):
                        # held / top (first, S > 0, not held: switch gap too small) / positive (S > 0) / wait (S <= 0)
                        status = "held" if sym == held else "wait" if sc <= 0 else "top" if sym == top else "positive"
                        scores.append({"close_ms": latest, "symbol": sym, "rank": rank, "alpha": float(a[sym]), "sigma": float(s[sym]),
                                       "score": float(sc), "size": float(min(1.0, TARGET_BAR_VOL / s[sym])), "status": status,
                                       "switch_gap": None if held is None or sym == held else float(a[sym] - a[held] - SWITCH_GAP),
                                       "version": VERSION})
        append(self.dir / "topcoin_trades.jsonl", trades)
        append(self.dir / "topcoin_scores.jsonl", scores)

        # 3. record
        btc = prices.get(BENCHMARK, np.nan)
        previous = START_EQUITY if state is None else state["equity"]
        row = {"close_ms": latest, "equity": equity, "ret": equity / previous - 1 if previous > 0 else 0.0,
               "funding": funding, "exposure_before": exposure, "market": market_move,
               "bench_ret": exposure * market_move if market_move is not None else 0.0,
               "symbol": pos["symbol"] if pos else None, "exposure": abs(pos["q"]) * pos["last"] / equity if pos and equity > 0 else 0.0,
               "btc": float(btc) if np.isfinite(btc) else None, "trades": len(trades), "note": why,
               "model_month": self.model.month, "version": VERSION, "made_ms": made}
        append(self.dir / "topcoin_book.jsonl", [row])
        self.save({"equity": equity, "position": pos, "last_close_ms": latest,
                   "started_ms": latest if state is None else state["started_ms"], "version": VERSION})
        held_text = f"holding {pos['symbol']} ({row['exposure']:.0%})" if pos else "flat"
        done = ", ".join(f"{t['action']} {t['symbol']}" for t in trades)
        return f"top-coin book: equity {equity:.2f}, {held_text}" + (f", {done}" if done else "") + (f" ({why})" if why else "")
