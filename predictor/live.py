"""Live direction predictions. No orders, no keys: public prices only.

It also collects data the exchange does not keep (predictor/collect.py) for later research.

Every 4-hour bar close it records, for each coin, the model's probability that the price is higher 4 hours later and
8 hours later (one model per horizon), and marks the "confident" ones: the top 10% and top 5% by the model's own
training-time measure. On coins the model never saw, the top 10% hit about 56% at 4 hours and 57% at 8 hours
(docs/research/direction_result_v7.md). When the horizon ends it records what happened. The dashboard reads both files
and keeps score. A prediction is only recorded within MAX_LATE_MS of the close, so a laptop that slept through a close
skips it instead of predicting after the fact.

It records a second kind of prediction too, "relative": the probability that the coin does better than the median
coin over the same 4 or 8 hours. Besides the direction features it sees where each feature ranks among the coins
(v11). On 46 coins that model never saw, its top 10% hit 57% and 58%, above 55% in every year
(docs/research/direction_result_v11.md). BTCDOMUSDT, an index product, is left out of it. Every row carries the
model's `version`, so a score can leave out predictions made by an earlier model.

Each model refits on the first of every month on rows whose answer was known by then, exactly as in the research.

On request from the dashboard it also predicts "as of now" (predict_now): the same direction models on 4-hour bars
that end at this moment instead of at a scheduled close. The research only ever tested scheduled closes, so these
are shown apart, never mixed into the score, and saved in now_predictions.jsonl so they can be checked later.

  .venv/bin/python -m predictor.live         (normally through ./run_predictor.sh)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.direction import (ABS_MOVE, CLIP, FEATURES, MARKET_FEATURES, RANK_FEATURES, add_ranks, add_relative, fit,  # noqa: E402
                                frame, predict)
from research.h4_data import BAR_MS, DEV_START_MS, SYMBOLS, V6_DIR, V6_SYMBOLS_FILE, WARMUP_START_MS, _pages  # noqa: E402
from predictor.collect import Collector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "runtime" / "predictor"
SIGNAL_GAIN = 0.005             # v4's fixed bet threshold: expected gain above 0.5%
HORIZONS = (1, 2)              # bars ahead: 4 hours and 8 hours
NOT_RELATIVE = {"BTCDOMUSDT"}   # an index product: no part of the median coin (v9)
MAX_LATE_MS = 15 * 60_000
WAKE_AFTER_CLOSE_MS = 60_000
NOW_CANDLE_MS = 15 * 60_000     # "as of now" bars are built from 15-minute candles
NOW_PAGES = 2                   # 3,000 candles: 31 days, enough for the 28-day feature
NOW_MIN_GAP_MS = 60_000         # at most one "as of now" run a minute, whatever is asked


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
        except ValueError:      # a line still being written
            continue
    return out


def write_status(directory: Path, **fields) -> None:
    tmp = directory / "status.tmp"
    tmp.write_text(json.dumps({"updated_ms": int(time.time() * 1000), **fields}), encoding="utf-8")
    os.replace(tmp, directory / "status.json")


class Market:
    """Completed 4h bars and funding rates for every coin, cached on disk and topped up from the exchange."""

    def __init__(self, directory: Path, client, symbols: list[str] = SYMBOLS) -> None:
        self.dir, self.client, self.symbols = directory, client, list(symbols)
        self.bars: dict[str, np.ndarray] = {}
        self.funding: dict[str, np.ndarray] = {}

    def _load(self, name: str, width: int) -> np.ndarray:
        for folder in (self.dir, V6_DIR):       # the research download saves a first-run fetch of six years
            try:
                return np.load(folder / name)
            except OSError:
                continue
        return np.empty((0, width))

    def _top_up(self, name: str, rows: np.ndarray, fetch, now_ms: int, complete) -> np.ndarray:
        since = int(rows[-1, 0]) + 1 if len(rows) else WARMUP_START_MS
        new = [r for r in _pages(fetch, since, now_ms) if complete(r)]
        if new or not (self.dir / name).exists():
            rows = np.vstack([rows, np.array(new, dtype=float).reshape(-1, rows.shape[1])])
            rows = rows[np.unique(rows[:, 0], return_index=True)[1]]
            np.save(self.dir / name, rows)
        return rows

    def update(self, now_ms: int, bars: bool = True) -> None:
        """Tops up the funding rates and, unless `bars` is off, the completed 4h bars."""
        for symbol in self.symbols:
            if symbol not in self.bars:
                self.bars[symbol] = self._load(f"{symbol}.bars.npy", 6)
                self.funding[symbol] = self._load(f"{symbol}.funding.npy", 2)
            klines = lambda c: [[int(r[0])] + [float(v) for v in r[1:6]] for r in
                                self.client.fapiPublicGetKlines({"symbol": symbol, "interval": "4h", "startTime": c, "limit": 1500})]
            rates = lambda c: [[int(r["fundingTime"]), float(r["fundingRate"])] for r in
                               self.client.fapiPublicGetFundingRate({"symbol": symbol, "startTime": c, "limit": 1000})]
            if bars:
                self.bars[symbol] = self._top_up(f"{symbol}.bars.npy", self.bars[symbol], klines, now_ms,
                                                 lambda r: r[0] + BAR_MS <= now_ms)   # completed bars only
            self.funding[symbol] = self._top_up(f"{symbol}.funding.npy", self.funding[symbol], rates, now_ms, lambda r: True)

    def frames(self, horizon: int = 1) -> pd.DataFrame:
        btc = frame("BTCUSDT", self.bars["BTCUSDT"], self.funding["BTCUSDT"], None, horizon).set_index("close_ms")[["r1", "r6"]]
        return pd.concat([frame(s, self.bars[s], self.funding[s], None if s == "BTCUSDT" else btc, horizon)
                          for s in self.symbols], ignore_index=True)


class Model:
    """The research model refit monthly: standardize on the training rows, clip, L2 logistic regression.

    It learns from every coin it predicts (v7 H2: learning from more coins predicted unseen coins better) and keeps
    `thr` and `thr95`, the 90th and 95th percentiles of |p - 0.5| on its training rows: the "confident" cutoffs.

    With `relative` it answers "will the coin beat the median coin?" instead (v9), with two more features and each
    feature's rank among the coins (v11). `version` is the research version that fixed the model.
    """

    def __init__(self, horizon: int = 1, relative: bool = False) -> None:
        self.horizon, self.relative = horizon, relative
        self.kind = "relative" if relative else "direction"
        self.features = FEATURES + MARKET_FEATURES + RANK_FEATURES if relative else FEATURES
        self.version = 11 if relative else 7
        self.month: int | None = None
        self.train_rows = 0

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """The rows this model learns from and predicts: for a relative model, measured against the median coin."""
        if not self.relative:
            return data
        data = data[~data["symbol"].isin(NOT_RELATIVE)]
        coins = set(data["symbol"])
        return add_ranks(add_relative(data, coins), coins)

    def refit_if_needed(self, data: pd.DataFrame, now_ms: int) -> None:
        month = int(pd.Timestamp(now_ms, unit="ms").to_period("M").start_time.value // 1_000_000)
        if month == self.month:
            return
        rows = data.dropna(subset=self.features + ["fwd", "sd42"])
        rows = rows[(rows["close_ms"] >= DEV_START_MS) & (rows["close_ms"] + self.horizon * BAR_MS <= month)]
        x = rows[self.features].to_numpy(float)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.w = fit(self._scale(x), (rows["fwd"] > 0).to_numpy(float))
        self.thr, self.thr95 = (float(v) for v in np.quantile(np.abs(predict(self.w, self._scale(x)) - 0.5), [0.9, 0.95]))
        self.month, self.train_rows = month, len(rows)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -CLIP, CLIP)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return predict(self.w, self._scale(rows[self.features].to_numpy(float)))


def predictions_for(rows: pd.DataFrame, p: np.ndarray, close_ms: int, model: Model, made_ms: int) -> list[dict]:
    out = []
    for (_, r), prob in zip(rows.iterrows(), p):
        gain = abs(2 * prob - 1) * r["sd42"] * math.sqrt(model.horizon) * ABS_MOVE
        out.append({"close_ms": close_ms, "horizon": model.horizon, "kind": model.kind, "symbol": r["symbol"], "price": float(math.exp(r["logc"])),
                    "p_up": float(prob), "expected_gain": float(gain), "signal": bool(gain > SIGNAL_GAIN),
                    "confident": bool(abs(prob - 0.5) >= model.thr), "top5": bool(abs(prob - 0.5) >= model.thr95),
                    "model_month": model.month, "version": model.version, "made_ms": made_ms})
    return out


def outcomes_for(predicted: list[dict], scored: set, data: pd.DataFrame) -> list[dict]:
    """The close at the end of the horizon for every prediction not yet scored whose horizon has ended.
    Rows written before 8-hour predictions existed have no `horizon`: they are 4-hour ones. A relative prediction also
    gets `relative_return`: the coin's return minus the median coin's (`fwd` of the relative model's rows)."""
    indexed = data.set_index(["symbol", "close_ms"])
    close = indexed["logc"]
    out = []
    for p in predicted:
        horizon, kind = p.get("horizon", 1), p.get("kind", "direction")
        key = (p["symbol"], p["close_ms"], horizon, kind)
        nxt = (p["symbol"], p["close_ms"] + horizon * BAR_MS)
        if key in scored or nxt not in close.index or not np.isfinite(close[nxt]):
            continue
        row = {"close_ms": p["close_ms"], "horizon": horizon, "kind": kind, "symbol": p["symbol"],
               "next_price": float(math.exp(close[nxt])), "log_return": float(close[nxt] - math.log(p["price"]))}
        if kind == "relative":
            relative = indexed["fwd"].get(key[:2], np.nan)
            if not np.isfinite(relative):
                continue
            row["relative_return"] = float(relative)
        out.append(row)
    return out


def with_closes(data: pd.DataFrame, bars: dict[str, np.ndarray]) -> pd.DataFrame:
    closes = pd.concat([pd.DataFrame({"symbol": s, "close_ms": b[:, 0].astype(np.int64) + BAR_MS, "logc": np.log(b[:, 4])})
                        for s, b in bars.items()], ignore_index=True)
    return data.merge(closes, on=["symbol", "close_ms"], how="left")


def cycle(market: Market, models: list[Model], directory: Path, now_ms: int, clock=time.time) -> str:
    market.update(now_ms)
    predicted = read_jsonl(directory / "predictions.jsonl")
    scored = {(o["symbol"], o["close_ms"], o.get("horizon", 1), o.get("kind", "direction"))
              for o in read_jsonl(directory / "outcomes.jsonl")}
    latest = now_ms // BAR_MS * BAR_MS
    messages, frames = [], {}
    for model in models:
        if model.horizon not in frames:
            frames[model.horizon] = with_closes(market.frames(model.horizon), market.bars)
        data = model.prepare(frames[model.horizon])
        model.refit_if_needed(data, now_ms)
        mine = [p for p in predicted if p.get("horizon", 1) == model.horizon and p.get("kind", "direction") == model.kind]
        append(directory / "outcomes.jsonl", outcomes_for(mine, scored, data))
        rows = data[data["close_ms"] == latest].dropna(subset=model.features + ["sd42", "logc"])
        made = int(clock() * 1000)
        if latest in {p["close_ms"] for p in mine}:
            messages.append("already predicted")
        elif rows.empty:
            messages.append("no completed bar for this close yet")
        elif made - latest > MAX_LATE_MS:
            messages.append("too late for this close; waiting for the next one")
        else:
            append(directory / "predictions.jsonl", predictions_for(rows, model.predict(rows), latest, model, made))
            messages.append(f"predicted {len(rows)} coins")
    return next((m for m in messages if m.startswith("predicted")), messages[0])     # the first model predicts every coin


def bars_ending_now(candles: np.ndarray) -> np.ndarray:
    """4h bars counted back from the newest candle, so the last bar ends now. Rows in and out are
    [open_ms, open, high, low, close, volume]; the newest candle is still forming and its close is the current price.
    The oldest bar is dropped when it is not whole. A gap in the candles leaves a gap in the bars."""
    c = pd.DataFrame(candles, columns=["t", "o", "h", "l", "c", "v"]).sort_values("t")
    end = int(c["t"].iloc[-1]) + NOW_CANDLE_MS
    c["back"] = (end - 1 - c["t"].astype(np.int64)) // BAR_MS       # 0 is the bar that ends now
    bars = c.groupby("back").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), n=("t", "size"))
    bars = bars[(bars["n"] == BAR_MS // NOW_CANDLE_MS) | (bars.index < bars.index.max())].sort_index(ascending=False)
    return np.column_stack([end - (bars.index.to_numpy() + 1) * BAR_MS, bars[["o", "h", "l", "c", "v"]].to_numpy()])


def predict_now(market: Market, models: list[Model], now_ms: int, clock=time.time) -> dict:
    """Every coin's 4h and 8h direction probability on bars that end now, with its place by confidence (1 = surest)."""
    market.update(now_ms, bars=False)       # the newest funding rates; the bars here come from 15m candles
    bars = {}
    for symbol in market.symbols:
        rows, before = [], None
        for _ in range(NOW_PAGES):
            query = {"symbol": symbol, "interval": "15m", "limit": 1500, **({"endTime": before} if before else {})}
            page = [[int(r[0])] + [float(v) for v in r[1:6]] for r in market.client.fapiPublicGetKlines(query)]
            if not page:
                break
            rows, before = page + rows, page[0][0] - 1
        if rows:
            bars[symbol] = bars_ending_now(np.array(rows, dtype=float))
    if "BTCUSDT" not in bars:
        raise RuntimeError("no BTCUSDT candles")
    end = max(int(b[-1, 0]) + BAR_MS for b in bars.values())
    btc = frame("BTCUSDT", bars["BTCUSDT"], market.funding["BTCUSDT"], None, 1).set_index("close_ms")[["r1", "r6"]]
    data = pd.concat([frame(s, b, market.funding[s], None if s == "BTCUSDT" else btc, 1) for s, b in bars.items()], ignore_index=True)
    data = data[data["close_ms"] == end]
    coins = {s: {"symbol": s, "price": float(b[-1, 4])} for s, b in bars.items() if int(b[-1, 0]) + BAR_MS == end}
    for model in models:
        if model.relative or model.month is None:
            continue
        rows = data.dropna(subset=model.features + ["sd42"])
        p = model.predict(rows)
        suffix = "" if model.horizon == 1 else f"_{4 * model.horizon}h"
        for place, i in enumerate(np.argsort(-np.abs(p - 0.5)), start=1):
            sure = abs(p[i] - 0.5)
            coins[rows["symbol"].iloc[i]].update({f"p_up{suffix}": float(p[i]), f"rank{suffix}": place,
                                                  f"confident{suffix}": bool(sure >= model.thr), f"top5{suffix}": bool(sure >= model.thr95)})
    done = sorted((c for c in coins.values() if "p_up" in c), key=lambda c: c["symbol"])
    if not done:
        raise RuntimeError("the models are not ready yet; they are fitted at the first 4-hour close after starting")
    return {"ok": True, "made_ms": int(clock() * 1000), "end_ms": end, "coins": done}


def serve_now(market: Market, models: list[Model], directory: Path, clock=time.time) -> bool:
    """Answers the dashboard's newest "as of now" request, if one is waiting. True when it wrote an answer."""
    try:
        asked = int(json.loads((directory / "now_request.json").read_text(encoding="utf-8"))["requested_ms"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    try:
        last = json.loads((directory / "now.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        last = {}
    now = int(clock() * 1000)
    if last.get("requested_ms", 0) >= asked or now - last.get("made_ms", 0) < NOW_MIN_GAP_MS:
        return False
    try:
        result = predict_now(market, models, now, clock)
        append(directory / "now_predictions.jsonl", [{**c, "made_ms": result["made_ms"], "end_ms": result["end_ms"]} for c in result["coins"]])
    except Exception as e:      # network or exchange trouble: tell the screen instead of leaving it waiting
        result = {"ok": False, "made_ms": int(clock() * 1000), "message": f"{type(e).__name__}: {str(e)[:200]}"}
    tmp = directory / "now.tmp"
    tmp.write_text(json.dumps({"requested_ms": asked, **result}), encoding="utf-8")
    os.replace(tmp, directory / "now.json")
    return True


def collect_symbols() -> list[str]:
    """The original 10 coins plus the 50 from the v6 research, when that list has been chosen."""
    try:
        return SYMBOLS + json.loads(V6_SYMBOLS_FILE.read_text())["symbols"]
    except (OSError, ValueError, KeyError):
        return list(SYMBOLS)


def main() -> int:
    from replay.market_data import _keyless_client
    DIR.mkdir(parents=True, exist_ok=True)
    client = _keyless_client()
    symbols = collect_symbols()
    market, models = Market(DIR, client, symbols), [Model(h, relative) for relative in (False, True) for h in HORIZONS]
    collector = Collector(DIR, client, symbols)
    collector.start_liquidations()
    print("direction predictor started: no orders, public prices only", flush=True)
    while True:
        now = int(time.time() * 1000)
        try:
            message = cycle(market, models, DIR, now)
            write_status(DIR, ok=True, message=message, model_month=models[0].month, train_rows=models[0].train_rows,
                         collected=collector.counts)
        except Exception as e:     # network or exchange trouble: say so and try again shortly
            message = f"{type(e).__name__}: {str(e)[:200]}"
            write_status(DIR, ok=False, message=message, model_month=models[0].month)
            print(time.strftime("%F %T"), message, flush=True)
            time.sleep(60)
            continue
        print(time.strftime("%F %T"), message, flush=True)
        try:                        # after the time-critical prediction; a failure here only loses this round's extras
            if message.startswith("predicted"):
                collector.order_books(now // BAR_MS * BAR_MS)
            collector.positioning(int(time.time() * 1000))
        except Exception as e:
            print(time.strftime("%F %T"), f"collector: {type(e).__name__}: {str(e)[:200]}", flush=True)
        write_status(DIR, ok=True, message=message, model_month=models[0].month, train_rows=models[0].train_rows,
                     collected=collector.counts)
        wake = (now // BAR_MS + 1) * BAR_MS + WAKE_AFTER_CLOSE_MS
        while time.time() * 1000 < wake:        # short sleeps follow the wall clock across laptop sleep
            time.sleep(5)
            if serve_now(market, models, DIR):
                print(time.strftime("%F %T"), "answered an as-of-now request", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
