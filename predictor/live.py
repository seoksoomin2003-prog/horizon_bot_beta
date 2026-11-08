"""Live direction predictions. No orders, no keys: public prices only.

Every 4-hour bar close it records, for each of the 10 coins, the v3 model's probability that the next 4 hours close
higher (the D4 candidate; docs/research/direction_result_v4.md). One bar later it records what happened. The dashboard
reads both files and keeps score. A prediction is only recorded within MAX_LATE_MS of the close, so a laptop that slept
through a close skips it instead of predicting after the fact.

The model refits on the first of every month on rows whose answer was known by then, exactly as in the research.

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

from research.direction import ABS_MOVE, CLIP, FEATURES, fit, frame, predict  # noqa: E402
from research.h4_data import BAR_MS, DEV_START_MS, SYMBOLS, WARMUP_START_MS, _pages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "runtime" / "predictor"
SIGNAL_GAIN = 0.005             # v4's fixed bet threshold: expected gain above 0.5%
MAX_LATE_MS = 15 * 60_000
WAKE_AFTER_CLOSE_MS = 60_000


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

    def __init__(self, directory: Path, client) -> None:
        self.dir, self.client = directory, client
        self.bars: dict[str, np.ndarray] = {}
        self.funding: dict[str, np.ndarray] = {}

    def _load(self, name: str, width: int) -> np.ndarray:
        try:
            return np.load(self.dir / name)
        except OSError:
            return np.empty((0, width))

    def update(self, now_ms: int) -> None:
        from replay.market_data import _market
        for symbol in SYMBOLS:
            bars = self.bars.get(symbol)
            if bars is None:
                bars = self._load(f"{symbol}.bars.npy", 6)
                self.funding[symbol] = self._load(f"{symbol}.funding.npy", 2)
            market = _market(symbol)
            since = int(bars[-1, 0]) + 1 if len(bars) else WARMUP_START_MS
            new = _pages(lambda c: self.client.fetch_ohlcv(market, "4h", since=c, limit=1500), since, now_ms)
            new = [r for r in new if r[0] + BAR_MS <= now_ms]       # completed bars only
            if new:
                bars = np.vstack([bars, np.array(new, dtype=float)])
                bars = bars[np.unique(bars[:, 0], return_index=True)[1]]
                np.save(self.dir / f"{symbol}.bars.npy", bars)
            self.bars[symbol] = bars
            funding = self.funding[symbol]
            since = int(funding[-1, 0]) + 1 if len(funding) else WARMUP_START_MS
            new = _pages(lambda c: [[int(p["timestamp"]), float(p["fundingRate"])]
                                    for p in self.client.fetch_funding_rate_history(market, since=c, limit=1000)], since, now_ms)
            if new:
                funding = np.vstack([funding, np.array(new, dtype=float)])
                funding = funding[np.unique(funding[:, 0], return_index=True)[1]]
                np.save(self.dir / f"{symbol}.funding.npy", funding)
            self.funding[symbol] = funding

    def frames(self) -> pd.DataFrame:
        btc = frame("BTCUSDT", self.bars["BTCUSDT"], self.funding["BTCUSDT"], None, 1).set_index("close_ms")[["r1", "r6"]]
        return pd.concat([frame(s, self.bars[s], self.funding[s], None if s == "BTCUSDT" else btc, 1) for s in SYMBOLS],
                         ignore_index=True)


class Model:
    """The research model refit monthly: standardize on the training rows, clip, L2 logistic regression."""

    def __init__(self) -> None:
        self.month: int | None = None

    def refit_if_needed(self, data: pd.DataFrame, now_ms: int) -> None:
        month = int(pd.Timestamp(now_ms, unit="ms").to_period("M").start_time.value // 1_000_000)
        if month == self.month:
            return
        rows = data.dropna(subset=FEATURES + ["fwd", "sd42"])
        rows = rows[(rows["close_ms"] >= DEV_START_MS) & (rows["close_ms"] + BAR_MS <= month)]
        x = rows[FEATURES].to_numpy(float)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.w = fit(self._scale(x), (rows["fwd"] > 0).to_numpy(float))
        self.month, self.train_rows = month, len(rows)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -CLIP, CLIP)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return predict(self.w, self._scale(rows[FEATURES].to_numpy(float)))


def predictions_for(rows: pd.DataFrame, p: np.ndarray, close_ms: int, model_month: int, made_ms: int) -> list[dict]:
    out = []
    for (_, r), prob in zip(rows.iterrows(), p):
        gain = abs(2 * prob - 1) * r["sd42"] * ABS_MOVE
        out.append({"close_ms": close_ms, "symbol": r["symbol"], "price": float(math.exp(r["logc"])), "p_up": float(prob),
                    "expected_gain": float(gain), "signal": bool(gain > SIGNAL_GAIN), "model_month": model_month,
                    "made_ms": made_ms})
    return out


def outcomes_for(predicted: list[dict], scored: set, data: pd.DataFrame) -> list[dict]:
    """The next bar's close for every prediction not yet scored whose next bar has closed."""
    close = data.set_index(["symbol", "close_ms"])["logc"]
    out = []
    for p in predicted:
        key = (p["symbol"], p["close_ms"])
        nxt = (p["symbol"], p["close_ms"] + BAR_MS)
        if key in scored or nxt not in close.index or not np.isfinite(close[nxt]):
            continue
        out.append({"close_ms": p["close_ms"], "symbol": p["symbol"], "next_price": float(math.exp(close[nxt])),
                    "log_return": float(close[nxt] - math.log(p["price"]))})
    return out


def with_closes(data: pd.DataFrame, bars: dict[str, np.ndarray]) -> pd.DataFrame:
    closes = pd.concat([pd.DataFrame({"symbol": s, "close_ms": b[:, 0].astype(np.int64) + BAR_MS, "logc": np.log(b[:, 4])})
                        for s, b in bars.items()], ignore_index=True)
    return data.merge(closes, on=["symbol", "close_ms"], how="left")


def cycle(market: Market, model: Model, directory: Path, now_ms: int, clock=time.time) -> str:
    market.update(now_ms)
    data = with_closes(market.frames(), market.bars)
    model.refit_if_needed(data, now_ms)

    predicted = read_jsonl(directory / "predictions.jsonl")
    scored = {(o["symbol"], o["close_ms"]) for o in read_jsonl(directory / "outcomes.jsonl")}
    append(directory / "outcomes.jsonl", outcomes_for(predicted, scored, data))

    latest = now_ms // BAR_MS * BAR_MS
    if latest in {p["close_ms"] for p in predicted}:
        return "already predicted"
    rows = data[data["close_ms"] == latest].dropna(subset=FEATURES + ["sd42", "logc"])
    if rows.empty:
        return "no completed bar for this close yet"
    made = int(clock() * 1000)
    if made - latest > MAX_LATE_MS:
        return "too late for this close; waiting for the next one"
    append(directory / "predictions.jsonl", predictions_for(rows, model.predict(rows), latest, model.month, made))
    return f"predicted {len(rows)} coins"


def main() -> int:
    from replay.market_data import _keyless_client
    DIR.mkdir(parents=True, exist_ok=True)
    market, model = Market(DIR, _keyless_client()), Model()
    print("direction predictor started: no orders, public prices only", flush=True)
    while True:
        now = int(time.time() * 1000)
        try:
            message = cycle(market, model, DIR, now)
            write_status(DIR, ok=True, message=message, model_month=model.month, train_rows=model.train_rows)
        except Exception as e:     # network or exchange trouble: say so and try again shortly
            message = f"{type(e).__name__}: {str(e)[:200]}"
            write_status(DIR, ok=False, message=message, model_month=model.month)
            print(time.strftime("%F %T"), message, flush=True)
            time.sleep(60)
            continue
        print(time.strftime("%F %T"), message, flush=True)
        wake = (now // BAR_MS + 1) * BAR_MS + WAKE_AFTER_CLOSE_MS
        while time.time() * 1000 < wake:        # short sleeps follow the wall clock across laptop sleep
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
