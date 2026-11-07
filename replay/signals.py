"""Runs the bot's real SignalEngine once per symbol per minute and stores the last row, as the bot would have seen it."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from replay.market_data import DATA_DIR, MINUTE_MS, WINDOW_BARS, closed_bars, load_bars, window_at

CHUNK_MINUTES = 7 * 24 * 60
REGIMES = {"TREND": 0, "RANGE": 1, "LOW_VOL": 2}
REGIME_NAMES = {v: k for k, v in REGIMES.items()}
TRIGGERS = {"LONG": 1, "SHORT": -1}
TRIGGER_NAMES = {1: "LONG", -1: "SHORT", 0: None}
COLUMNS = ["trigger", "regime", "bias", "score", "confidence", "atr"]


def fast_percentile_apply(rolling, func, raw=False, **kwargs):
    """Same numbers as SignalEngine's rolling rank of the last value, without a Python call per window.

    SignalEngine has one rolling apply, taking half of its run time. tests/test_replay.py compares the two.
    """
    import pandas as pd
    from numpy.lib.stride_tricks import sliding_window_view
    series = rolling.obj
    window = int(rolling.window)
    out = np.full(len(series), np.nan)
    if len(series) >= window:
        views = sliding_window_view(series.to_numpy(dtype=float), window)
        last = views[:, -1:]
        lower, equal = (views < last).sum(axis=1), (views == last).sum(axis=1)
        ranks = (lower + (equal + 1) / 2) / window * 100
        ranks[np.isnan(views).any(axis=1)] = np.nan
        out[window - 1:] = ranks
    return pd.Series(out, index=series.index, name=series.name)


def signal_rows(bars: np.ndarray, start_minute: int, end_minute: int) -> dict:
    import pandas as pd
    from pandas.core.window.rolling import Rolling
    from signal_engine import SignalEngine
    Rolling.apply = fast_percentile_apply     # worker processes only; nothing else runs in them
    engine = SignalEngine()
    closed, first = closed_bars(bars)
    n = end_minute - start_minute
    out = {"trigger": np.zeros(n, np.int8), "regime": np.zeros(n, np.int8), "bias": np.zeros(n, np.int8),
           "score": np.zeros(n, np.int8), "confidence": np.zeros(n), "atr": np.zeros(n)}
    names = ["timestamp", "open", "high", "low", "close", "volume"]
    for i, minute in enumerate(range(start_minute, end_minute)):
        rows = window_at(bars, closed, first, minute)
        if len(rows) < WINDOW_BARS:
            raise ValueError("not enough history before the replay start")
        df = pd.DataFrame(rows, columns=names)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        last = engine.compute_all_signals(df, "REPLAY").iloc[-1]
        out["trigger"][i] = TRIGGERS.get(last["trigger"], 0)
        out["regime"][i] = REGIMES[last["regime"]]
        out["bias"][i] = TRIGGERS[last["trend_bias"]]
        out["score"][i] = int(last["score"])
        out["confidence"][i] = float(last["confidence"])
        out["atr"][i] = float(last["atr"])
    return out


def _chunk_path(data_dir: Path, symbol: str, start_ms: int) -> Path:
    return data_dir / "signals" / f"{symbol}_{start_ms}.npz"


def _compute_chunk(task: tuple) -> str:
    symbol, start_minute, end_minute, data_dir = task
    os.nice(15)     # the running bot always comes first
    bars = load_bars(symbol, data_dir)
    path = _chunk_path(data_dir, symbol, int(bars[start_minute, 0]))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **signal_rows(bars, start_minute, end_minute))
    os.replace(tmp, path)
    return path.name


def chunks_for(symbol: str, start_ms: int, end_ms: int, data_dir: Path = DATA_DIR) -> list:
    """Week-long pieces on a fixed calendar grid, so runs over different periods share finished work."""
    bars = load_bars(symbol, data_dir)
    t0 = int(bars[0, 0])
    first = (start_ms - t0) // MINUTE_MS
    last = min((end_ms - t0) // MINUTE_MS, len(bars))
    tasks, minute = [], first - first % CHUNK_MINUTES
    while minute < last:
        stop = min(minute + CHUNK_MINUTES, len(bars))
        if minute >= WINDOW_BARS * 15 + 15:
            tasks.append((symbol, minute, stop, data_dir))
        elif stop > first:
            tasks.append((symbol, first, stop, data_dir))
        minute = stop
    return tasks


def compute(symbols: list[str], start_ms: int, end_ms: int, workers: int = 3, data_dir: Path = DATA_DIR) -> None:
    (data_dir / "signals").mkdir(parents=True, exist_ok=True)
    tasks = [t for s in symbols for t in chunks_for(s, start_ms, end_ms, data_dir)]
    # Newest weeks first, across all symbols, so a shorter recent period can be replayed before the rest is done.
    tasks.sort(key=lambda t: (-int(np.load(data_dir / f"{t[0]}.npy", mmap_mode="r")[t[1], 0]), t[0]))
    pending = []
    for t in tasks:
        bars_t0 = int(np.load(data_dir / f"{t[0]}.npy", mmap_mode="r")[t[1], 0])
        path = _chunk_path(data_dir, t[0], bars_t0)
        # A chunk cut short by the end of the data is redone once more minutes exist.
        if not path.exists() or len(np.load(path)["trigger"]) != t[2] - t[1]:
            pending.append(t)
    print(f"signal chunks: {len(tasks)} total, {len(pending)} to compute, {workers} low-priority workers", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, name in enumerate(pool.imap_unordered(_compute_chunk, pending), 1):
            print(f"[{n}/{len(pending)}] {name}", flush=True)


def load_signals(symbol: str, start_ms: int, end_ms: int, data_dir: Path = DATA_DIR) -> dict:
    """Per-minute arrays over [start_ms, end_ms), refusing any gap: a missing minute must not read as 'no signal'."""
    bars = np.load(data_dir / f"{symbol}.npy", mmap_mode="r")
    t0 = int(bars[0, 0])
    n = (end_ms - start_ms) // MINUTE_MS
    out = {c: None for c in COLUMNS}
    seen = np.zeros(n, dtype=bool)
    for path in sorted((data_dir / "signals").glob(f"{symbol}_*.npz")):
        begin = int(path.stem.split("_")[1])
        with np.load(path) as chunk:
            size = len(chunk["trigger"])
            lo, hi = max(begin, start_ms), min(begin + size * MINUTE_MS, end_ms)
            if lo >= hi:
                continue
            a, b = (lo - start_ms) // MINUTE_MS, (hi - start_ms) // MINUTE_MS
            c = (lo - begin) // MINUTE_MS
            for name in COLUMNS:
                if out[name] is None:
                    out[name] = np.zeros(n, chunk[name].dtype)
                out[name][a:b] = chunk[name][c:c + b - a]
            seen[a:b] = True
    if not seen.all():
        raise ValueError(f"{symbol}: signals missing for {int((~seen).sum())} minutes, starting at index {int(np.argmin(seen))}")
    assert (start_ms - t0) % MINUTE_MS == 0
    return out
