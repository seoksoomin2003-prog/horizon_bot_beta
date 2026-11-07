"""Past 1-minute bars, funding rates and market rules. Public endpoints only, through a keyless client."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MINUTE_MS = 60_000
BAR_MINUTES = 15
BAR_MS = BAR_MINUTES * MINUTE_MS
WINDOW_BARS = 300       # the bot's fetch_ohlcv limit
DATA_DIR = Path(__file__).resolve().parent.parent / "replay_data"


def _market(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT:USDT"


def _keyless_client():
    import ccxt
    return ccxt.binance({"enableRateLimit": True, "timeout": 20_000, "options": {"defaultType": "future"}})


def regular_grid(rows: np.ndarray) -> np.ndarray:
    """One row per minute. A missing minute becomes a flat bar at the last close, with no volume."""
    rows = rows[np.unique(rows[:, 0], return_index=True)[1]]
    start, end = int(rows[0, 0]), int(rows[-1, 0])
    idx = ((rows[:, 0] - start) // MINUTE_MS).astype(np.int64)
    out = np.empty(((end - start) // MINUTE_MS + 1, 6))
    out[:, 0] = start + np.arange(len(out)) * MINUTE_MS
    filled = np.zeros(len(out), dtype=bool)
    out[idx, 1:] = rows[:, 1:]
    filled[idx] = True
    for i in np.flatnonzero(~filled):
        out[i, 1:5] = out[i - 1, 4]
        out[i, 5] = 0.0
    return out


def _fetch_bars(client, symbol: str, since: int, until: int) -> list:
    rows, cursor = [], since
    while cursor < until:
        page = client.fetch_ohlcv(_market(symbol), "1m", since=cursor, limit=1500)
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1][0]) + MINUTE_MS
        time.sleep(0.2)     # leaves most of the public rate limit to the running bot
    return [r for r in rows if since <= r[0] < until]


def _fetch_funding(client, symbol: str, since: int, until: int) -> list:
    rows, cursor = [], since
    while cursor < until:
        page = client.fetch_funding_rate_history(_market(symbol), since=cursor, limit=1000)
        if not page:
            break
        rows.extend([int(p["timestamp"]), float(p["fundingRate"])] for p in page)
        cursor = int(page[-1]["timestamp"]) + 1
        time.sleep(0.2)
    return [r for r in rows if since <= r[0] < until]


def download(symbols: list[str], start_ms: int, end_ms: int, data_dir: Path = DATA_DIR) -> None:
    """Fetches what is missing, so an interrupted download continues. Never keeps a bar that is still forming."""
    data_dir.mkdir(parents=True, exist_ok=True)
    client = _keyless_client()
    end_ms = min(end_ms, (int(time.time() * 1000) // MINUTE_MS) * MINUTE_MS)
    start_ms = start_ms // MINUTE_MS * MINUTE_MS
    markets = client.load_markets()
    rules = {}
    for symbol in symbols:
        m = markets[_market(symbol)]
        rules[symbol] = {"amount_step": float(m["precision"]["amount"]), "price_tick": float(m["precision"]["price"]),
                         "min_amount": float(m["limits"]["amount"]["min"] or 0.0)}
    (data_dir / "markets.json").write_text(json.dumps(rules, indent=1))
    for n, symbol in enumerate(symbols, 1):
        path = data_dir / f"{symbol}.npy"
        have = np.load(path) if path.exists() else None
        parts = [] if have is None else [have]
        if have is None:
            parts.append(np.array(_fetch_bars(client, symbol, start_ms, end_ms), dtype=float))
        else:
            if start_ms < have[0, 0]:
                parts.insert(0, np.array(_fetch_bars(client, symbol, start_ms, int(have[0, 0])), dtype=float))
            if end_ms > have[-1, 0] + MINUTE_MS:
                parts.append(np.array(_fetch_bars(client, symbol, int(have[-1, 0]) + MINUTE_MS, end_ms), dtype=float))
        parts = [p for p in parts if len(p)]
        np.save(path, regular_grid(np.concatenate(parts)))
        funding = _fetch_funding(client, symbol, start_ms, end_ms)
        np.save(data_dir / f"{symbol}.funding.npy", np.array(funding, dtype=float).reshape(-1, 2))
        print(f"[{n}/{len(symbols)}] {symbol}: {len(np.load(path)):,} minutes, {len(funding):,} funding rates", flush=True)


def load_bars(symbol: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    return np.load(data_dir / f"{symbol}.npy")


def closed_bars(bars: np.ndarray) -> tuple[np.ndarray, int]:
    """Aggregates minutes into closed 15-minute bars. Returns them and the minute index of the first bar's start."""
    first = int((-bars[0, 0] % BAR_MS) // MINUTE_MS)
    count = (len(bars) - first) // BAR_MINUTES
    body = bars[first:first + count * BAR_MINUTES].reshape(count, BAR_MINUTES, 6)
    out = np.column_stack([body[:, 0, 0], body[:, 0, 1], body[:, :, 2].max(axis=1), body[:, :, 3].min(axis=1),
                           body[:, -1, 4], body[:, :, 5].sum(axis=1)])
    return out, first


def window_at(bars: np.ndarray, closed: np.ndarray, first: int, minute: int) -> np.ndarray:
    """What the bot's fetch_ohlcv returns at the end of this minute: up to 299 closed bars and the forming one."""
    offset = minute - first
    bar_index, inside = offset // BAR_MINUTES, offset % BAR_MINUTES
    part = bars[minute - inside:minute + 1]
    forming = [part[0, 0], part[0, 1], part[:, 2].max(), part[:, 3].min(), part[-1, 4], part[:, 5].sum()]
    return np.vstack([closed[max(0, bar_index - (WINDOW_BARS - 1)):bar_index], forming])
