"""Collects market data the exchange does not keep, so a later model can be tested on it. Public endpoints, no keys.

  positioning.jsonl   hourly open interest and long/short ratios (the exchange serves only the last 30 days)
  orderbook.jsonl     order-book imbalance and spread, one snapshot per coin right after each 4h close (no history exists)
  liquidations.jsonl  forced liquidations as they happen (no history exists)

Nothing here feeds today's predictions. A failure here never stops them.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

HOUR_MS = 3_600_000
HISTORY_MS = 30 * 24 * HOUR_MS
PAGE = 500
POSITIONING = {"oi": ("fapiDataGetOpenInterestHist", "sumOpenInterest"),
               "oi_value": ("fapiDataGetOpenInterestHist", "sumOpenInterestValue"),
               "long_accounts": ("fapiDataGetGlobalLongShortAccountRatio", "longAccount"),
               "top_long_positions": ("fapiDataGetTopLongShortPositionRatio", "longAccount")}
LIQUIDATION_STREAM = "wss://fstream.binance.com/market/stream?streams=!forceOrder@arr"


def append(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))


def last_times(path: Path) -> dict[str, int]:
    """The newest saved time per coin, so a restart continues where it stopped."""
    last: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    last[r["symbol"]] = max(last.get(r["symbol"], 0), int(r["ts"]))
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return last


def imbalance(bids: list, asks: list, levels: int) -> float | None:
    """(bid value - ask value) / total over the best `levels` price levels: +1 all bids, -1 all asks."""
    b = sum(float(p) * float(q) for p, q, *_ in bids[:levels])
    a = sum(float(p) * float(q) for p, q, *_ in asks[:levels])
    return (b - a) / (b + a) if b + a > 0 else None


def book_row(symbol: str, ts: int, bids: list, asks: list) -> dict | None:
    if not bids or not asks:
        return None
    bid, ask = float(bids[0][0]), float(asks[0][0])
    return {"ts": ts, "symbol": symbol, "mid": (bid + ask) / 2, "spread": (ask - bid) / ((bid + ask) / 2),
            "imbalance_20": imbalance(bids, asks, 20), "imbalance_500": imbalance(bids, asks, 500)}


def liquidation_row(message: dict, symbols: set[str]) -> dict | None:
    """One forced order from the !forceOrder@arr stream. SELL closes a long, BUY closes a short."""
    o = message.get("data", message).get("o") or {}
    if o.get("s") not in symbols:
        return None
    try:
        return {"ts": int(o["T"]), "symbol": o["s"], "side": o["S"], "value": float(o["ap"]) * float(o["z"])}
    except (KeyError, ValueError):
        return None


class Collector:
    def __init__(self, directory: Path, client, symbols: list[str], pause: float = 0.25) -> None:
        self.dir, self.client, self.symbols, self.pause = directory, client, symbols, pause
        self.last = last_times(directory / "positioning.jsonl")
        self.counts = {"positioning": 0, "orderbook": 0, "liquidations": 0}

    def _history(self, method: str, symbol: str, since: int, now_ms: int) -> list[dict]:
        # The endpoint returns the newest rows of the range, so page backwards from now.
        rows, end = [], now_ms
        while True:
            page = getattr(self.client, method)({"symbol": symbol, "period": "1h", "startTime": since, "endTime": end, "limit": PAGE})
            time.sleep(self.pause)
            rows = list(page) + rows
            if len(page) < PAGE:
                break
            end = int(page[0]["timestamp"]) - 1
        return rows

    def positioning(self, now_ms: int) -> int:
        """Appends every completed hour not yet saved, back to 30 days on the first run."""
        added = 0
        for symbol in self.symbols:
            since = max(self.last.get(symbol, 0) + 1, now_ms - HISTORY_MS + HOUR_MS)
            merged: dict[int, dict] = {}
            fetched: dict[str, list] = {}
            for field, (method, key) in POSITIONING.items():
                if method not in fetched:
                    fetched[method] = self._history(method, symbol, since, now_ms)
                for r in fetched[method]:
                    ts = int(r["timestamp"])
                    if since <= ts <= now_ms - HOUR_MS:
                        merged.setdefault(ts, {"ts": ts, "symbol": symbol})[field] = float(r[key])
            rows = [merged[ts] for ts in sorted(merged) if "oi" in merged[ts]]
            append(self.dir / "positioning.jsonl", rows)
            if rows:
                self.last[symbol] = rows[-1]["ts"]
            added += len(rows)
        self.counts["positioning"] += added
        return added

    def order_books(self, now_ms: int) -> int:
        rows = []
        for symbol in self.symbols:
            book = self.client.fapiPublicGetDepth({"symbol": symbol, "limit": 500})
            time.sleep(self.pause)
            row = book_row(symbol, now_ms, book.get("bids", []), book.get("asks", []))
            if row:
                rows.append(row)
        append(self.dir / "orderbook.jsonl", rows)
        self.counts["orderbook"] += len(rows)
        return len(rows)

    def start_liquidations(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._liquidations()), name="liquidations", daemon=True).start()

    async def _liquidations(self) -> None:
        import aiohttp
        wanted = set(self.symbols)
        while True:
            try:
                async with aiohttp.ClientSession() as session, session.ws_connect(LIQUIDATION_STREAM, heartbeat=60) as ws:
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            break
                        row = liquidation_row(json.loads(message.data), wanted)
                        if row:
                            append(self.dir / "liquidations.jsonl", [row])
                            self.counts["liquidations"] += 1
            except Exception as e:      # dropped connection: the exchange also closes every 24h
                print(time.strftime("%F %T"), f"liquidation stream: {type(e).__name__}: {str(e)[:120]}", flush=True)
            await asyncio.sleep(10)
