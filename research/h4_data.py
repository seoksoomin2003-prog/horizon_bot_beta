"""4-hour bars and funding for the 4h rule research. See docs/research/4h_preregistration.md.

Development code cannot fetch or read anything from the validation period onward.
"""

from __future__ import annotations

import calendar
import time
from pathlib import Path

import numpy as np

BAR_MS = 4 * 3_600_000
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT", "DOGEUSDT", "BNBUSDT", "SUIUSDT", "ADAUSDT"]
DATA_DIR = Path(__file__).resolve().parent.parent / "replay_data" / "h4"


def utc_ms(text: str) -> int:
    return calendar.timegm(time.strptime(text, "%Y-%m-%d")) * 1000


WARMUP_START_MS = utc_ms("2020-09-01")
DEV_START_MS = utc_ms("2021-01-01")
DEV_END_MS = utc_ms("2025-11-06")


def _guard(end_ms: int) -> None:
    if end_ms > DEV_END_MS:
        raise PermissionError("the validation period is closed to development code")


def _pages(fetch, since: int, until: int) -> list:
    rows, cursor = [], since
    while cursor < until:
        page = fetch(cursor)
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1][0]) + 1
        time.sleep(0.2)     # leaves the public rate limit to the running bot
    return [r for r in rows if since <= r[0] < until]


def download(symbols: list[str] = SYMBOLS, data_dir: Path = DATA_DIR, end_ms: int = DEV_END_MS) -> None:
    _guard(end_ms)
    from replay.market_data import _keyless_client, _market
    data_dir.mkdir(parents=True, exist_ok=True)
    client = _keyless_client()
    client.load_markets()
    for symbol in symbols:
        market = _market(symbol)
        bars = _pages(lambda c: client.fetch_ohlcv(market, "4h", since=c, limit=1500), WARMUP_START_MS, end_ms)
        funding = _pages(lambda c: [[int(p["timestamp"]), float(p["fundingRate"])]
                                    for p in client.fetch_funding_rate_history(market, since=c, limit=1000)],
                         WARMUP_START_MS, end_ms)
        bars = np.array(bars, dtype=float)
        bars = bars[np.unique(bars[:, 0], return_index=True)[1]]
        funding = np.array(funding, dtype=float).reshape(-1, 2)
        funding = funding[np.unique(funding[:, 0], return_index=True)[1]]
        np.save(data_dir / f"{symbol}.bars.npy", bars)
        np.save(data_dir / f"{symbol}.funding.npy", funding)
        print(f"{symbol}: {len(bars)} bars from {time.strftime('%F', time.gmtime(bars[0, 0] / 1000))}, {len(funding)} funding rates", flush=True)


def load(symbol: str, end_ms: int = DEV_END_MS, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    _guard(end_ms)
    bars = np.load(data_dir / f"{symbol}.bars.npy")
    funding = np.load(data_dir / f"{symbol}.funding.npy")
    return bars[bars[:, 0] < end_ms], funding[funding[:, 0] < end_ms]
