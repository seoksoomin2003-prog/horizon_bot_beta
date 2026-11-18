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
VAL_END_MS = utc_ms("2026-08-06")
VAL_DIR = DATA_DIR / "validation"


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
    _download(symbols, data_dir, WARMUP_START_MS, end_ms)


def _download(symbols: list[str], data_dir: Path, start_ms: int, end_ms: int) -> None:
    from replay.market_data import _keyless_client, _market
    data_dir.mkdir(parents=True, exist_ok=True)
    client = _keyless_client()
    client.load_markets()
    for symbol in symbols:
        market = _market(symbol)
        bars = _pages(lambda c: client.fetch_ohlcv(market, "4h", since=c, limit=1500), start_ms, end_ms)
        funding = _pages(lambda c: [[int(p["timestamp"]), float(p["fundingRate"])]
                                    for p in client.fetch_funding_rate_history(market, since=c, limit=1000)],
                         start_ms, end_ms)
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


def download_validation() -> None:
    """Validation-period bars only, into their own folder. Spends the validation period (direction v4, approved 2026-11-08)."""
    _download(SYMBOLS, VAL_DIR, DEV_END_MS, VAL_END_MS)


def load_validation(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Development data with the validation period appended, up to VAL_END_MS."""
    dev_bars, dev_funding = load(symbol)
    bars, funding = np.load(VAL_DIR / f"{symbol}.bars.npy"), np.load(VAL_DIR / f"{symbol}.funding.npy")
    bars, funding = np.vstack([dev_bars, bars]), np.vstack([dev_funding, funding])
    return bars[bars[:, 0] < VAL_END_MS], funding[funding[:, 0] < VAL_END_MS]


FLOW_DIR = DATA_DIR / "flow"
HOLDOUT_END_MS = utc_ms("2026-11-06")     # v5 final check: VAL_END_MS .. HOLDOUT_END_MS, once, with the user's approval


def _flow_guard(end_ms: int) -> None:
    if end_ms > VAL_END_MS:
        raise PermissionError("the v5 final-check period is closed to development code")


def download_flow(symbols: list[str] = SYMBOLS, data_dir: Path = FLOW_DIR, end_ms: int = VAL_END_MS) -> None:
    """Taker-buy volume and premium index per 4h bar: [open_ms, volume, taker_buy_volume, premium_close]."""
    _flow_guard(end_ms)
    _download_flow(symbols, data_dir, WARMUP_START_MS, end_ms)


def _download_flow(symbols: list[str], data_dir: Path, start_ms: int, end_ms: int) -> None:
    from replay.market_data import _keyless_client
    data_dir.mkdir(parents=True, exist_ok=True)
    client = _keyless_client()
    for symbol in symbols:
        query = lambda c: {"symbol": symbol, "interval": "4h", "startTime": c, "limit": 1500}
        klines = _pages(lambda c: [[int(r[0]), float(r[5]), float(r[9])] for r in client.fapiPublicGetKlines(query(c))],
                        start_ms, end_ms)
        premium = dict((int(r[0]), float(r[4])) for r in
                       _pages(lambda c: [[int(r[0])] + r[1:] for r in client.fapiPublicGetPremiumIndexKlines(query(c))],
                              start_ms, end_ms))
        rows = np.array([k + [premium.get(k[0], np.nan)] for k in klines], dtype=float)
        rows = rows[np.unique(rows[:, 0], return_index=True)[1]]
        np.save(data_dir / f"{symbol}.flow.npy", rows)
        print(f"{symbol}: {len(rows)} flow bars from {time.strftime('%F', time.gmtime(rows[0, 0] / 1000))}, "
              f"{int(np.isnan(rows[:, 3]).sum())} without premium", flush=True)


def load_flow(symbol: str, end_ms: int = VAL_END_MS, data_dir: Path = FLOW_DIR) -> np.ndarray:
    _flow_guard(end_ms)
    rows = np.load(data_dir / f"{symbol}.flow.npy")
    return rows[rows[:, 0] < end_ms]


V6_DIR = DATA_DIR / "v6"
V6_SYMBOLS_FILE = DATA_DIR / "v6_symbols.json"
EXTERNAL_DIR = DATA_DIR / "external"
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD"}


def choose_v6_symbols(count: int = 50) -> list[str]:
    """Picks and saves the v6 coins once: perpetuals listed before 2023, largest 24h turnover today, not the original 10."""
    return _choose(V6_SYMBOLS_FILE, set(SYMBOLS), count)


def _choose(file: Path, exclude: set[str], count: int) -> list[str]:
    import json
    if file.exists():
        return json.loads(file.read_text())["symbols"]
    from replay.market_data import _keyless_client
    client = _keyless_client()
    listed = {s["symbol"] for s in client.fapiPublicGetExchangeInfo()["symbols"]
              if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
              and int(s["onboardDate"]) < utc_ms("2023-01-01") and s["baseAsset"] not in STABLE_BASES} - exclude
    turnover = {t["symbol"]: float(t["quoteVolume"]) for t in client.fapiPublicGetTicker24hr() if t["symbol"] in listed}
    symbols = sorted(turnover, key=turnover.get, reverse=True)[:count]
    file.write_text(json.dumps({"chosen": time.strftime("%FT%TZ", time.gmtime()), "symbols": symbols}))
    return symbols


def download_v6() -> None:
    _download_raw(choose_v6_symbols(), V6_DIR, WARMUP_START_MS, VAL_END_MS)


def _download_raw(symbols: list[str], data_dir: Path, start_ms: int, end_ms: int) -> None:
    """Like _download, by exchange symbol name (1000PEPEUSDT and the like have no simple ccxt name)."""
    from replay.market_data import _keyless_client
    data_dir.mkdir(parents=True, exist_ok=True)
    client = _keyless_client()
    for symbol in symbols:
        if (data_dir / f"{symbol}.funding.npy").exists():
            continue
        bars = _pages(lambda c: [[int(r[0])] + [float(v) for v in r[1:6]] for r in
                                 client.fapiPublicGetKlines({"symbol": symbol, "interval": "4h", "startTime": c, "limit": 1500})],
                      start_ms, end_ms)
        funding = _pages(lambda c: [[int(r["fundingTime"]), float(r["fundingRate"])] for r in
                                    client.fapiPublicGetFundingRate({"symbol": symbol, "startTime": c, "limit": 1000})],
                         start_ms, end_ms)
        bars = np.array(bars, dtype=float)
        bars = bars[np.unique(bars[:, 0], return_index=True)[1]]
        funding = np.array(funding, dtype=float).reshape(-1, 2)
        funding = funding[np.unique(funding[:, 0], return_index=True)[1]]
        np.save(data_dir / f"{symbol}.bars.npy", bars)
        np.save(data_dir / f"{symbol}.funding.npy", funding)
        print(f"{symbol}: {len(bars)} bars from {time.strftime('%F', time.gmtime(bars[0, 0] / 1000))}, {len(funding)} funding rates", flush=True)


def load_v6(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    bars, funding = np.load(V6_DIR / f"{symbol}.bars.npy"), np.load(V6_DIR / f"{symbol}.funding.npy")
    return bars[bars[:, 0] < VAL_END_MS], funding[funding[:, 0] < VAL_END_MS]


def download_external(end_ms: int = VAL_END_MS) -> None:
    """Daily S&P 500, VIX and dollar index closes (Yahoo chart API) and hourly Deribit BTC DVOL, as [time_ms, close]."""
    import json
    import urllib.request
    _flow_guard(end_ms)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    def get(url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-script"})
        return json.loads(urllib.request.urlopen(request, timeout=30).read())

    for name, ticker in (("spx", "%5EGSPC"), ("vix", "%5EVIX"), ("dxy", "DX-Y.NYB")):
        result = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={WARMUP_START_MS // 1000}"
                     f"&period2={end_ms // 1000}&interval=1d")["chart"]["result"][0]
        rows = [[t * 1000 // 86_400_000 * 86_400_000, c] for t, c in zip(result["timestamp"], result["indicators"]["quote"][0]["close"])
                if c is not None and t * 1000 < end_ms]
        np.save(EXTERNAL_DIR / f"{name}.npy", np.array(rows, dtype=float))
        print(f"{name}: {len(rows)} daily closes", flush=True)
    rows, cursor = [], end_ms
    while cursor > utc_ms("2021-03-01"):
        page = get("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC"
                   f"&start_timestamp={utc_ms('2021-03-01')}&end_timestamp={cursor - 1}&resolution=3600")["result"]["data"]
        if not page:
            break
        rows.extend([int(r[0]), float(r[4])] for r in page)
        cursor = int(page[0][0])
        time.sleep(0.2)
    dvol = np.array(rows, dtype=float)
    dvol = dvol[np.unique(dvol[:, 0], return_index=True)[1]]
    np.save(EXTERNAL_DIR / "dvol.npy", dvol[dvol[:, 0] < end_ms])
    print(f"dvol: {len(dvol)} hourly closes from {time.strftime('%F', time.gmtime(dvol[0, 0] / 1000))}", flush=True)


def load_external() -> dict[str, np.ndarray]:
    out = {name: np.load(EXTERNAL_DIR / f"{name}.npy") for name in ("spx", "vix", "dxy", "dvol")}
    return {name: rows[rows[:, 0] < VAL_END_MS] for name, rows in out.items()}


# v7: the exchange's public archive (data.binance.vision). Files are parsed in memory; only the rows needed are kept.
ARCHIVE = "https://data.binance.vision/data/futures/um"
M5_DIR = DATA_DIR / "m5"              # [open_ms, open, high, low, close] of the 5m bars in the first 40 minutes after a 4h close
METRICS_DIR = DATA_DIR / "metrics"    # [time_ms, open_interest, accounts_long_short, top_positions_long_short], last 15 minutes before a 4h close
M5_MS = 300_000
V7_START_MS = utc_ms("2021-12-01")


def _days(start_ms: int, end_ms: int) -> list[str]:
    return [time.strftime("%Y-%m-%d", time.gmtime(t / 1000)) for t in range(start_ms, end_ms, 86_400_000)]


def archive_files(symbol: str, start_ms: int = V7_START_MS, end_ms: int = VAL_END_MS) -> dict[str, list[str]]:
    """Archive paths wholly before end_ms: monthly 5m klines, daily ones for the last partial month, daily metrics."""
    _flow_guard(end_ms)
    days = _days(start_ms, end_ms)
    last_month = days[-1][:7]
    months = sorted({d[:7] for d in days} - {last_month})
    kline = lambda period, stamp: f"{ARCHIVE}/{period}/klines/{symbol}/5m/{symbol}-5m-{stamp}.zip"
    return {"m5": [kline("monthly", m) for m in months] + [kline("daily", d) for d in days if d[:7] == last_month],
            "metrics": [f"{ARCHIVE}/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip" for d in _days(utc_ms("2021-01-01") - 86_400_000, end_ms)]}


def parse_m5(text: str) -> list[list[float]]:
    rows = []
    for line in text.splitlines():
        f = line.split(",")
        if f[0].isdigit() and int(f[0]) % BAR_MS < 8 * M5_MS:
            rows.append([int(f[0])] + [float(v) for v in f[1:5]])
    return rows


def parse_metrics(text: str) -> list[list[float]]:
    rows = []
    for line in text.splitlines():
        f = line.replace('"', "").split(",")
        if len(f) < 7 or not f[0][:1].isdigit():
            continue
        t = calendar.timegm(time.strptime(f[0], "%Y-%m-%d %H:%M:%S")) * 1000
        if t % BAR_MS >= BAR_MS - 3 * M5_MS:
            number = lambda v: float(v) if v else np.nan
            rows.append([t, number(f[2]), number(f[6]), number(f[5])])
    return rows


FLOW_ARCHIVE_DIR = DATA_DIR / "flow_archive"   # 4h klines and premium index from the archive, every reference coin
DEPTH_DIR = DATA_DIR / "depth"                 # [close_ms, snapshots, notional within -5..-1 % and +1..+5 % of the mid], before a 4h close
DEPTH_START_MS = utc_ms("2023-01-01")         # the archive's order-book depth starts here
DEPTH_LEVELS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def _monthly_then_daily(path, start_ms: int, end_ms: int) -> list[str]:
    """path(period, stamp) for every whole month before the last one, then the days of the last (partial) month."""
    days = _days(start_ms, end_ms)
    last_month = days[-1][:7]
    months = sorted({d[:7] for d in days} - {last_month})
    return [path("monthly", m) for m in months] + [path("daily", d) for d in days if d[:7] == last_month]


def flow_files(symbol: str, end_ms: int = VAL_END_MS) -> dict[str, list[str]]:
    _flow_guard(end_ms)
    klines = lambda kind: (lambda period, stamp: f"{ARCHIVE}/{period}/{kind}/{symbol}/4h/{symbol}-4h-{stamp}.zip")
    return {"klines4h": _monthly_then_daily(klines("klines"), WARMUP_START_MS, end_ms),
            "premium4h": _monthly_then_daily(klines("premiumIndexKlines"), WARMUP_START_MS, end_ms)}


def depth_files(symbol: str, end_ms: int = VAL_END_MS) -> list[str]:
    _flow_guard(end_ms)
    return [f"{ARCHIVE}/daily/bookDepth/{symbol}/{symbol}-bookDepth-{d}.zip" for d in _days(DEPTH_START_MS, end_ms)]


def _ms(field: str) -> int:
    t = int(field)
    return t // 1000 if t > 10**14 else t      # some archive files are in microseconds


def parse_klines(text: str) -> list[list[float]]:
    """[open_ms, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote]."""
    rows = []
    for line in text.splitlines():
        f = line.split(",")
        if f[0].isdigit():
            rows.append([_ms(f[0])] + [float(f[k]) for k in (1, 2, 3, 4, 5, 7, 8, 9, 10)])
    return rows


def parse_premium(text: str) -> list[list[float]]:
    """[open_ms, premium index close]."""
    return [[_ms(f[0]), float(f[4])] for f in (line.split(",") for line in text.splitlines()) if f[0].isdigit()]


def parse_depth(text: str) -> list[list[float]]:
    """One row per 4h close in the file's day: the mean notional per level over the snapshots stamped in the 15 minutes
    before the close, [close_ms, snapshots, notional at DEPTH_LEVELS]. Only those minutes are parsed (the files are large)."""
    rows = []
    day = text[text.find("\n") + 1:][:10]
    if len(day) < 10 or not day[:4].isdigit():
        return rows
    base = calendar.timegm(time.strptime(day, "%Y-%m-%d")) * 1000
    for hour in (3, 7, 11, 15, 19, 23):
        starts = [i for i in (text.find(f"{day} {hour:02d}:{m}:") for m in range(45, 60)) if i >= 0]
        if not starts:
            continue
        sums, stamps = np.zeros(len(DEPTH_LEVELS)), set()
        counts = np.zeros(len(DEPTH_LEVELS))
        for line in text[min(starts):].splitlines():
            f = line.split(",")
            if len(f) < 4 or f[0][11:13] != f"{hour:02d}":
                break
            level = int(float(f[1]))
            if level in DEPTH_LEVELS:
                k = DEPTH_LEVELS.index(level)
                sums[k] += float(f[3])
                counts[k] += 1
                stamps.add(f[0])
        with np.errstate(invalid="ignore"):
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        rows.append([base + (hour + 1) * 3_600_000, float(len(stamps))] + list(mean))
    return rows


ARCHIVE_KINDS = {     # kind: (folder, file name, row width, parser, the coin's archive URLs)
    "m5": (M5_DIR, "{s}.npy", 5, parse_m5, lambda s, end: archive_files(s, end_ms=end)["m5"]),
    "metrics": (METRICS_DIR, "{s}.npy", 4, parse_metrics, lambda s, end: archive_files(s, end_ms=end)["metrics"]),
    "klines4h": (FLOW_ARCHIVE_DIR, "{s}.klines.npy", 10, parse_klines, lambda s, end: flow_files(s, end)["klines4h"]),
    "premium4h": (FLOW_ARCHIVE_DIR, "{s}.premium.npy", 2, parse_premium, lambda s, end: flow_files(s, end)["premium4h"]),
    "depth": (DEPTH_DIR, "{s}.npy", 2 + len(DEPTH_LEVELS), parse_depth, lambda s, end: depth_files(s, end)),
}


def download_archive(symbols: list[str], kinds: tuple[str, ...] = ("m5", "metrics"), end_ms: int = VAL_END_MS, parallel: int = 24) -> None:
    """Fetches the archive files of each coin and saves the kept rows. A missing file (404: not listed yet, or a day the
    archive lacks) is skipped. A coin with a file that keeps failing is not saved; the others go on, and the run raises at
    the end. A coin already saved is skipped, so running again resumes."""
    import asyncio
    import io
    import zipfile

    import aiohttp
    _flow_guard(end_ms)

    async def fetch(session, gate, url: str, parse) -> list:
        async with gate:
            for attempt in range(8):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as r:
                        if r.status == 404:
                            return []
                        r.raise_for_status()
                        body = await r.read()
                    with zipfile.ZipFile(io.BytesIO(body)) as z:
                        return parse(z.read(z.namelist()[0]).decode())
                except (aiohttp.ClientError, asyncio.TimeoutError, zipfile.BadZipFile, OSError):
                    await asyncio.sleep(min(5 * 2 ** attempt, 120))
            return None                                   # still failing: the coin is not saved and is fetched again next run

    async def run() -> None:
        gate = asyncio.Semaphore(parallel)
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                for kind in kinds:
                    folder, name, width, parse, urls = ARCHIVE_KINDS[kind]
                    path = folder / name.format(s=symbol)
                    if path.exists():
                        continue
                    files = urls(symbol, end_ms)
                    pages = await asyncio.gather(*(fetch(session, gate, u, parse) for u in files))
                    if any(p is None for p in pages):
                        failed.append(f"{symbol} {kind}")
                        print(time.strftime("%T"), f"{symbol} {kind}: {sum(p is None for p in pages)} files failed, not saved", flush=True)
                        continue
                    rows = np.array([r for page in pages for r in page], dtype=float).reshape(-1, width)
                    rows = rows[np.unique(rows[:, 0], return_index=True)[1]]
                    np.save(path, rows[rows[:, 0] < end_ms])
                    print(time.strftime("%T"), f"{symbol} {kind}: {len(rows)} rows from {sum(bool(p) for p in pages)} of {len(pages)} files", flush=True)

    failed: list[str] = []
    for kind in kinds:
        ARCHIVE_KINDS[kind][0].mkdir(parents=True, exist_ok=True)
    asyncio.run(run())
    if failed:
        raise RuntimeError(f"not saved, run again to resume: {', '.join(failed)}")


def load_m5(symbol: str) -> np.ndarray:
    rows = np.load(M5_DIR / f"{symbol}.npy")
    return rows[rows[:, 0] < VAL_END_MS]


def load_metrics(symbol: str) -> np.ndarray:
    rows = np.load(METRICS_DIR / f"{symbol}.npy")
    return rows[rows[:, 0] < VAL_END_MS]


def load_flow_archive(symbol: str) -> np.ndarray:
    """[open_ms, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote, premium_close]."""
    klines = np.load(FLOW_ARCHIVE_DIR / f"{symbol}.klines.npy")
    premium = np.load(FLOW_ARCHIVE_DIR / f"{symbol}.premium.npy")
    i = np.searchsorted(premium[:, 0], klines[:, 0])
    hit = (i < len(premium)) & (premium[np.minimum(i, len(premium) - 1), 0] == klines[:, 0])
    rows = np.column_stack([klines, np.where(hit, premium[np.minimum(i, len(premium) - 1), 1], np.nan)])
    return rows[rows[:, 0] < VAL_END_MS]


def load_depth(symbol: str) -> np.ndarray:
    rows = np.load(DEPTH_DIR / f"{symbol}.npy")
    return rows[rows[:, 0] < VAL_END_MS]


V9_DIR = DATA_DIR / "v9"
V9_SYMBOLS_FILE = DATA_DIR / "v9_symbols.json"


def choose_v9_symbols(count: int = 50) -> list[str]:
    """The v9 coins, picked and saved once: the same rule as v6, without the 60 coins already used."""
    return _choose(V9_SYMBOLS_FILE, set(SYMBOLS) | set(choose_v6_symbols()), count)


def download_v9() -> None:
    _download_raw(choose_v9_symbols(), V9_DIR, WARMUP_START_MS, VAL_END_MS)


def load_v9(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    bars, funding = np.load(V9_DIR / f"{symbol}.bars.npy"), np.load(V9_DIR / f"{symbol}.funding.npy")
    return bars[bars[:, 0] < VAL_END_MS], funding[funding[:, 0] < VAL_END_MS]
