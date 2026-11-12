"""Direction model for docs/research/direction_preregistration_v3.md (v4: fixed bet threshold, v5: order-flow and basis
features, v6: external features and transfer, v7: limit-order fills, positioning and market-relative features,
v8: stricter fills and per-decision-time baskets, v9: direction relative to the median coin, v10: weights that
change with the market's state).
Parameters are fixed in those documents; do not tune."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.h4_sim import BAR_MS

DAY_MS = 86_400_000
LAMBDA = 10.0
CLIP = 5.0
SIDE_COST = 0.0010
WEIGHT = 0.10
ABS_MOVE = math.sqrt(2 / math.pi)       # mean |x| of a normal with std 1 (0.798)
HORIZONS = {"D4": 1, "D24": 6}
FEATURES = ["r1", "r6", "r42", "r168", "vol", "vol_change", "btc_r1", "btc_r6", "funding", "volume", "range_pos",
            "hour_sin", "hour_cos"]
FLOW_FEATURES = ["taker_1", "taker_6", "btc_taker_1", "basis", "basis_change"]     # v5
EXTERNAL_FEATURES = ["spx_r", "vix", "vix_change", "dxy_r", "dvol", "dvol_change"]  # v6
POSITION_FEATURES = {"OI": ["oi_change_1", "oi_change_6", "oi_x_r1"],                # v7
                     "LS": ["ls_accounts", "ls_accounts_change", "ls_top", "ls_top_change"]}
MARKET_FEATURES = ["xs_r1", "xs_r6"]                                                # v7
M5_MS = 300_000
MAKER_COST = 0.0002         # v7: the exchange's base limit-order fee per side
FILL_BARS = 6               # v7: a limit order waits 30 minutes
BASKET_CAP = 10             # v8: a trade gets at most a tenth of the capital
REGIMES = ["up_calm", "up_wild", "down_calm", "down_wild"]      # v10
REGIME_WINDOW = 360         # v10: "usual" volatility is the median of the last 360 closes (60 days)


def _funding_sum(fts: np.ndarray, cum: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of funding events with start < ts <= end."""
    a, b = np.searchsorted(fts, start, side="right"), np.searchsorted(fts, end, side="right")
    return cum[b] - cum[a], b - a


def frame(symbol: str, bars: np.ndarray, funding: np.ndarray, btc: pd.DataFrame | None, horizon: int,
          flow: np.ndarray | None = None) -> pd.DataFrame:
    """One row per 4h bar close: the features from data up to that close, the forward log return and the funding paid while held.

    flow rows are [open_ms, volume, taker_buy_volume, premium_close]; with them the v5 FLOW_FEATURES are added.
    """
    close_ms = bars[:, 0].astype(np.int64) + BAR_MS
    grid = np.arange(close_ms[0], close_ms[-1] + 1, BAR_MS)
    df = pd.DataFrame(bars[:, 2:6], index=close_ms, columns=["high", "low", "close", "volume_raw"]).reindex(grid)
    logc = np.log(df["close"])
    r = logc.diff()
    out = pd.DataFrame(index=grid)
    out["r1"], out["r6"], out["r42"], out["r168"] = r, logc.diff(6), logc.diff(42), logc.diff(168)
    sd42 = r.rolling(42, min_periods=42).std()
    out["sd42"] = sd42
    out["vol"] = np.log(sd42)
    out["vol_change"] = np.log(r.rolling(6, min_periods=6).std() / sd42)
    if btc is None:
        out["btc_r1"], out["btc_r6"] = out["r1"], out["r6"]
    else:
        out["btc_r1"], out["btc_r6"] = btc["r1"].reindex(grid), btc["r6"].reindex(grid)
    fts = funding[:, 0].astype(np.int64)
    cum = np.concatenate([[0.0], np.cumsum(funding[:, 1])])
    total, count = _funding_sum(fts, cum, grid - DAY_MS, grid)
    out["funding"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    vol6, vol42 = df["volume_raw"].rolling(6, min_periods=6).sum(), df["volume_raw"].rolling(42, min_periods=42).sum()
    out["volume"] = np.log(vol6 / (vol42 / 7))
    lo, hi = df["low"].rolling(42, min_periods=42).min(), df["high"].rolling(42, min_periods=42).max()
    out["range_pos"] = (df["close"] - lo) / (hi - lo)
    hour = (grid // 3_600_000) % 24
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    if flow is not None:
        f = pd.DataFrame(flow[:, 1:], index=flow[:, 0].astype(np.int64) + BAR_MS, columns=["vol", "buy", "basis"]).reindex(grid)
        out["taker_1"] = f["buy"] / f["vol"] - 0.5
        out["taker_6"] = f["buy"].rolling(6, min_periods=6).sum() / f["vol"].rolling(6, min_periods=6).sum() - 0.5
        out["btc_taker_1"] = out["taker_1"] if btc is None else btc["taker_1"].reindex(grid)
        out["basis"] = f["basis"]
        out["basis_change"] = f["basis"].diff(6)
    out["fwd"] = logc.shift(-horizon) - logc
    out["fund_hold"], _ = _funding_sum(fts, cum, grid, grid + horizon * BAR_MS)
    out["symbol"] = symbol
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.rename_axis("close_ms").reset_index()


def add_external(data: pd.DataFrame, external: dict[str, np.ndarray]) -> pd.DataFrame:
    """Adds the v6 EXTERNAL_FEATURES. A daily close dated D is used from D+1 00:00 UTC; DVOL is the hourly close at the bar close."""
    out = data.copy()
    when = out["close_ms"].to_numpy()

    def daily(rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        i = np.searchsorted(rows[:, 0] + DAY_MS, when, side="right") - 1
        return np.where(i >= 0, values[np.maximum(i, 0)], np.nan)

    def log_change(rows: np.ndarray) -> np.ndarray:
        return np.concatenate([[np.nan], np.diff(np.log(rows[:, 1]))])

    out["spx_r"] = daily(external["spx"], log_change(external["spx"]))
    out["vix"] = daily(external["vix"], np.log(external["vix"][:, 1]))
    out["vix_change"] = daily(external["vix"], log_change(external["vix"]))
    out["dxy_r"] = daily(external["dxy"], log_change(external["dxy"]))
    hourly = pd.Series(np.log(external["dvol"][:, 1]), index=external["dvol"][:, 0].astype(np.int64) + 3_600_000)   # hour close time
    out["dvol"] = hourly.reindex(when).to_numpy()
    out["dvol_change"] = out["dvol"] - hourly.reindex(when - 6 * BAR_MS).to_numpy()
    return out


def add_positioning(data: pd.DataFrame, metrics: dict[str, np.ndarray]) -> pd.DataFrame:
    """Adds the v7 POSITION_FEATURES. metrics[symbol] rows are [time_ms, open_interest, accounts_long_short, top_positions_long_short].

    A close at T uses the last row stamped T-15min .. T-5min, so nothing recorded at or after the close leaks in.
    `data` holds each coin on a regular 4h grid (as `frame` returns it), so shifting by rows shifts by bars.
    """
    out = data.copy()
    when = out["close_ms"].to_numpy()
    values = np.full((len(out), 3), np.nan)
    for symbol, idx in out.groupby("symbol").indices.items():
        rows = metrics.get(symbol)
        if rows is None or not len(rows):
            continue
        i = np.maximum(np.searchsorted(rows[:, 0], when[idx] - M5_MS, side="right") - 1, 0)
        fresh = (rows[i, 0] <= when[idx] - M5_MS) & (rows[i, 0] >= when[idx] - 3 * M5_MS)
        values[idx] = np.where(fresh[:, None], rows[i, 1:4], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        oi, accounts, top = (pd.Series(np.log(values[:, k]), index=out.index) for k in range(3))
    back = lambda s, bars: s.groupby(out["symbol"]).shift(bars)
    out["oi_change_1"], out["oi_change_6"] = oi - back(oi, 1), oi - back(oi, 6)
    out["oi_x_r1"] = out["oi_change_1"] * out["r1"] / out["sd42"]
    out["ls_accounts"], out["ls_accounts_change"] = accounts, accounts - back(accounts, 6)
    out["ls_top"], out["ls_top_change"] = top, top - back(top, 6)
    features = sum(POSITION_FEATURES.values(), [])
    out[features] = out[features].replace([np.inf, -np.inf], np.nan)
    return out


def add_market(data: pd.DataFrame) -> pd.DataFrame:
    """Adds the v7 MARKET_FEATURES: each coin's return minus the median return of every coin at that close."""
    out = data.copy()
    for k in ("r1", "r6"):
        out[f"xs_{k}"] = out[k] - out.groupby("close_ms")[k].transform("median")
    return out


def add_relative(data: pd.DataFrame, reference: set[str]) -> pd.DataFrame:
    """v9: the question becomes "will this coin beat the median reference coin?". `fwd` turns into the forward return
    minus the reference coins' median forward return at that close (the plain one stays in `fwd_abs`), and the
    MARKET_FEATURES are measured against the same coins. A coin outside `reference` never moves the median."""
    out = data.copy()
    ref = out[out["symbol"].isin(reference)]
    median = lambda column: out["close_ms"].map(ref.groupby("close_ms")[column].median())
    out["xs_r1"], out["xs_r6"] = out["r1"] - median("r1"), out["r6"] - median("r6")
    out["fwd_abs"] = out["fwd"]
    out["fwd"] = out["fwd"] - median("fwd")
    return out


def add_regime(data: pd.DataFrame, reference: set[str], features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """v10: lets every weight differ by the market's state. Returns the data and the names of the added columns.

    The state at a close comes from the reference coins at that close and earlier, nothing later: "up" when their
    median 7-day return is above zero, "wild" when their median volatility is above its own median over the last
    REGIME_WINDOW closes. Each state gets a 0/1 column and a copy of every feature that is zero outside that state,
    so the fit can move a weight in one state while the penalty pulls it back toward the shared weight.
    """
    out = data.copy()
    market = out[out["symbol"].isin(reference)].groupby("close_ms")[["r42", "vol"]].median().sort_index()
    usual = market["vol"].rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).median()
    state = pd.Series(np.where(market["r42"] > 0, 0, 2) + np.where(market["vol"] > usual, 1, 0), index=market.index, dtype=float)
    out["regime"] = out["close_ms"].map(state.where(usual.notna() & market["r42"].notna()))
    names = []
    for k, name in enumerate(REGIMES):
        on = (out["regime"] == k).astype(float).where(out["regime"].notna())
        out[f"is_{name}"] = on
        names.append(f"is_{name}")
        for f in features:
            out[f"{f}@{name}"] = out[f] * on
            names.append(f"{f}@{name}")
    return out, names


def fill_windows(m5: np.ndarray) -> pd.DataFrame:
    """Per 4h close T with all seven 5m bars present: `limit` is the first 5m close (the order price, placed at T+5min),
    `low`/`high`/`last` are the lowest low, highest high and final close of the next FILL_BARS bars. m5 rows are
    [open_ms, open, high, low, close], sorted by time."""
    t = m5[:, 0].astype(np.int64)
    df = pd.DataFrame({"T": t // BAR_MS * BAR_MS, "k": (t % BAR_MS) // M5_MS, "h": m5[:, 2], "l": m5[:, 3], "c": m5[:, 4]})
    rest = df[(df["k"] >= 1) & (df["k"] <= FILL_BARS)].groupby("T").agg(low=("l", "min"), high=("h", "max"), last=("c", "last"), n=("c", "size"))
    first = df[df["k"] == 0].set_index("T")["c"].rename("limit")
    return rest[rest["n"] == FILL_BARS].join(first, how="inner")[["limit", "low", "high", "last"]]


def maker_trades(selected: pd.DataFrame, windows: pd.DataFrame, horizon: int, maker_cost: float = MAKER_COST,
                 taker_cost: float = SIDE_COST, gap: float = 0.0) -> pd.DataFrame:
    """Each selected prediction as a limit-order trade (v7 M). `windows` is fill_windows per coin, indexed (symbol, T).

    A limit order fills only when the price trades through it: a buy when the window's low is below the limit, a sell
    when its high is above. An unfilled entry goes to market at the window's last close (drop those rows for "limit
    only"); an unfilled exit always does. Trades without complete 5m bars at either end are left out.
    `net_1x`/`net_2x` are after funding and both sides' costs; `gross_market` is first-5m-close to first-5m-close.
    With `gap` the price must pass the order by that fraction before it counts as filled (v8: 0.05%).
    """
    t = selected.join(windows.add_prefix("e_"), on=["symbol", "close_ms"], how="inner")
    t["exit_ms"] = t["close_ms"] + horizon * BAR_MS
    t = t.join(windows.add_prefix("x_"), on=["symbol", "exit_ms"], how="inner")
    long = (t["p"] > 0.5).to_numpy()
    side = np.where(long, 1.0, -1.0)
    below, above = 1 - gap, 1 + gap
    t["entry_filled"] = np.where(long, t["e_low"] < t["e_limit"] * below, t["e_high"] > t["e_limit"] * above)
    t["exit_filled"] = np.where(long, t["x_high"] > t["x_limit"] * above, t["x_low"] < t["x_limit"] * below)
    entry = np.where(t["entry_filled"], t["e_limit"], t["e_last"])
    exit_ = np.where(t["exit_filled"], t["x_limit"], t["x_last"])
    funding = side * t["fund_hold"]
    t["move"] = side * (exit_ / entry - 1)
    t["gross_market"] = side * (t["x_limit"] / t["e_limit"] - 1) - funding
    makers = t["entry_filled"].astype(int) + t["exit_filled"].astype(int)
    cost = makers * maker_cost + (2 - makers) * taker_cost
    t["net_1x"], t["net_2x"] = t["move"] - funding - cost, t["move"] - funding - 2 * cost
    return t


def basket_returns(trades: pd.DataFrame, column: str) -> pd.Series:
    """v8: the return of each decision time's basket. Every trade gets min(1/BASKET_CAP, 1/n) of the capital, so a
    time with 3 trades risks 30% and a time with 60 trades risks 100%, never more."""
    by_time = trades.groupby("close_ms")[column]
    return by_time.sum() * np.minimum(1 / BASKET_CAP, 1 / by_time.size())


def fit(x: np.ndarray, y: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """L2 logistic regression by Newton steps. w[0] is the unpenalized intercept."""
    a = np.hstack([np.ones((len(x), 1)), x])
    reg = lam * np.eye(a.shape[1])
    reg[0, 0] = 0.0
    w = np.zeros(a.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-a @ w))
        step = np.linalg.solve((a * (p * (1 - p))[:, None]).T @ a + reg, a.T @ (p - y) + reg @ w)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] + x @ w[1:])))


def month_starts(start_ms: int, end_ms: int) -> list[int]:
    out, t = [], pd.Timestamp(start_ms, unit="ms")
    while t.value // 1_000_000 < end_ms:
        out.append(t.value // 1_000_000)
        t += pd.DateOffset(months=1)
    return out


def walk_forward(data: pd.DataFrame, horizon: int, train_start_ms: int, score_start_ms: int, end_ms: int,
                 daily_only: bool, features: list[str] = FEATURES, require: list[str] = (),
                 train_symbols: set[str] | None = None) -> pd.DataFrame:
    """Refits on the first of every month on rows whose answer is already known, and predicts that month.

    Rows missing any of `require` are dropped too, so two feature sets can be compared on the same rows.
    With `train_symbols` only those coins are learned from; every coin is still predicted (v6 transfer test).
    `thr` is the 90th percentile of |p - 0.5| on the training rows: the v6 "top 10% confidence" cutoff for that month
    (`thr95` and `thr98`: the v7 top 5% and top 2% cutoffs).
    """
    data = data.dropna(subset=list(features) + list(require) + ["fwd", "sd42"])
    data = data[(data["close_ms"] >= train_start_ms) & (data["close_ms"] + horizon * BAR_MS <= end_ms)]
    y_all = (data["fwd"] > 0).to_numpy(float)
    x_all = data[features].to_numpy(float)
    known = (data["close_ms"] + horizon * BAR_MS).to_numpy()
    when = data["close_ms"].to_numpy()
    learnable = np.ones(len(data), bool) if train_symbols is None else data["symbol"].isin(train_symbols).to_numpy()
    starts = month_starts(score_start_ms, end_ms)
    parts = []
    for m, nxt in zip(starts, starts[1:] + [end_ms]):
        train = (known <= m) & learnable
        test = (when >= m) & (when < nxt)
        if daily_only:
            test &= when % DAY_MS == 0
        if train.sum() < 100 or not test.any():
            continue
        mean, std = x_all[train].mean(axis=0), x_all[train].std(axis=0)
        std[std == 0] = 1.0
        scale = lambda x: np.clip((x - mean) / std, -CLIP, CLIP)
        w = fit(scale(x_all[train]), y_all[train])
        part = data[test].copy()
        part["p"] = predict(w, scale(x_all[test]))
        part["base"] = y_all[train].mean()
        part["thr"], part["thr95"], part["thr98"] = np.quantile(np.abs(predict(w, scale(x_all[train])) - 0.5), [0.9, 0.95, 0.98])
        part["month"] = m
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["y"] = (out["fwd"] > 0).astype(float)
    return out


def trades(pred: pd.DataFrame, horizon: int, side_cost: float = SIDE_COST, min_gain: float | None = None) -> pd.DataFrame:
    """Bets only when the expected gain |2p-1| * E|move| beats min_gain (v3: the round-trip cost; v4: a fixed 0.5%)."""
    expected = (2 * pred["p"] - 1).abs() * pred["sd42"] * math.sqrt(horizon) * ABS_MOVE
    t = pred[expected > (2 * side_cost if min_gain is None else min_gain)].copy()
    t["dir"] = np.where(t["p"] > 0.5, 1.0, -1.0)
    t["ret"] = t["dir"] * np.expm1(t["fwd"]) - 2 * side_cost - t["dir"] * t["fund_hold"]
    t["hit"] = ((t["dir"] > 0) == (t["y"] > 0)).astype(float)
    t["exit_ms"] = t["close_ms"] + horizon * BAR_MS
    return t


def equity_curve(t: pd.DataFrame, initial: float = 1000.0) -> list[tuple]:
    """Compounds at each exit; holdings never overlap because decisions are one horizon apart."""
    equity, curve = initial, []
    for exit_ms, ret in t.groupby("exit_ms")["ret"].sum().items():
        equity *= 1 + WEIGHT * ret
        curve.append((int(exit_ms), equity))
    return curve


def month_interval(num: pd.Series, den: pd.Series, rounds: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% interval of sum(num)/sum(den), resampling whole months."""
    num, den = num.to_numpy(float), den.to_numpy(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(num), size=(rounds, len(num)))
    ratios = np.sort(num[picks].sum(axis=1) / den[picks].sum(axis=1))
    return float(ratios[int(rounds * 0.025)]), float(ratios[int(rounds * 0.975)])


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
