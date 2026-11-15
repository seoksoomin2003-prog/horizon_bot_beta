"""Quant architecture, alpha layer: simple signals measured one by one, then combined into an expected return.

See docs/QUANT_ARCHITECTURE_KO.md.

- Each signal is one number per coin and close, computed from data up to that close (research/direction.frame's
  features, plus the risk layer's beta). `SIGNALS` gives the column and the sign that makes "higher = expected to do
  better"; the exploration measures whether that holds.
- The universe is point-in-time: on the first of every month, the `count` coins with the most turnover in the year
  before, among those already listed for half a year.
- The target is what a long position would have earned over the horizon after funding, minus the universe's median
  at that close. Ranking coins on it is ranking what a market-neutral book earns.
- A signal is turned into a rank score per close (uniform, unit variance). Its IC is the per-close rank correlation
  with the target.
- AlphaModel combines the chosen scores with a ridge regression of the target divided by the coin's volatility over
  the horizon, refit monthly; its prediction times that volatility is the expected relative return (alpha) in
  return units, which the portfolio layer weighs against the trading cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.direction import month_starts
from research.h4_data import BAR_MS
from research.size_model import TRIM, fit

DAY_MS = 86_400_000
SIGNALS = {                         # name: (column, sign)
    "mom_28d": ("r168", 1),         # momentum over four weeks
    "mom_7d": ("r42", 1),           # momentum over a week
    "rev_1d": ("r6", -1),           # reversal of the last day
    "rev_4h": ("r1", -1),           # reversal of the last bar
    "carry": ("funding", -1),       # a long earns the funding when it is negative
    "low_vol": ("vol", -1),         # calmer coins
    "low_beta": ("beta", -1),       # coins that move less with the market
    "volume_surge": ("volume", 1),  # the last day's volume against the week's
    "range_pos": ("range_pos", 1),  # where the close sits in the week's range
    "calming": ("vol_change", -1),  # the last day calmer than the week
}
LISTED_DAYS = 180
TURNOVER_DAYS = 365
RIDGE = 10.0
MIN_TRAIN = 1000
MIN_COINS = 10                      # a close with fewer universe coins gets no scores and no IC


def universe(bars_by_symbol: dict[str, np.ndarray], month_ms: int, count: int) -> list[str]:
    """The `count` coins with the most turnover (close x volume) in the TURNOVER_DAYS before `month_ms`, among the
    coins whose first bar is at least LISTED_DAYS before it and that still traded in the day before it."""
    start = month_ms - TURNOVER_DAYS * DAY_MS
    turnover = {}
    for s, b in bars_by_symbol.items():
        if not len(b) or b[0, 0] > month_ms - LISTED_DAYS * DAY_MS or not ((b[:, 0] >= month_ms - DAY_MS) & (b[:, 0] < month_ms)).any():
            continue
        w = b[(b[:, 0] >= start) & (b[:, 0] < month_ms)]
        turnover[s] = float((w[:, 4] * w[:, 5]).sum())
    return sorted(turnover, key=lambda s: (-turnover[s], s))[:count]


def members(bars_by_symbol: dict[str, np.ndarray], start_ms: int, end_ms: int, count: int) -> pd.DataFrame:
    """symbol, month_ms: the universe of every month from the month of start_ms to end_ms."""
    first = int(pd.Timestamp(start_ms, unit="ms").to_period("M").start_time.value // 1_000_000)
    return pd.DataFrame([{"symbol": s, "month_ms": m} for m in month_starts(first, end_ms)
                         for s in universe(bars_by_symbol, m, count)])


def month_of(close_ms: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_datetime(np.asarray(close_ms), unit="ms").to_period("M").start_time.values.astype("datetime64[ms]").astype(np.int64)


def in_universe(data: pd.DataFrame, member_rows: pd.DataFrame) -> pd.Series:
    """True for the rows whose coin was in the universe of the month of its close."""
    keys = set(zip(member_rows["symbol"], member_rows["month_ms"]))
    return pd.Series([k in keys for k in zip(data["symbol"], month_of(data["close_ms"]))], index=data.index)


def membership(member_rows: pd.DataFrame, index: np.ndarray, coins: list[str]) -> pd.DataFrame:
    """index close_ms x coins: True where the coin is in the universe of the close's month."""
    month = month_of(index)
    keys = set(zip(member_rows["symbol"], member_rows["month_ms"]))
    return pd.DataFrame({s: [(s, m) in keys for m in month] for s in coins}, index=index)


def prepare(data: pd.DataFrame, member_rows: pd.DataFrame, sigma: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """direction.frame() rows of every coin -> the universe rows with sigma, beta, the signals and the target."""
    stack = lambda p, name: p.stack(future_stack=True).rename(name).rename_axis(["close_ms", "symbol"]).reset_index()
    data = data.merge(stack(sigma, "sigma"), on=["close_ms", "symbol"], how="left").merge(stack(beta, "beta"), on=["close_ms", "symbol"], how="left")
    data = data[in_universe(data, member_rows).to_numpy()]
    return add_signals(add_target(data))


def add_signals(data: pd.DataFrame) -> pd.DataFrame:
    """The raw signal columns, oriented by SIGNALS so that higher is expected to be better."""
    out = data.copy()
    for name, (column, sign) in SIGNALS.items():
        out[name] = sign * out[column] if column in out else np.nan
    return out


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """`net`: a long's simple return over the horizon after funding (needs fwd and fund_hold from frame()), and
    `target`: net minus the median net of the rows at the same close. Pass the universe rows only."""
    out = data.copy()
    out["net"] = np.expm1(out["fwd"]) - out["fund_hold"]
    out["target"] = out["net"] - out.groupby("close_ms")["net"].transform("median")
    return out


def add_scores(data: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """`z_<name>`: the signal's rank among the rows at the same close as a uniform score with unit variance
    (-1.73 lowest, +1.73 highest); a missing signal scores 0. Closes with fewer than MIN_COINS rows score NaN."""
    out = data.copy()
    size = out.groupby("close_ms")["symbol"].transform("size")
    for name in names:
        pct = out.groupby("close_ms")[name].rank(pct=True, method="average")
        n = out.groupby("close_ms")[name].transform("count")
        z = ((pct * n - 1) / (n - 1).where(n > 1) - 0.5) * np.sqrt(12)      # the rank mapped to 0 .. 1, centred
        out[f"z_{name}"] = z.fillna(0.0).where(size >= MIN_COINS)
    return out


def ic_by_close(data: pd.DataFrame, name: str, target: str = "target") -> pd.Series:
    """The rank correlation of the signal and the target at every close with at least MIN_COINS answered rows."""
    rows = data.loc[data[name].notna() & data[target].notna(), ["close_ms", name, target]]
    g = rows.groupby("close_ms")
    x, y = g[name].rank(), g[target].rank()
    x, y = x - x.groupby(rows["close_ms"]).transform("mean"), y - y.groupby(rows["close_ms"]).transform("mean")
    by = rows["close_ms"]
    sums = pd.DataFrame({"xy": x * y, "xx": x * x, "yy": y * y, "n": 1.0}).groupby(by.to_numpy()).sum()
    ic = sums["xy"] / np.sqrt(sums["xx"] * sums["yy"])
    ic.index.name = "close_ms"
    return ic[(sums["n"] >= MIN_COINS) & (sums["xx"] > 0) & (sums["yy"] > 0)]


def ic_report(ic: pd.Series, horizon: int) -> dict:
    """Mean IC, its t-statistic over months (closes within a month overlap; months mostly do not), by year,
    and the mean over non-overlapping closes only (every `horizon`-th close)."""
    when = pd.to_datetime(ic.index, unit="ms")
    monthly = ic.groupby(when.to_period("M")).mean()
    t = float(monthly.mean() / monthly.std() * np.sqrt(len(monthly))) if len(monthly) > 1 and monthly.std() > 0 else 0.0
    spaced = ic.iloc[::max(horizon, 1)]
    return {"mean": float(ic.mean()), "t_months": t, "months": int(len(monthly)), "positive_months": float((monthly > 0).mean()),
            "by_year": {int(y): float(v) for y, v in ic.groupby(when.year).mean().items()},
            "spaced_mean": float(spaced.mean()), "spaced_n": int(len(spaced))}


def autocorr(data: pd.DataFrame, name: str, lag: int) -> float:
    """Mean rank correlation of a signal with itself `lag` bars later, over coins present at both closes: how slowly
    the signal changes (1 = never, so it needs little trading)."""
    wide = data.pivot_table(index="close_ms", columns="symbol", values=name)
    ranks = wide.rank(axis=1)
    later = ranks.shift(-lag)
    both = ranks.notna() & later.notna()
    a, b = ranks.where(both), later.where(both)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    c = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return float(c[both.sum(axis=1) >= MIN_COINS].mean())


class AlphaModel:
    """Ridge regression of target / (sigma * sqrt(horizon)) on the scores of `names`, fitted on answered rows.

    predict() returns alpha: the expected return over the horizon relative to the universe's median coin, after
    funding, in return units. Rows need `sigma` (per-bar volatility from the risk layer) and the z_ columns.

    `k` calibrates it: the slope of the target on the raw prediction over the same training rows, so that alpha is
    in the target's own units. Without it the exploration found the spread between the highest and lowest tenth
    three times larger than what followed (the fit is on volatility-scaled returns, the mean is not).
    """

    def __init__(self, names: list[str], horizon: int, ridge: float = RIDGE, trim: tuple[float, float] | None = TRIM) -> None:
        self.names, self.horizon, self.ridge, self.trim = list(names), horizon, ridge, trim
        self.columns = [f"z_{n}" for n in self.names]
        self.month: int | None = None
        self.train_rows = 0

    def scale(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["sigma"].to_numpy(float) * np.sqrt(self.horizon)

    def fit(self, rows: pd.DataFrame, month_ms: int) -> None:
        """Fits on the rows whose answer was known by month_ms."""
        rows = rows.dropna(subset=self.columns + ["target", "sigma"])
        rows = rows[(rows["close_ms"] + self.horizon * BAR_MS <= month_ms) & (rows["sigma"] > 0)]
        if len(rows) < MIN_TRAIN:
            raise ValueError(f"only {len(rows)} answered rows before the month")
        y = rows["target"].to_numpy(float) / self.scale(rows)
        if self.trim is not None:
            y = np.clip(y, *np.quantile(y, self.trim))
        self.w, self.intercept = fit(rows[self.columns].to_numpy(float), y, self.ridge)
        self.k = 1.0
        raw = self.predict(rows)
        self.k = float(np.clip(np.cov(raw, rows["target"])[0, 1] / raw.var(), 0.0, 2.0)) if raw.var() > 0 else 0.0
        self.month, self.train_rows = month_ms, len(rows)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """The intercept is left out: it only says how far the average coin sits above the median coin, and times
        sigma it would turn every coin's volatility into a bet (found by the smoke run on random signals)."""
        return self.k * (rows[self.columns].to_numpy(float) @ self.w) * self.scale(rows)


def walk_forward(data: pd.DataFrame, names: list[str], horizon: int, train_start_ms: int, score_start_ms: int,
                 end_ms: int, ridge: float = RIDGE, trim: tuple[float, float] | None = TRIM) -> pd.DataFrame:
    """Refits AlphaModel on the first of every month and predicts that month's closes: symbol, close_ms, alpha,
    target (NaN when unanswered). `data` holds the universe rows with scores, sigma and target."""
    data = data[(data["close_ms"] >= train_start_ms) & (data["close_ms"] < end_ms)]
    model, parts = AlphaModel(names, horizon, ridge, trim), []
    starts = month_starts(score_start_ms, end_ms)
    for m, nxt in zip(starts, starts[1:] + [end_ms]):
        test = data[(data["close_ms"] >= m) & (data["close_ms"] < nxt)].dropna(subset=model.columns + ["sigma"])
        if test.empty:
            continue
        model.fit(data, m)
        parts.append(test[["symbol", "close_ms", "target"]].assign(alpha=model.predict(test)))
    return pd.concat(parts, ignore_index=True)
