"""v14: predicts how much a coin will beat the median coin, and until when. See docs/research/direction_preregistration_v14.md.

One ridge regression per holding period on the v11 features; the target is the relative return itself, not its sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.direction import CLIP, month_starts
from research.h4_data import BAR_MS

LAMBDA = 10.0
TRIM = (0.01, 0.99)                                     # the target is cut at these quantiles of the training rows
PERIODS = {"8h": 2, "24h": 6, "3d": 18, "7d": 42}       # holding periods in 4h bars
TURNOVER_DAYS = 365


def largest(bars_by_symbol: dict[str, np.ndarray], end_ms: int, count: int) -> list[str]:
    """The `count` coins with the highest mean daily turnover (close x volume) over the TURNOVER_DAYS before end_ms."""
    start = end_ms - TURNOVER_DAYS * 86_400_000
    turnover = {}
    for symbol, bars in bars_by_symbol.items():
        b = bars[(bars[:, 0] >= start) & (bars[:, 0] < end_ms)]
        turnover[symbol] = float((b[:, 4] * b[:, 5]).sum())
    return sorted(turnover, key=lambda s: (-turnover[s], s))[:count]


def fit(x: np.ndarray, y: np.ndarray, lam: float = LAMBDA) -> tuple[np.ndarray, float]:
    """Ridge weights and intercept; x is already standardized."""
    mean = float(y.mean())
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ (y - mean)), mean


def walk_forward(data: pd.DataFrame, horizon: int, train_start_ms: int, score_start_ms: int, end_ms: int,
                 features: list[str]) -> pd.DataFrame:
    """Refits on the first of every month on rows whose `horizon`-bar answer is already known, and predicts that month.

    Unlike the sign model's walk-forward, a row is predicted whenever its features exist, whether or not its answer
    ever becomes known, so the last closes of the data are predicted too. `size` is the predicted relative return.
    """
    data = data.dropna(subset=list(features))
    data = data[(data["close_ms"] >= train_start_ms) & (data["close_ms"] < end_ms)]
    x_all, y_all = data[features].to_numpy(float), data["fwd"].to_numpy(float)
    when = data["close_ms"].to_numpy()
    answered = np.isfinite(y_all) & (when + horizon * BAR_MS <= end_ms)
    starts, parts = month_starts(score_start_ms, end_ms), []
    for m, nxt in zip(starts, starts[1:] + [end_ms]):
        train = answered & (when + horizon * BAR_MS <= m)
        test = (when >= m) & (when < nxt)
        if train.sum() < 100 or not test.any():
            continue
        mean, std = x_all[train].mean(axis=0), x_all[train].std(axis=0)
        std[std == 0] = 1.0
        scale = lambda x: np.clip((x - mean) / std, -CLIP, CLIP)
        low, high = np.quantile(y_all[train], TRIM)
        w, intercept = fit(scale(x_all[train]), np.clip(y_all[train], low, high))
        part = data.loc[test, ["symbol", "close_ms", "fwd"]].copy()
        part["size"] = scale(x_all[test]) @ w + intercept
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def combine(sizes: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One row per coin and close that has every period's prediction: `up` is the largest predicted relative gain over
    the periods and `up_bars` the period it belongs to ("how much, until when"); `down` and `down_bars` are the largest
    predicted relative loss. Equal predictions take the shorter period."""
    periods = sorted(sizes)
    wide = pd.concat([sizes[h].set_index(["symbol", "close_ms"])["size"].rename(h) for h in periods], axis=1).dropna()
    values = wide.to_numpy(float)
    best, worst = values.argmax(axis=1), values.argmin(axis=1)
    rows = np.arange(len(wide))
    out = pd.DataFrame({"up": values[rows, best], "up_bars": np.array(periods)[best],
                        "down": values[rows, worst], "down_bars": np.array(periods)[worst]}, index=wide.index)
    return out.reset_index()
