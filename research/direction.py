"""Direction model for docs/research/direction_preregistration_v3.md. Parameters are fixed there; do not tune."""

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


def _funding_sum(fts: np.ndarray, cum: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of funding events with start < ts <= end."""
    a, b = np.searchsorted(fts, start, side="right"), np.searchsorted(fts, end, side="right")
    return cum[b] - cum[a], b - a


def frame(symbol: str, bars: np.ndarray, funding: np.ndarray, btc: pd.DataFrame | None, horizon: int) -> pd.DataFrame:
    """One row per 4h bar close: the 13 features from data up to that close, the forward log return and the funding paid while held."""
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
    out["fwd"] = logc.shift(-horizon) - logc
    out["fund_hold"], _ = _funding_sum(fts, cum, grid, grid + horizon * BAR_MS)
    out["symbol"] = symbol
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.rename_axis("close_ms").reset_index()


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
                 daily_only: bool) -> pd.DataFrame:
    """Refits on the first of every month on rows whose answer is already known, and predicts that month."""
    data = data.dropna(subset=FEATURES + ["fwd", "sd42"])
    data = data[(data["close_ms"] >= train_start_ms) & (data["close_ms"] + horizon * BAR_MS <= end_ms)]
    y_all = (data["fwd"] > 0).to_numpy(float)
    x_all = data[FEATURES].to_numpy(float)
    known = (data["close_ms"] + horizon * BAR_MS).to_numpy()
    when = data["close_ms"].to_numpy()
    starts = month_starts(score_start_ms, end_ms)
    parts = []
    for m, nxt in zip(starts, starts[1:] + [end_ms]):
        train = known <= m
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
