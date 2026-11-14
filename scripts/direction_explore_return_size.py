"""Exploration after v13 (seen data: the 59 reference coins only, NOT evidence, no gates): train on the size of the
relative return instead of its sign, and keep a coin while it stays in a band. See the last section of
docs/research/direction_result_v13.md. Anything this suggests needs its own registration on coins never used.

  .venv/bin/python scripts/direction_explore_return_size.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from research import h4_data, ls_sim
from research.direction import CLIP, FEATURES, MARKET_FEATURES, RANK_FEATURES, add_ranks, add_relative, month_starts
from research.h4_data import BAR_MS
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START
from scripts.direction_v9_dev import dataset
from scripts.direction_v12_dev import panels

os.nice(15)
MODEL = FEATURES + MARKET_FEATURES + RANK_FEATURES
ref = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
closes, funding = panels(ref, [])

def ridge_walk(data, h, target, lam=10.0):
    data = data.dropna(subset=MODEL + ["fwd", "sd42"])
    data = data[(data["close_ms"] >= TRAIN_START) & (data["close_ms"] + h * BAR_MS <= END)]
    x_all, when, known = data[MODEL].to_numpy(float), data["close_ms"].to_numpy(), (data["close_ms"] + h * BAR_MS).to_numpy()
    y_all = target(data)
    starts, parts = month_starts(SCORE_START, END), []
    for m, nxt in zip(starts, starts[1:] + [END]):
        train, test = known <= m, (when >= m) & (when < nxt)
        if train.sum() < 100 or not test.any():
            continue
        mean, std = x_all[train].mean(0), x_all[train].std(0); std[std == 0] = 1.0
        scale = lambda x: np.clip((x - mean) / std, -CLIP, CLIP)
        lo, hi = np.quantile(y_all[train], [0.01, 0.99])
        xt, yt = scale(x_all[train]), np.clip(y_all[train], lo, hi)
        mu = yt.mean()
        w = np.linalg.solve(xt.T @ xt + lam * np.eye(xt.shape[1]), xt.T @ (yt - mu))
        part = data.loc[test, ["symbol", "close_ms", "fwd", "sd42"]].copy()
        part["p"] = scale(x_all[test]) @ w + mu
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def held(label, pred, h, keep):
    out = []
    for cost in (1, 2):
        s = ls_sim.summary(ls_sim.run_held(pred[["symbol", "close_ms", "p"]], closes, funding, h, None, keep, side_cost=cost * ls_sim.SIDE_COST), h4_data.DEV_END_MS)
        out.append(s)
    s, d = out
    pre = sum(s["parts"][k] for k in ("long", "short", "funding"))
    print(f"  {label:<26} before cost {pre:+5.0%} | turnover/day {s['turnover_per_day']:.2f} | break-even {s['break_even_side_cost']:+.3%} | net {s['total']:+.0%}, at double cost {d['total']:+.0%} | "
          f"month {s['month_mean']:+.2%} [{s['month_ci'][0]:+.2%} ~ {s['month_ci'][1]:+.2%}] | drawdown {s['max_drawdown']:.0%} | beta {s['beta']:+.2f} | "
          f"years " + " ".join(f"{v:+.0%}" for v in s["by_year"].values()) + f" | recent {s['recent']:+.0%}", flush=True)

for h in (2, 6, 18):
    data = add_ranks(add_relative(dataset(h, ref, []), set(ref)), set(ref))
    raw = ridge_walk(data, h, lambda d: d["fwd"].to_numpy(float))
    print(f"[{h * 4}h] ridge on the relative return, 59 seen coins")
    for keep in (0.1, 0.3, 0.5):
        held(f"keep within {keep:.0%}", raw, h, keep)
