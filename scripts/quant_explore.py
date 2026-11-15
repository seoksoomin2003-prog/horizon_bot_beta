"""Exploration for the quant architecture (docs/QUANT_ARCHITECTURE_KO.md). Seen data only: this picks candidates, it
is not evidence. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/quant_explore.py            signal IC table, then the combined book on the chosen signals
  .venv/bin/python scripts/quant_explore.py --smoke    the same on random signals, to check the plumbing

Rules fixed before running (written here so the choice is mechanical):
- Universe: each month the 30 coins with the most turnover in the year before (alpha.universe), among the 59.
- A signal is kept when, at the 7-day horizon, its monthly t-statistic is at least 3 in the expected direction and its
  yearly mean IC has that sign in at least 4 of the 6 years 2021-2026. (3, not 2: in the smoke run one of ten random
  signals reached t 2.8 and was kept.)
- The book trades the 7-day alpha (horizon 42 bars), rebalanced every 4h with the cost band.
- lambda is the one of LAMBDAS whose realized yearly volatility at cost x1 is closest to 20%; return is not looked at.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research import alpha, h4_data, ls_sim, portfolio, risk
from research.direction import _funding_sum
from research.h4_data import BAR_MS
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START
from scripts.direction_v9_dev import dataset

COUNT = 30
HORIZONS = (1, 6, 18, 42)
HORIZON = 42
YEARS = range(2021, 2027)
MIN_T, MIN_YEARS = 3.0, 4
LAMBDAS = (1.0, 2.5, 5.0, 10.0, 20.0)
TARGET_VOL = 0.20
OUT = h4_data.DATA_DIR / "quant_explore_result.json"


def reference() -> list[str]:
    return [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]


def load(s: str) -> tuple[np.ndarray, np.ndarray]:
    return h4_data.load_validation(s) if s in ORIGINAL else h4_data.load_v6(s)


def market_panels(coins: list[str], start_ms: int, end_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close at every 4h close from start_ms to end_ms and the funding summed over (close, next close]."""
    grid = np.arange(start_ms, end_ms + 1, BAR_MS)
    closes, funding = {}, {}
    for s in coins:
        bars, rates = load(s)
        closes[s] = pd.Series(bars[:, 4], index=bars[:, 0].astype(np.int64) + BAR_MS).reindex(grid)
        cum = np.concatenate([[0.0], np.cumsum(rates[:, 1])])
        funding[s] = _funding_sum(rates[:, 0].astype(np.int64), cum, grid, grid + BAR_MS)[0]
    return pd.DataFrame(closes, index=grid), pd.DataFrame(funding, index=grid)


def build(horizon: int, coins: list[str], member_rows: pd.DataFrame, sig: pd.DataFrame, bet: pd.DataFrame) -> pd.DataFrame:
    """The universe rows with signals, sigma and the target for one horizon."""
    data = dataset(horizon, coins, [])
    return alpha.prepare(data[data["close_ms"] < END], member_rows, sig, bet)


def main() -> None:
    os.nice(15)
    smoke = "--smoke" in sys.argv
    coins = reference()
    started = time.time()
    member_rows = alpha.members({s: load(s)[0] for s in coins}, TRAIN_START, END, COUNT)
    closes, funding = market_panels(coins, h4_data.WARMUP_START_MS, END)
    rets = risk.returns(closes)
    inside = alpha.membership(member_rows, closes.index.to_numpy(), coins)
    sig = risk.sigma(rets)
    bet = risk.beta(rets, risk.market_return(rets, inside))
    names = list(alpha.SIGNALS)
    out: dict = {"smoke": smoke, "universe_months": int(member_rows["month_ms"].nunique()),
                 "universe_coins_ever": sorted(member_rows["symbol"].unique()), "signals": {}, "horizons": list(HORIZONS)}
    rng = np.random.default_rng(11)
    datas = {}
    for h in HORIZONS:
        d = build(h, coins, member_rows, sig, bet)
        if smoke:
            for n in names:
                d[n] = rng.normal(size=len(d))
        datas[h] = d
        for n in names:
            ic = alpha.ic_by_close(d[d["close_ms"] >= TRAIN_START], n)
            out["signals"].setdefault(n, {})[str(h)] = alpha.ic_report(ic, h)
        print(f"horizon {h}: {len(d)} universe rows ({time.time() - started:.0f}s)", flush=True)
    week = datas[HORIZON]
    for n in names:
        out["signals"][n]["autocorr"] = {str(lag): alpha.autocorr(week, n, lag) for lag in (1, 6, 42)}
    scores = alpha.add_scores(week, names)
    zc = [f"z_{n}" for n in names]
    out["signal_corr"] = {a: {b: float(v) for b, v in row.items()} for a, row in
                          scores.groupby("close_ms")[zc].corr().groupby(level=1).mean().rename(lambda c: c[2:]).rename(columns=lambda c: c[2:]).to_dict("index").items()}

    print("\nIC of each signal with the target (return after funding minus the universe median), 30 largest coins, 2021-01 .. 2026-08")
    print(f"{'signal':>13} | " + " | ".join(f"{h * 4}h mean (t)" if h < 6 else f"{h // 6}d mean (t)" for h in HORIZONS)
          + " | 7d by year 2021..2026 | autocorr 4h/1d/7d")
    chosen = []
    for n in names:
        s = out["signals"][n]
        w = s[str(HORIZON)]
        years = [w["by_year"].get(y) for y in YEARS]
        good = sum(v is not None and v > 0 for v in years)
        keep = w["t_months"] >= MIN_T and good >= MIN_YEARS
        if keep:
            chosen.append(n)
        print(f"{n:>13} | " + " | ".join(f"{s[str(h)]['mean']:+.4f} ({s[str(h)]['t_months']:+.1f})" for h in HORIZONS)
              + " | " + " ".join(f"{v:+.3f}" if v is not None else "  -  " for v in years)
              + f" | {s['autocorr']['1']:.2f}/{s['autocorr']['6']:.2f}/{s['autocorr']['42']:.2f}" + ("  KEPT" if keep else ""), flush=True)
    out["chosen"] = chosen
    print(f"\nkept by the rule (t >= {MIN_T} and right sign in >= {MIN_YEARS} of 6 years at 7d): {chosen}")
    if smoke and not chosen:
        chosen = names[:3]
        print(f"(smoke: running the book on three random signals anyway: {chosen})")
    if not chosen:
        print("no signal kept: no book to run")
        return

    data = alpha.add_scores(week, chosen)
    pred = alpha.walk_forward(data, chosen, HORIZON, TRAIN_START, SCORE_START, END)
    combined = alpha.ic_by_close(pred.rename(columns={"alpha": "a"}), "a")
    out["combined_ic"] = alpha.ic_report(combined, HORIZON)
    print(f"combined alpha IC at 7d: {out['combined_ic']['mean']:+.4f} (t {out['combined_ic']['t_months']:+.1f}), by year "
          + " ".join(f"{y}:{v:+.3f}" for y, v in out["combined_ic"]["by_year"].items()))
    top = pred.groupby("close_ms")["alpha"].rank(pct=True)
    calib = {"top_alpha": float(pred.loc[top > 0.9, "alpha"].mean()), "top_target": float(pred.loc[top > 0.9, "target"].mean()),
             "bottom_alpha": float(pred.loc[top <= 0.1, "alpha"].mean()), "bottom_target": float(pred.loc[top <= 0.1, "target"].mean())}
    out["calibration"] = calib
    print(f"top tenth told {calib['top_alpha']:+.2%}, got {calib['top_target']:+.2%}; bottom tenth told {calib['bottom_alpha']:+.2%}, got {calib['bottom_target']:+.2%}")

    span = (closes.index >= SCORE_START) & (closes.index < END)
    c, f, s_, b_ = closes[span], funding[span], sig[span], bet[span]
    out["books"] = {}
    for lam in LAMBDAS:
        r = {}
        for k in (1, 2):
            bars = portfolio.run(pred, c, f, s_, b_, HORIZON, lam=lam, side_cost=k * ls_sim.SIDE_COST)
            summ = ls_sim.summary(bars, h4_data.DEV_END_MS)
            summ["vol_year"] = float(bars["ret"].std() * np.sqrt(6 * 365))
            summ["positions"] = float(bars["positions"].mean())
            summ["net_exposure"] = float(bars["net"].mean())
            r[f"cost{k}"] = summ
        out["books"][str(lam)] = r
        one, two = r["cost1"], r["cost2"]
        print(f"lambda {lam:>4}: vol {one['vol_year']:.0%}/yr, gross {one['gross']:.2f}, positions {one['positions']:.1f}, "
              f"turnover {one['turnover_per_day']:.2f}/day | net x1 {one['total']:+.1%}, x2 {two['total']:+.1%}, "
              f"month {one['month_mean']:+.2%} [{one['month_ci'][0]:+.2%}, {one['month_ci'][1]:+.2%}], "
              f"years " + " ".join(f"{v:+.0%}" for v in one["by_year"].values()) + f", recent {one['recent']:+.1%}, "
              f"drawdown {one['max_drawdown']:.0%}, beta {one['beta']:+.2f}, break-even {one['break_even_side_cost'] or 0:.3%}", flush=True)
    lam = min(LAMBDAS, key=lambda l: abs(out["books"][str(l)]["cost1"]["vol_year"] - TARGET_VOL))
    out["lambda"] = lam
    print(f"\nlambda by the volatility rule (closest to {TARGET_VOL:.0%} a year): {lam}")
    out["when"] = time.strftime("%FT%TZ", time.gmtime())
    if smoke:
        print("(smoke run on random signals: nothing saved, not a result)")
    else:
        OUT.write_text(json.dumps(out, indent=1, default=str))
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()
