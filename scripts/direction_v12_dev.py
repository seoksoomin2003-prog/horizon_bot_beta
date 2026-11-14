"""Runs the v12 test: does trading the relative model as long-short baskets leave money after costs?
See docs/research/direction_preregistration_v12.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v12_dev.py            the registered run
  .venv/bin/python scripts/direction_v12_dev.py --smoke    trades random numbers instead of the model, to check the plumbing
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

from research import h4_data, ls_sim
from research.direction import FEATURES, MARKET_FEATURES, RANK_FEATURES, _funding_sum, add_ranks, add_relative, walk_forward
from research.h4_data import BAR_MS
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START
from scripts.direction_v9_dev import HORIZONS, YEARS, dataset

MODEL = FEATURES + MARKET_FEATURES + RANK_FEATURES
CANDIDATES = {"A8": ("8h", None), "B8": ("8h", 2), "A4": ("4h", None), "B4": ("4h", 2)}      # K=None: a tenth of the coins
MAX_DRAWDOWN, MAX_BETA, GOOD_YEARS = 0.30, 0.3, 4


def panels(symbols: list[str], new: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price at every 4h close of the trading period and the funding rates summed over (close, next close]."""
    load = lambda s: h4_data.load_validation(s) if s in ORIGINAL else h4_data.load_v9(s) if s in new else h4_data.load_v6(s)
    grid = np.arange(SCORE_START, END + 1, BAR_MS)
    closes, funding = {}, {}
    for s in symbols:
        bars, rates = load(s)
        closes[s] = pd.Series(bars[:, 4], index=bars[:, 0].astype(np.int64) + BAR_MS).reindex(grid)
        cum = np.concatenate([[0.0], np.cumsum(rates[:, 1])])
        funding[s] = _funding_sum(rates[:, 0].astype(np.int64), cum, grid, grid + BAR_MS)[0]
    return pd.DataFrame(closes, index=grid), pd.DataFrame(funding, index=grid)


def predictions(horizon: int, reference: list[str], new: list[str], smoke: bool) -> pd.DataFrame:
    """The v11 rank model's p for every coin and close, fitted on the reference coins exactly as in v11."""
    data = add_ranks(add_relative(dataset(horizon, reference, new), set(reference)), set(reference))
    if smoke:
        rows = data.dropna(subset=MODEL + ["fwd", "sd42"])
        rows = rows[(rows["close_ms"] >= SCORE_START) & (rows["close_ms"] + horizon * BAR_MS <= END)].copy()
        rows["p"] = np.random.default_rng(7).random(len(rows))
        return rows[["symbol", "close_ms", "p"]]
    pred = walk_forward(data, horizon, TRAIN_START, SCORE_START, END, False, features=MODEL, require=RANK_FEATURES,
                        train_symbols=set(reference))
    return pred[["symbol", "close_ms", "p"]]


def checks(new1: dict, new2: dict, seen1: dict) -> dict:
    good_years = sum(new1["by_year"].get(y, 0.0) > 0 for y in YEARS)
    return {"G1 monthly mean and CI low above 0": bool(new1["month_mean"] > 0 and new1["month_ci"][0] > 0),
            "G2 net above 0 at double cost": bool(new2["total"] > 0),
            "G3 four of five years and lately in profit": bool(good_years >= GOOD_YEARS and new1["recent"] > 0),
            "G4 drawdown within 30%": bool(new1["max_drawdown"] <= MAX_DRAWDOWN),
            "G5 |beta| within 0.3": bool(abs(new1["beta"]) <= MAX_BETA),
            "G6 net above 0 on the 59 seen coins": bool(seen1["total"] > 0)}


def show(label: str, s: dict) -> None:
    be = s["break_even_side_cost"]
    print(f"   {label}: net {s['total']:+.1%} | month {s['month_mean']:+.2%} [{s['month_ci'][0]:+.2%} ~ {s['month_ci'][1]:+.2%}], "
          f"sharpe {s['month_sharpe']:+.2f}, months up {s['positive_months']:.0%} | "
          + " ".join(f"{y} {v:+.1%}" for y, v in s["by_year"].items()) + f" | since 2025-11 {s['recent']:+.1%}\n"
          f"      drawdown {s['max_drawdown']:.1%}, beta {s['beta']:+.2f}, turnover/day {s['turnover_per_day']:.2f}, gross {s['gross']:.2f}, "
          f"break-even side cost {'n/a' if be is None else f'{be:+.3%}'} | parts: long {s['parts']['long']:+.1%}, "
          f"short {s['parts']['short']:+.1%}, funding {s['parts']['funding']:+.1%}, cost {s['parts']['cost']:+.1%}", flush=True)


def main() -> None:
    os.nice(15)
    smoke = "--smoke" in sys.argv
    reference = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
    new = h4_data.choose_v9_symbols()
    closes, funding = panels(reference + new, new)
    pred = {name: predictions(h, reference, new, smoke) for name, h in HORIZONS.items()}
    out: dict = {"smoke": smoke, "reference_coins": len(reference), "new_coins": len(new), "candidates": {}}
    passed = []
    for name, (model, k) in CANDIDATES.items():
        h, r = HORIZONS[model], {}
        for universe, coins in (("new", new), ("seen", reference)):
            p = pred[model][pred[model]["symbol"].isin(coins)]
            for cost in (1, 2):
                bars = ls_sim.run(p, closes[coins], funding[coins], h, k, side_cost=cost * ls_sim.SIDE_COST)
                r[f"{universe}_cost{cost}"] = ls_sim.summary(bars, h4_data.DEV_END_MS)
        r["checks"] = checks(r["new_cost1"], r["new_cost2"], r["seen_cost1"])
        out["candidates"][name] = r
        print(f"[{name}] {model} model, {'a tenth of the coins' if k is None else f'{k} coins'} a side")
        show("46 new coins, cost x1", r["new_cost1"])
        show("46 new coins, cost x2", r["new_cost2"])
        show("59 seen coins, cost x1", r["seen_cost1"])
        show("59 seen coins, cost x2", r["seen_cost2"])
        print("   checks " + ", ".join(f"{c.split()[0]} {'PASS' if v else 'FAIL'}" for c, v in r["checks"].items()) + "\n", flush=True)
        if all(r["checks"].values()):
            passed.append(name)
    best = max(passed, key=lambda n: out["candidates"][n]["new_cost2"]["month_sharpe"]) if passed else None
    out.update(passed=passed, chosen=best, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed G1-G6: {passed}\nverdict: " + (f"PASS_FOR_PAPER ({best})" if best else "NO_VERIFIED_PROFIT_EDGE"))
    if smoke:
        print("(smoke run on random numbers: nothing saved, not a result)")
    else:
        (h4_data.DATA_DIR / "direction_v12_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
