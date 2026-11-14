"""Runs the v14 test: "how much, until when" on the 30 largest coins, held for the predicted period.
See docs/research/direction_preregistration_v14.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v14_dev.py            the registered run
  .venv/bin/python scripts/direction_v14_dev.py --smoke    trades random numbers instead of the model, to check the plumbing
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

from research import h4_data, ls_sim, size_model
from research.direction import FEATURES, MARKET_FEATURES, RANK_FEATURES, add_ranks, add_relative
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START
from scripts.direction_v9_dev import YEARS, dataset
from scripts.direction_v12_dev import panels, show

MODEL = FEATURES + MARKET_FEATURES + RANK_FEATURES
CANDIDATES = {"P10": 0.1, "P50": 0.5}           # the band a coin must be in to stay past its planned exit
LARGE = 30
MAX_DRAWDOWN, MAX_BETA, GOOD_YEARS = 0.30, 0.3, 4


def checks(one: dict, two: dict) -> dict:
    good_years = sum(one["by_year"].get(y, 0.0) > 0 for y in YEARS)
    return {"G1 monthly mean and CI low above 0": bool(one["month_mean"] > 0 and one["month_ci"][0] > 0),
            "G2 net above 0 at double cost": bool(two["total"] > 0),
            "G3 four of five years and lately in profit": bool(good_years >= GOOD_YEARS and one["recent"] > 0),
            "G4 drawdown within 30%": bool(one["max_drawdown"] <= MAX_DRAWDOWN),
            "G5 |beta| within 0.3": bool(abs(one["beta"]) <= MAX_BETA)}


def calibration(sizes: dict[int, pd.DataFrame], coins: list[str]) -> dict:
    """Per period: what the tenth with the highest and the lowest predictions was told and what it got."""
    out = {}
    for h, pred in sizes.items():
        p = pred[pred["symbol"].isin(coins)].dropna(subset=["fwd"])
        rank = p.groupby("close_ms")["size"].rank(pct=True)
        top, bottom = p[rank > 0.9], p[rank <= 0.1]
        out[h] = {"top_predicted": float(top["size"].mean()), "top_actual": float(top["fwd"].mean()),
                  "top_right": float((top["fwd"] > 0).mean()), "bottom_predicted": float(bottom["size"].mean()),
                  "bottom_actual": float(bottom["fwd"].mean()), "bottom_right": float((bottom["fwd"] < 0).mean()),
                  "rank_corr": float(p.groupby("close_ms").apply(lambda g: g["size"].rank().corr(g["fwd"].rank())).mean())}
    return out


def main() -> None:
    os.nice(15)
    smoke = "--smoke" in sys.argv
    reference = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
    load = lambda s: h4_data.load_validation(s) if s in ORIGINAL else h4_data.load_v6(s)
    large = size_model.largest({s: load(s)[0] for s in reference}, END, LARGE)
    closes, funding = panels(reference, [])
    sizes = {}
    for h in size_model.PERIODS.values():
        data = add_ranks(add_relative(dataset(h, reference, []), set(reference)), set(reference))
        sizes[h] = size_model.walk_forward(data, h, TRAIN_START, SCORE_START, END, MODEL)
        if smoke:
            sizes[h]["size"] = np.random.default_rng(h).normal(0, 0.01 * np.sqrt(h), len(sizes[h]))
    pred = size_model.combine(sizes)
    out: dict = {"smoke": smoke, "large": large, "candidates": {}, "reference": {}}

    def trade(coins: list[str], keep: float, cost: int) -> dict:
        bars = ls_sim.run_planned(pred[pred["symbol"].isin(coins)], closes[coins], funding[coins], keep, side_cost=cost * ls_sim.SIDE_COST)
        s = ls_sim.summary(bars, h4_data.DEV_END_MS)
        entries = max(int(bars["entries"].sum()), 1)
        return s | {"entries": entries, "planned_days": float(bars["planned_bars"].sum() / entries / 6),
                    "kept_share": float(bars["kept"].sum() / max(bars["kept"].sum() + bars["exits"].sum(), 1))}

    def line(label: str, s: dict) -> None:
        show(label, s)
        print(f"      entries {s['entries']}, planned holding {s['planned_days']:.1f} days on average, "
              f"kept at the planned exit {s['kept_share']:.0%}", flush=True)

    print(f"the {LARGE} largest coins: {' '.join(c[:-4] for c in large)}\n")
    passed = []
    for name, keep in CANDIDATES.items():
        r = {"large_cost1": trade(large, keep, 1), "large_cost2": trade(large, keep, 2),
             "all_cost1": trade(reference, keep, 1), "all_cost2": trade(reference, keep, 2)}
        r["checks"] = checks(r["large_cost1"], r["large_cost2"])
        out["candidates"][name] = r
        print(f"[{name}] stays past its planned exit while within {keep:.0%}")
        line("30 largest coins, cost x1", r["large_cost1"])
        line("30 largest coins, cost x2", r["large_cost2"])
        line("(reference) all 59 coins, cost x1", r["all_cost1"])
        line("(reference) all 59 coins, cost x2", r["all_cost2"])
        print("   checks " + ", ".join(f"{c.split()[0]} {'PASS' if v else 'FAIL'}" for c, v in r["checks"].items()) + "\n", flush=True)
        if all(r["checks"].values()):
            passed.append(name)

    # reference only: the best combination of the exploration (the 3-day model alone, kept while within 50%) on the 30 largest
    three = sizes[size_model.PERIODS["3d"]]
    three = three[three["symbol"].isin(large)].rename(columns={"size": "p"})[["symbol", "close_ms", "p"]]
    fixed = {f"cost{c}": ls_sim.summary(ls_sim.run_held(three, closes[large], funding[large], size_model.PERIODS["3d"], None, 0.5,
                                                        side_cost=c * ls_sim.SIDE_COST), h4_data.DEV_END_MS) for c in (1, 2)}
    out["reference"]["three_day_keep_50"] = fixed
    print("[reference only] the 3-day model alone, 18 baskets, kept while within 50%, 30 largest coins")
    show("cost x1", fixed["cost1"])
    show("cost x2", fixed["cost2"])

    out["reference"]["calibration"] = cal = calibration(sizes, large)
    periods = {v: k for k, v in size_model.PERIODS.items()}
    print("\n[reference only] how well the sizes are predicted, 30 largest coins (the tenth with the highest / lowest prediction at each close)")
    for h, c in cal.items():
        print(f"   {periods[h]:>3}: highest tenth told {c['top_predicted']:+.2%}, got {c['top_actual']:+.2%} (above the median {c['top_right']:.1%}) | "
              f"lowest tenth told {c['bottom_predicted']:+.2%}, got {c['bottom_actual']:+.2%} (below the median {c['bottom_right']:.1%}) | "
              f"rank corr {c['rank_corr']:+.3f}")
    picked = pred[pred["symbol"].isin(large)]
    share = {periods[h]: float((picked["up_bars"] == h).mean()) for h in sorted(periods)}
    out["reference"]["up_period_share"] = share
    print("   period with the largest predicted gain: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))

    best = max(passed, key=lambda n: out["candidates"][n]["large_cost2"]["month_sharpe"]) if passed else None
    out.update(passed=passed, chosen=best, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"\npassed G1-G5: {passed}\nverdict: " + (f"PASS_DEV ({best}): a candidate for the final check, which needs the user's approval"
                                                     if best else "NO_VERIFIED_PROFIT_EDGE"))
    if smoke:
        print("(smoke run on random numbers: nothing saved, not a result)")
    else:
        (h4_data.DATA_DIR / "direction_v14_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
