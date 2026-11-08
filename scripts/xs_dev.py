"""Scores the v2 cross-sectional rules on the development period against the fixed gates. Never touches validation data.

  .venv/bin/python scripts/xs_dev.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from research import h4_data
from research.h4_sim import block_interval, max_drawdown, sharpe, year_log_returns
from research.xs_sim import RULES, SIDE_COST, UNIVERSE, Coin, run

START, END = h4_data.utc_ms("2021-01-04"), h4_data.DEV_END_MS


def score(key: str, coins: list) -> dict:
    one, two = run(key, coins, START, END, SIDE_COST), run(key, coins, START, END, SIDE_COST * 2)
    if one["ruined_ms"] is not None:
        when = time.strftime("%F", time.gmtime(one["ruined_ms"] / 1000))
        return {"rule": RULES[key], "ruined": when, "weeks": len(one["weeks"]), "traded_weeks": one["traded_weeks"],
                "hit_rate": one["hit_rate"], "return": -1.0, "return_2x": two["cash"] / 1000 - 1,
                "gates": {"G1 net>0 at 1x and 2x": False, "G2 weekly CI low>0": False, "G3 weeks>=150": False,
                          "G4 majority years and without best>0": False, "G5 mdd<=30%": False, "G6 |beta|<=0.3": False}}
    weeks, bench = one["weeks"], one["bench"]
    low, high = block_interval(weeks)
    years = year_log_returns(one["curve"], 1000.0)
    full = [y for y, row in years.items() if row["months"] >= 6]
    wins = sum(years[y]["log"] > 0 for y in full)
    without_best = sum(r["log"] for r in years.values()) - max(r["log"] for r in years.values())
    beta = float(np.polyfit(bench, weeks, 1)[0])
    r = {"rule": RULES[key], "weeks": len(weeks), "traded_weeks": one["traded_weeks"], "return": one["cash"] / 1000 - 1,
         "return_2x": two["cash"] / 1000 - 1, "mdd": max_drawdown(one["curve"], 1000.0), "sharpe": sharpe(weeks),
         "sharpe_2x": sharpe(two["weeks"]), "week_low": low, "week_high": high, "beta": beta, "hit_rate": one["hit_rate"],
         "years": {y: math.exp(v["log"]) - 1 for y, v in years.items()}, "year_wins": wins, "full_years": len(full),
         "without_best": math.exp(without_best) - 1}
    r["gates"] = {"G1 net>0 at 1x and 2x": bool(r["return"] > 0 and r["return_2x"] > 0), "G2 weekly CI low>0": bool(low > 0),
                  "G3 weeks>=150": bool(one["traded_weeks"] >= 150),
                  "G4 majority years and without best>0": bool(wins * 2 > len(full) and without_best > 0),
                  "G5 mdd<=30%": bool(r["mdd"] >= -0.30), "G6 |beta|<=0.3": bool(abs(beta) <= 0.3)}
    return r


def main() -> None:
    os.nice(15)
    coins = [Coin(s, *h4_data.load(s)) for s in UNIVERSE]
    results, passed = {}, []
    for key in RULES:
        r = score(key, coins)
        results[key] = r
        if all(r["gates"].values()):
            passed.append(key)
        print(f"{key} {r['rule']}: {r['traded_weeks']} weeks traded of {r['weeks']}, weekly hit rate {r['hit_rate']:.1%}")
        if r.get("ruined"):
            print(f"  RUINED: equity reached zero on {r['ruined']} (no stop, 25% per leg); all gates fail\n")
            continue
        print(f"  return {r['return']:+.1%}, at 2x cost {r['return_2x']:+.1%}, mdd {r['mdd']:.1%}, sharpe {r['sharpe']:.2f} (2x {r['sharpe_2x']:.2f}), beta {r['beta']:+.2f}")
        print(f"  mean week 95% [{r['week_low'] * 100:+.3f}% ~ {r['week_high'] * 100:+.3f}%], without best year {r['without_best']:+.1%}")
        print("  years " + ", ".join(f"{y} {v:+.1%}" for y, v in r["years"].items()))
        print("  gates " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["gates"].items()) + "\n")
    chosen = max(passed, key=lambda k: results[k]["sharpe_2x"]) if passed else None
    print(f"passed: {passed}\nverdict: " + (f"CANDIDATE {chosen} {RULES[chosen]} (validation only with the user's approval)" if chosen
                                             else "NO_VERIFIED_XS_EDGE (validation period left unused)"))
    out = {"when": time.strftime("%FT%TZ", time.gmtime()), "results": results, "passed": passed, "chosen": chosen}
    (h4_data.DATA_DIR / "xs_dev_result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
