"""Scores the registered 4h rules on the development period against the fixed gates. Never touches the validation period.

  .venv/bin/python scripts/h4_dev.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from research import h4_data
from research.h4_rules import RULES, signals
from research.h4_sim import Market, SIDE_COST, benchmark, block_interval, max_drawdown, run, sharpe, weekly_returns, year_log_returns

START, END = h4_data.DEV_START_MS, h4_data.DEV_END_MS


def score(key: str) -> dict:
    markets = []
    for symbol in h4_data.SYMBOLS:
        bars, funding = h4_data.load(symbol)
        markets.append(Market(symbol, bars, funding, signals(key, bars)))
    rule = RULES[key]
    one, two = run(rule, markets, START, END, SIDE_COST), run(rule, markets, START, END, SIDE_COST * 2)
    weeks, weeks2 = weekly_returns(one["curve"], 1000.0), weekly_returns(two["curve"], 1000.0)
    low, high = block_interval(weeks)
    years = year_log_returns(one["curve"], 1000.0)
    full = [y for y, row in years.items() if row["months"] >= 6]
    wins = sum(years[y]["log"] > 0 for y in full)
    total_log = sum(r["log"] for r in years.values())
    without_best = total_log - max(r["log"] for r in years.values())
    ps = one["positions"]
    side = {name: [p["pnl"] for p in ps if p["side"] == s] for name, s in (("long", 1), ("short", -1))}
    return {"rule": rule.name, "positions": len(ps), "skipped": one["skipped"], "cash": one["cash"], "cash_2x": two["cash"],
            "return": one["cash"] / 1000 - 1, "return_2x": two["cash"] / 1000 - 1, "mdd": max_drawdown(one["curve"], 1000.0),
            "sharpe": sharpe(weeks), "sharpe_2x": sharpe(weeks2), "week_low": low, "week_high": high,
            "years": {y: float(np.exp(r["log"]) - 1) for y, r in years.items()}, "year_wins": wins, "full_years": len(full),
            "without_best": float(np.exp(without_best) - 1), "win_rate": float(np.mean([p["pnl"] > 0 for p in ps])) if ps else None,
            "long_pnl": sum(side["long"]), "short_pnl": sum(side["short"]), "long_n": len(side["long"]), "short_n": len(side["short"]),
            "per_year": len(ps) / ((END - START) / 86_400_000 / 365.25)}


def gates(r: dict, bench_sharpe: float) -> dict:
    checks = {"G1 net>0 at 1x and 2x": r["return"] > 0 and r["return_2x"] > 0, "G2 weekly CI low>0": r["week_low"] > 0,
              "G3 positions>=100": r["positions"] >= 100,
              "G4 majority years and without best>0": r["year_wins"] * 2 > r["full_years"] and r["without_best"] > 0,
              "G5 mdd<=30%": r["mdd"] >= -0.30, "G6 sharpe>=benchmark": r["sharpe"] >= bench_sharpe}
    return {k: bool(v) for k, v in checks.items()}


def main() -> None:
    os.nice(15)
    markets = []
    for symbol in h4_data.SYMBOLS:
        bars, funding = h4_data.load(symbol)
        markets.append(Market(symbol, bars, funding, signals("A", bars)))
    bench = benchmark(markets, START, END)
    bench_weeks = weekly_returns(bench, 1000.0)
    b = {"return": bench[-1][1] / 1000 - 1, "mdd": max_drawdown(bench, 1000.0), "sharpe": sharpe(bench_weeks)}
    print(f"benchmark: return {b['return']:+.1%}, mdd {b['mdd']:.1%}, weekly sharpe {b['sharpe']:.2f}\n")
    results, passed = {}, []
    for key in RULES:
        r = score(key)
        g = gates(r, b["sharpe"])
        r["gates"] = g
        results[key] = r
        if all(g.values()):
            passed.append(key)
        print(f"{r['rule']}: {r['positions']} positions ({r['per_year']:.0f}/yr, skipped {r['skipped']}), win {r['win_rate']:.1%}")
        print(f"  return {r['return']:+.1%}, at 2x cost {r['return_2x']:+.1%}, mdd {r['mdd']:.1%}, sharpe {r['sharpe']:.2f} (2x {r['sharpe_2x']:.2f})")
        print(f"  mean week 95% [{r['week_low'] * 100:+.3f}% ~ {r['week_high'] * 100:+.3f}%], without best year {r['without_best']:+.1%}")
        print(f"  long {r['long_n']} {r['long_pnl']:+.1f}, short {r['short_n']} {r['short_pnl']:+.1f}")
        print("  years " + ", ".join(f"{y} {v:+.1%}" for y, v in r["years"].items()))
        print("  gates " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in g.items()) + "\n")
    chosen = max(passed, key=lambda k: results[k]["sharpe_2x"]) if passed else None
    verdict = f"SELECTED {RULES[chosen].name}" if chosen else "NO_VERIFIED_4H_EDGE (validation period left unused)"
    print(f"passed: {passed}\nverdict: {verdict}")
    out = {"when": time.strftime("%FT%TZ", time.gmtime()), "benchmark": b, "results": results, "passed": passed, "chosen": chosen}
    (h4_data.DATA_DIR / "dev_result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
