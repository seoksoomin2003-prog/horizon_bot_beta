"""Runs the v13 test: do fewer replacements or limit orders let the relative model's baskets keep money after costs?
See docs/research/direction_preregistration_v13.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v13_dev.py            the registered run
  .venv/bin/python scripts/direction_v13_dev.py --smoke    trades random numbers instead of the model, to check the plumbing
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
from research.direction import fill_windows
from scripts.direction_v6_dev import ORIGINAL
from scripts.direction_v12_dev import checks, panels, predictions, show

HORIZON = 2                                                              # the 8h model, two baskets
CANDIDATES = {"T30": (0.3, False), "T50": (0.5, False), "M10": (0.1, True), "M50": (0.5, True)}    # (keep band, limit orders)
REFERENCE = {"T10": (0.1, False)}                                        # v12's A8 without re-sizing the coins that stay
NAMES = ("limit", "low", "high", "last")


def windows(symbols: list[str], grid: np.ndarray) -> dict[str, pd.DataFrame]:
    """Per coin and 4h close: the limit price five minutes later and the low, high and last of the next 30 minutes."""
    per_coin = {s: fill_windows(h4_data.load_m5(s)).reindex(grid) for s in symbols}
    return {name: pd.DataFrame({s: w[name] for s, w in per_coin.items()}, index=grid) for name in NAMES}


def fills(bars: pd.DataFrame) -> dict:
    return {"entry_fill": float(bars["filled"].sum() / max(bars["entries"].sum(), 1)),
            "exit_limit": float(bars["exits_limit"].sum() / max(bars["exits"].sum(), 1)), "timing": float(bars["timing"].sum())}


def main() -> None:
    os.nice(15)
    smoke = "--smoke" in sys.argv
    reference = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
    new = h4_data.choose_v9_symbols()
    closes, funding = panels(reference + new, new)
    limits = windows(reference + new, closes.index.to_numpy())
    pred = predictions(HORIZON, reference, new, smoke)
    out: dict = {"smoke": smoke, "candidates": {}, "reference": {}}

    def trade(coins: list[str], keep: float, limit: bool, k: int | None, cost: int) -> dict:
        bars = ls_sim.run_held(pred[pred["symbol"].isin(coins)], closes[coins], funding[coins], HORIZON, k, keep,
                               side_cost=cost * ls_sim.SIDE_COST, maker_cost=cost * ls_sim.MAKER_COST,
                               windows={n: limits[n][coins] for n in NAMES} if limit else None)
        return ls_sim.summary(bars, h4_data.DEV_END_MS) | (fills(bars) if limit else {})

    def line(label: str, s: dict) -> None:
        show(label, s)
        if "entry_fill" in s:
            print(f"      entries filled {s['entry_fill']:.1%}, exits by limit {s['exit_limit']:.1%}, "
                  f"gain from the order prices {s['timing']:+.1%}", flush=True)

    passed = []
    for name, (keep, limit) in (CANDIDATES | REFERENCE).items():
        r = {f"{u}_cost{c}": trade(coins, keep, limit, None, c) for u, coins in (("new", new), ("seen", reference)) for c in (1, 2)}
        r["two_coins_new"], r["two_coins_seen"] = trade(new, keep, limit, 2, 1), trade(reference, keep, limit, 2, 1)
        r["checks"] = checks(r["new_cost1"], r["new_cost2"], r["seen_cost1"])
        judged = name in CANDIDATES
        out["candidates" if judged else "reference"][name] = r
        print(f"[{name}]{'' if judged else ' (reference only)'} keep within {keep:.0%}, {'limit' if limit else 'market'} orders")
        line("46 new coins, cost x1", r["new_cost1"])
        line("46 new coins, cost x2", r["new_cost2"])
        line("59 seen coins, cost x1", r["seen_cost1"])
        line("(reference) 2 coins a side, 46 new", r["two_coins_new"])
        line("(reference) 2 coins a side, 59 seen", r["two_coins_seen"])
        print("   checks " + ", ".join(f"{c.split()[0]} {'PASS' if v else 'FAIL'}" for c, v in r["checks"].items()) + "\n", flush=True)
        if judged and all(r["checks"].values()):
            passed.append(name)
    best = max(passed, key=lambda n: out["candidates"][n]["new_cost2"]["month_sharpe"]) if passed else None
    out.update(passed=passed, chosen=best, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed G1-G6: {passed}\nverdict: " + (f"PASS_FOR_PAPER ({best})" if best else "NO_VERIFIED_PROFIT_EDGE"))
    if smoke:
        print("(smoke run on random numbers: nothing saved, not a result)")
    else:
        (h4_data.DATA_DIR / "direction_v13_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
