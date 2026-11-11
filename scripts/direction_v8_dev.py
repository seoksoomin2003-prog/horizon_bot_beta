"""Runs the v8 tests: the 8h limit-order candidate judged per decision time, with stricter fills and four variants.
See docs/research/direction_preregistration_v8.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v8_dev.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from research import h4_data
from research.direction import BASKET_CAP, basket_returns, fill_windows, maker_trades, walk_forward
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START, dataset, mean_ci

HORIZON = 2                 # 8 hours
GAP = 0.0005                # the price must pass the order by 0.05%
VARIANTS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}     # (long only, broad times only)
YEARS = range(2022, 2027)


def evaluate(selected: pd.DataFrame, windows: pd.DataFrame, new: list[str], long_only: bool, broad: bool) -> dict:
    if long_only:
        selected = selected[selected["p"] > 0.5]
    if broad:
        selected = selected[selected.groupby("close_ms")["p"].transform("size") >= BASKET_CAP]
    empty = {"selected": len(selected), "trades": 0, "checks": {"V1 enough times and trades": False}}
    if selected.empty:
        return empty
    placed = maker_trades(selected, windows, HORIZON, gap=GAP)
    t = placed[placed["entry_filled"]]
    if t.empty:
        return empty
    r1, r2 = basket_returns(t, "net_1x"), basket_returns(t, "net_2x")
    month = t.groupby("close_ms")["month"].first()
    year = pd.to_datetime(r1.index.to_series(), unit="ms").dt.year
    monthly = r1.groupby(month).sum().sort_values()
    group = t["symbol"].isin(new).map({True: "new", False: "original"})
    out = {"selected": len(selected), "trades": len(t), "times": len(r1), "fill_share": len(t) / max(len(placed), 1),
           "hit": float((t["move"] > 0).mean()), "per_trade_net_1x": float(t["net_1x"].mean()), "mean_2x": float(r2.mean()),
           "by_year": {int(k): float(v) for k, v in r1.groupby(year).mean().items()},
           "recent": float(r1[r1.index >= h4_data.DEV_END_MS].mean()), "recent_times": int((r1.index >= h4_data.DEV_END_MS).sum()),
           "per_trade_by_group": {k: float(v) for k, v in t["net_1x"].groupby(group).mean().items()},
           "months": len(monthly), "positive_months": int((monthly > 0).sum()), "without_best_3_months": float(monthly.iloc[:-3].sum())}
    out["mean_1x"], out["mean_1x_ci"] = mean_ci(r1, month)
    equity = 0.5 * r1.sort_index().cumsum()          # two overlapping baskets, half the capital each
    out["total_simple_return"], out["max_drawdown"] = float(equity.iloc[-1]), float((equity.cummax() - equity).max())
    out["checks"] = {"V1 enough times and trades": bool(out["times"] >= 300 and out["trades"] >= 3000),
                     "V2 mean>0 with CI low>0 at 1x": bool(out["mean_1x_ci"][0] > 0),
                     "V3 mean>0 at 2x": bool(out["mean_2x"] > 0),
                     "V4 positive in 4 of 5 years": bool(sum(out["by_year"].get(y, 0) > 0 for y in YEARS) >= 4),
                     "V5 positive since 2025-11-06": bool(out["recent"] > 0),
                     "V6 positive in both coin groups": bool(len(out["per_trade_by_group"]) == 2 and all(v > 0 for v in out["per_trade_by_group"].values())),
                     "V7 positive without the best 3 months": bool(out["without_best_3_months"] > 0)}
    return out


def show(name: str, r: dict) -> None:
    if not r["trades"]:
        return print(f"{name}: no trades -> V1 FAIL", flush=True)
    print(f"{name}: {r['trades']} trades at {r['times']} times (fill {r['fill_share']:.1%}), hit {r['hit']:.2%}, per trade {r['per_trade_net_1x']:+.3%}")
    print(f"   per decision time {r['mean_1x']:+.4%} [{r['mean_1x_ci'][0]:+.4%} ~ {r['mean_1x_ci'][1]:+.4%}], at 2x {r['mean_2x']:+.4%}; "
          f"since 2025-11-06 {r['recent']:+.4%} ({r['recent_times']} times); positive months {r['positive_months']}/{r['months']}; "
          f"sum without the best 3 months {r['without_best_3_months']:+.2%}")
    print("   by year " + ", ".join(f"{y} {v:+.4%}" for y, v in r["by_year"].items())
          + " | per trade " + ", ".join(f"{g} {v:+.3%}" for g, v in r["per_trade_by_group"].items())
          + f" | total {r['total_simple_return']:+.1%}, max drawdown {r['max_drawdown']:.1%}")
    print("   " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()), flush=True)


def main() -> None:
    os.nice(15)
    new = h4_data.choose_v6_symbols()
    everything = ORIGINAL + new
    windows = pd.concat({s: fill_windows(h4_data.load_m5(s)) for s in everything}, names=["symbol", "T"])
    pred = walk_forward(dataset(HORIZON, everything), HORIZON, TRAIN_START, SCORE_START, END, False)      # learns from all 60, as the live predictor does
    selected = pred[(pred["p"] - 0.5).abs() >= pred["thr"]]
    out: dict = {"predictions": len(pred), "selected": len(selected), "variants": {}}
    for name, (long_only, broad) in VARIANTS.items():
        out["variants"][name] = r = evaluate(selected, windows, new, long_only, broad)
        show(f"[{name}] {'long only' if long_only else 'long and short'}{', broad times only' if broad else ''}", r)
    passed = [k for k, r in out["variants"].items() if all(r["checks"].values())]
    chosen = max(passed, key=lambda k: out["variants"][k]["mean_2x"]) if passed else None
    out.update(passed=passed, chosen=chosen, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed: {passed}\nverdict: " + (f"CANDIDATE {chosen} (final check only with the user's approval)" if chosen
                                             else "NO_VERIFIED_DIRECTION_EDGE (final-check period left unused)"))
    (h4_data.DATA_DIR / "direction_v8_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
