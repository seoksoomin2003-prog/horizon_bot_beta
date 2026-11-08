"""Scores the v3 direction model on the development period against the fixed gates. Never touches validation data.

  .venv/bin/python scripts/direction_dev.py
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
import pandas as pd

from research import h4_data
from research.direction import HORIZONS, SIDE_COST, equity_curve, frame, log_loss, month_interval, trades, walk_forward
from research.h4_sim import max_drawdown, year_log_returns

TRAIN_START, SCORE_START, END = h4_data.DEV_START_MS, h4_data.utc_ms("2022-01-01"), h4_data.DEV_END_MS
YEARS = (2022, 2023, 2024, 2025)


def dataset(horizon: int) -> pd.DataFrame:
    btc_bars, btc_funding = h4_data.load("BTCUSDT")
    btc = frame("BTCUSDT", btc_bars, btc_funding, None, horizon).set_index("close_ms")[["r1", "r6"]]
    return pd.concat([frame(s, *h4_data.load(s), None if s == "BTCUSDT" else btc, horizon) for s in h4_data.SYMBOLS],
                     ignore_index=True)


def score(key: str) -> dict:
    h = HORIZONS[key]
    pred = walk_forward(dataset(h), h, TRAIN_START, SCORE_START, END, daily_only=(key == "D24"))
    pred["gain"] = log_loss(pred["base"].to_numpy(), pred["y"].to_numpy()) - log_loss(pred["p"].to_numpy(), pred["y"].to_numpy())
    by_month = pred.groupby("month").agg(gain=("gain", "sum"), n=("gain", "size"))
    info_low, info_high = month_interval(by_month["gain"], by_month["n"])
    r = {"predictions": len(pred), "accuracy": float(((pred["p"] > 0.5) == (pred["y"] > 0)).mean()),
         "always_up": float(pred["y"].mean()), "logloss_gain": float(pred["gain"].mean()), "logloss_gain_ci": [info_low, info_high]}
    pred["conf"] = (pred["p"] - 0.5).abs()
    deciles = pd.qcut(pred["conf"], 10, labels=False, duplicates="drop")
    r["accuracy_by_confidence_decile"] = [float(v) for v in ((pred["p"] > 0.5) == (pred["y"] > 0)).groupby(deciles).mean()]
    for mult in (1, 2):
        t = trades(pred, h, SIDE_COST * mult)
        curve = equity_curve(t)
        tag = "" if mult == 1 else "_2x"
        r["return" + tag] = (curve[-1][1] / 1000 - 1) if curve else 0.0
        if mult == 1:
            hits = t.groupby("month").agg(hit=("hit", "sum"), n=("hit", "size"))
            r["trades"], r["hit_rate"] = len(t), float(t["hit"].mean()) if len(t) else None
            r["hit_ci"] = list(month_interval(hits["hit"], hits["n"])) if len(t) else [0.0, 0.0]
            r["mdd"] = max_drawdown(curve, 1000.0) if curve else 0.0
            years = year_log_returns(curve, 1000.0) if curve else {}
            logs = {y: years.get(y, {"log": 0.0})["log"] for y in YEARS}
            r["years"] = {y: math.exp(v) - 1 for y, v in logs.items()}
            r["year_wins"] = sum(v > 0 for v in logs.values())
            r["without_best"] = math.exp(sum(logs.values()) - max(logs.values())) - 1
            r["by_coin"] = {s: float(v) for s, v in (0.10 * t.groupby("symbol")["ret"].sum()).items()}
    r["gates"] = {"G1 logloss CI low>0": bool(info_low > 0), "G2 hit CI low>50%": bool(r["hit_ci"][0] > 0.5),
                  "G3 net>0 at 1x and 2x": bool(r["return"] > 0 and r["return_2x"] > 0),
                  "G4 3+ of 4 years and without best>0": bool(r["year_wins"] >= 3 and r["without_best"] > 0),
                  "G5 mdd<=30%": bool(r["mdd"] >= -0.30), "G6 trades>=300": bool(r["trades"] >= 300)}
    return r


def main() -> None:
    os.nice(15)
    results, passed = {}, []
    for key in HORIZONS:
        r = results[key] = score(key)
        if all(r["gates"].values()):
            passed.append(key)
        print(f"{key}: {r['predictions']} predictions, accuracy {r['accuracy']:.2%} (always-up {r['always_up']:.2%}), "
              f"log-loss gain {r['logloss_gain']:+.5f} [{r['logloss_gain_ci'][0]:+.5f} ~ {r['logloss_gain_ci'][1]:+.5f}]")
        print("  accuracy by confidence decile (low->high): " + " ".join(f"{v:.1%}" for v in r["accuracy_by_confidence_decile"]))
        hit = f"{r['hit_rate']:.2%} [{r['hit_ci'][0]:.2%} ~ {r['hit_ci'][1]:.2%}]" if r["hit_rate"] is not None else "-"
        print(f"  bets {r['trades']}, hit {hit}, return {r['return']:+.1%}, at 2x cost {r['return_2x']:+.1%}, mdd {r['mdd']:.1%}")
        print("  years " + ", ".join(f"{y} {v:+.1%}" for y, v in r["years"].items()) + f", without best {r['without_best']:+.1%}")
        print("  by coin " + ", ".join(f"{s[:-4]} {v:+.1%}" for s, v in r["by_coin"].items()))
        print("  gates " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["gates"].items()) + "\n")
    chosen = max(passed, key=lambda k: results[k]["return_2x"]) if passed else None
    print(f"passed: {passed}\nverdict: " + (f"CANDIDATE {chosen} (validation only with the user's approval)" if chosen
                                             else "NO_VERIFIED_DIRECTION_EDGE (validation period left unused)"))
    out = {"when": time.strftime("%FT%TZ", time.gmtime()), "results": results, "passed": passed, "chosen": chosen}
    (h4_data.DATA_DIR / "direction_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
