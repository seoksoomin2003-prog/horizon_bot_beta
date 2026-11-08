"""Scores the v4 direction model once on the validation period. See docs/research/direction_preregistration_v4.md.

  .venv/bin/python scripts/direction_validation.py

Refuses to run a second time: the validation period is spent by the first run.
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
from research.direction import HORIZONS, SIDE_COST, equity_curve, frame, log_loss, month_interval, trades, walk_forward
from research.h4_sim import max_drawdown

MIN_GAIN = 0.005
OUT = h4_data.DATA_DIR / "direction_validation_result.json"


def dataset(horizon: int) -> pd.DataFrame:
    btc = frame("BTCUSDT", *h4_data.load_validation("BTCUSDT"), None, horizon).set_index("close_ms")[["r1", "r6"]]
    return pd.concat([frame(s, *h4_data.load_validation(s), None if s == "BTCUSDT" else btc, horizon) for s in h4_data.SYMBOLS],
                     ignore_index=True)


def score(key: str) -> dict:
    h = HORIZONS[key]
    pred = walk_forward(dataset(h), h, h4_data.DEV_START_MS, h4_data.DEV_END_MS, h4_data.VAL_END_MS, daily_only=(key == "D24"))
    pred["gain"] = log_loss(pred["base"].to_numpy(), pred["y"].to_numpy()) - log_loss(pred["p"].to_numpy(), pred["y"].to_numpy())
    by_month = pred.groupby("month").agg(gain=("gain", "sum"), n=("gain", "size"))
    r = {"predictions": len(pred), "refits": len(by_month), "accuracy": float(((pred["p"] > 0.5) == (pred["y"] > 0)).mean()),
         "always_up": float(pred["y"].mean()), "logloss_gain": float(pred["gain"].mean()),
         "logloss_gain_ci": list(month_interval(by_month["gain"], by_month["n"]))}
    for mult in (1, 2):
        t = trades(pred, h, SIDE_COST * mult, min_gain=MIN_GAIN)
        curve = equity_curve(t)
        r["return" if mult == 1 else "return_2x"] = (curve[-1][1] / 1000 - 1) if curve else 0.0
        if mult == 1:
            hits = t.groupby("month").agg(hit=("hit", "sum"), n=("hit", "size"))
            r["trades"], r["hit_rate"] = len(t), float(t["hit"].mean()) if len(t) else 0.0
            r["hit_ci"] = list(month_interval(hits["hit"], hits["n"])) if len(t) else [0.0, 0.0]
            r["mdd"] = max_drawdown(curve, 1000.0) if curve else 0.0
            t["ym"] = pd.to_datetime(t["exit_ms"], unit="ms").dt.strftime("%Y-%m")
            r["by_month"] = {m: float(v) for m, v in (0.10 * t.groupby("ym")["ret"].sum()).items()}
            r["by_coin"] = {s: float(v) for s, v in (0.10 * t.groupby("symbol")["ret"].sum()).items()}
    r["checks"] = {"1 net>0 at 1x and 2x": bool(r["return"] > 0 and r["return_2x"] > 0), "2 mdd<=30%": bool(r["mdd"] >= -0.30),
                   "3 trades>=50": bool(r["trades"] >= 50), "4 hit>50%": bool(r["hit_rate"] > 0.5)}
    r["verdict"] = "PASS_FOR_PAPER" if all(r["checks"].values()) else "FAIL"
    return r


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} exists: the validation period has already been used once")
    os.nice(15)
    results = {key: score(key) for key in HORIZONS}
    for key, r in results.items():
        print(f"{key}: {r['predictions']} predictions over {r['refits']} refits, accuracy {r['accuracy']:.2%} "
              f"(always-up {r['always_up']:.2%}), log-loss gain {r['logloss_gain']:+.5f} "
              f"[{r['logloss_gain_ci'][0]:+.5f} ~ {r['logloss_gain_ci'][1]:+.5f}]")
        print(f"  bets {r['trades']}, hit {r['hit_rate']:.2%} [{r['hit_ci'][0]:.2%} ~ {r['hit_ci'][1]:.2%}], "
              f"return {r['return']:+.1%}, at 2x cost {r['return_2x']:+.1%}, mdd {r['mdd']:.1%}")
        print("  months " + ", ".join(f"{m} {v:+.1%}" for m, v in r["by_month"].items()))
        print("  by coin " + ", ".join(f"{s[:-4]} {v:+.1%}" for s, v in r["by_coin"].items()))
        print("  checks " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items())
              + f" -> {r['verdict']}\n")
    passed = [k for k, r in results.items() if r["verdict"] == "PASS_FOR_PAPER"]
    chosen = max(passed, key=lambda k: results[k]["return_2x"]) if passed else None
    print(f"passed: {passed}, paper candidate: {chosen}")
    OUT.write_text(json.dumps({"when": time.strftime("%FT%TZ", time.gmtime()), "results": results, "passed": passed,
                               "chosen": chosen}, indent=1, default=str))


if __name__ == "__main__":
    main()
