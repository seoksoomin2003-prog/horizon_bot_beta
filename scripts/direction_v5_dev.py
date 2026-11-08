"""Scores the v5 direction model (v3 + order flow and basis) on the development period. See
docs/research/direction_preregistration_v5.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v5_dev.py
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
from research.direction import (BAR_MS, FEATURES, FLOW_FEATURES, HORIZONS, SIDE_COST, equity_curve, fit, frame, log_loss,
                                month_interval, trades, walk_forward)
from research.h4_sim import max_drawdown, year_log_returns

TRAIN_START, SCORE_START, END = h4_data.DEV_START_MS, h4_data.utc_ms("2022-01-01"), h4_data.VAL_END_MS
RECENT_START = h4_data.DEV_END_MS          # 2025-11-06, where v3 failed
YEARS = (2022, 2023, 2024, 2025, 2026)
MIN_GAIN = 0.005
V5 = FEATURES + FLOW_FEATURES


def dataset(horizon: int) -> pd.DataFrame:
    btc = frame("BTCUSDT", *h4_data.load_validation("BTCUSDT"), None, horizon, h4_data.load_flow("BTCUSDT"))
    btc = btc.set_index("close_ms")[["r1", "r6", "taker_1"]]
    return pd.concat([frame(s, *h4_data.load_validation(s), None if s == "BTCUSDT" else btc, horizon, h4_data.load_flow(s))
                      for s in h4_data.SYMBOLS], ignore_index=True)


def trading(pred: pd.DataFrame, h: int) -> dict:
    r = {}
    for mult in (1, 2):
        t = trades(pred, h, SIDE_COST * mult, min_gain=MIN_GAIN)
        curve = equity_curve(t)
        r["return" if mult == 1 else "return_2x"] = (curve[-1][1] / 1000 - 1) if curve else 0.0
        if mult == 1:
            hits = t.groupby("month").agg(hit=("hit", "sum"), n=("hit", "size"))
            r["trades"], r["hit_rate"] = len(t), float(t["hit"].mean()) if len(t) else 0.0
            r["hit_ci"] = list(month_interval(hits["hit"], hits["n"])) if len(t) else [0.0, 0.0]
            r["mdd"] = max_drawdown(curve, 1000.0) if curve else 0.0
            years = year_log_returns(curve, 1000.0) if curve else {}
            logs = {y: years.get(y, {"log": 0.0})["log"] for y in YEARS}
            r["years"] = {y: math.exp(v) - 1 for y, v in logs.items()}
            r["year_wins"] = sum(v > 0 for v in logs.values())
            r["without_best"] = math.exp(sum(logs.values()) - max(logs.values())) - 1
            r["by_coin"] = {s: float(v) for s, v in (0.10 * t.groupby("symbol")["ret"].sum()).items()}
    return r


def last_coefficients(data: pd.DataFrame, h: int) -> dict:
    """The final monthly model's weights in standardized units (reported only)."""
    data = data.dropna(subset=V5 + ["fwd", "sd42"])
    data = data[(data["close_ms"] >= TRAIN_START) & (data["close_ms"] + h * BAR_MS <= END)]
    x = data[V5].to_numpy(float)
    last_month = int(pd.Timestamp(END, unit="ms").replace(day=1).value // 1_000_000)
    train = (data["close_ms"] + h * BAR_MS).to_numpy() <= last_month
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    w = fit(np.clip((x[train] - mean) / std, -5, 5), (data["fwd"] > 0).to_numpy(float)[train])
    return {name: float(v) for name, v in zip(V5, w[1:])}


def score(key: str) -> dict:
    h = HORIZONS[key]
    data = dataset(h)
    daily = key == "D24"
    p5 = walk_forward(data, h, TRAIN_START, SCORE_START, END, daily, features=V5)
    p3 = walk_forward(data, h, TRAIN_START, SCORE_START, END, daily, features=FEATURES, require=FLOW_FEATURES)
    assert (p5["close_ms"].to_numpy() == p3["close_ms"].to_numpy()).all() and (p5["symbol"] == p3["symbol"]).all()
    y = p5["y"].to_numpy()
    ll5 = log_loss(p5["p"].to_numpy(), y)
    p5["vs_v3"] = log_loss(p3["p"].to_numpy(), y) - ll5
    p5["vs_base"] = log_loss(p5["base"].to_numpy(), y) - ll5
    m = p5.groupby("month").agg(vs_v3=("vs_v3", "sum"), vs_base=("vs_base", "sum"), n=("vs_v3", "size"))
    right5, right3 = (p5["p"] > 0.5) == (y > 0), (p3["p"] > 0.5) == (y > 0)
    recent = p5["close_ms"] >= RECENT_START
    r = {"predictions": len(p5), "accuracy": float(right5.mean()), "accuracy_v3_same_rows": float(right3.mean()),
         "always_up": float(y.mean()), "always_down": float(1 - y.mean()),
         "gain_vs_v3": float(p5["vs_v3"].mean()), "gain_vs_v3_ci": list(month_interval(m["vs_v3"], m["n"])),
         "gain_vs_base": float(p5["vs_base"].mean()), "gain_vs_base_ci": list(month_interval(m["vs_base"], m["n"])),
         "recent_gain_vs_base": float(p5.loc[recent, "vs_base"].mean()),
         "recent_accuracy": float(right5[recent].mean()), "recent_always_up": float(y[recent.to_numpy()].mean())}
    conf = (p5["p"] - 0.5).abs()
    r["accuracy_by_confidence_decile"] = [float(v) for v in right5.groupby(pd.qcut(conf, 10, labels=False, duplicates="drop")).mean()]
    r.update(trading(p5, h))
    r["v3_same_rows_trading"] = {k: v for k, v in trading(p3, h).items() if k in ("trades", "hit_rate", "return", "return_2x", "mdd")}
    r["coefficients"] = last_coefficients(data, h)
    r["gates"] = {"G0 vs v3 CI low>0": bool(r["gain_vs_v3_ci"][0] > 0), "G1 vs base CI low>0": bool(r["gain_vs_base_ci"][0] > 0),
                  "G2 recent gain>0": bool(r["recent_gain_vs_base"] > 0), "G3 hit CI low>50%": bool(r["hit_ci"][0] > 0.5),
                  "G4 net>0 at 1x and 2x": bool(r["return"] > 0 and r["return_2x"] > 0),
                  "G5 4+ of 5 years and without best>0": bool(r["year_wins"] >= 4 and r["without_best"] > 0),
                  "G6 mdd<=30%": bool(r["mdd"] >= -0.30), "G7 trades>=300": bool(r["trades"] >= 300)}
    return r


def main() -> None:
    os.nice(15)
    results, passed = {}, []
    for key in HORIZONS:
        r = results[key] = score(key)
        if all(r["gates"].values()):
            passed.append(key)
        print(f"{key}: {r['predictions']} predictions, accuracy {r['accuracy']:.2%} (v3 same rows {r['accuracy_v3_same_rows']:.2%}, "
              f"always-up {r['always_up']:.2%})")
        print(f"  log-loss gain vs v3 {r['gain_vs_v3']:+.5f} [{r['gain_vs_v3_ci'][0]:+.5f} ~ {r['gain_vs_v3_ci'][1]:+.5f}], "
              f"vs base {r['gain_vs_base']:+.5f} [{r['gain_vs_base_ci'][0]:+.5f} ~ {r['gain_vs_base_ci'][1]:+.5f}]")
        print(f"  recent (2025-11-06~): gain vs base {r['recent_gain_vs_base']:+.5f}, accuracy {r['recent_accuracy']:.2%} "
              f"(always-up {r['recent_always_up']:.2%})")
        print("  accuracy by confidence decile: " + " ".join(f"{v:.1%}" for v in r["accuracy_by_confidence_decile"]))
        print(f"  bets {r['trades']}, hit {r['hit_rate']:.2%} [{r['hit_ci'][0]:.2%} ~ {r['hit_ci'][1]:.2%}], return {r['return']:+.1%}, "
              f"at 2x cost {r['return_2x']:+.1%}, mdd {r['mdd']:.1%}")
        v3 = r["v3_same_rows_trading"]
        print(f"  v3 same rows: bets {v3['trades']}, hit {v3['hit_rate']:.2%}, return {v3['return']:+.1%}, 2x {v3['return_2x']:+.1%}, "
              f"mdd {v3['mdd']:.1%}")
        print("  years " + ", ".join(f"{y} {v:+.1%}" for y, v in r["years"].items()) + f", without best {r['without_best']:+.1%}")
        print("  by coin " + ", ".join(f"{s[:-4]} {v:+.1%}" for s, v in r["by_coin"].items()))
        print("  new coefficients " + ", ".join(f"{k} {r['coefficients'][k]:+.3f}" for k in FLOW_FEATURES))
        print("  gates " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["gates"].items()) + "\n")
    chosen = max(passed, key=lambda k: results[k]["return_2x"]) if passed else None
    print(f"passed: {passed}\nverdict: " + (f"CANDIDATE {chosen} (final check only with the user's approval)" if chosen
                                             else "NO_VERIFIED_DIRECTION_EDGE (final-check period left unused)"))
    out = {"when": time.strftime("%FT%TZ", time.gmtime()), "results": results, "passed": passed, "chosen": chosen}
    (h4_data.DATA_DIR / "direction_v5_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
