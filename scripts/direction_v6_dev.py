"""Runs the v6 tests: does the v3 direction model hold on 50 coins it never saw, selectively and at other horizons?
See docs/research/direction_preregistration_v6.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v6_dev.py
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

from research import h4_data
from research.direction import (EXTERNAL_FEATURES, FEATURES, SIDE_COST, add_external, frame, log_loss, month_interval,
                                walk_forward)

TRAIN_START, SCORE_START, END = h4_data.DEV_START_MS, h4_data.utc_ms("2022-01-01"), h4_data.VAL_END_MS
ORIGINAL = list(h4_data.SYMBOLS)
HORIZONS = {"4h": 1, "8h": 2, "12h": 3}


def dataset(horizon: int, symbols: list[str]) -> pd.DataFrame:
    btc = frame("BTCUSDT", *h4_data.load_validation("BTCUSDT"), None, horizon).set_index("close_ms")[["r1", "r6"]]
    load = lambda s: h4_data.load_validation(s) if s in ORIGINAL else h4_data.load_v6(s)
    return pd.concat([frame(s, *load(s), None if s == "BTCUSDT" else btc, horizon) for s in symbols], ignore_index=True)


def mean_ci(values: pd.Series, month: pd.Series) -> tuple[float, list[float]]:
    g = pd.DataFrame({"v": values, "m": month}).groupby("m")["v"].agg(["sum", "size"])
    return float(values.mean()), list(month_interval(g["sum"], g["size"]))


def metrics(pred: pd.DataFrame) -> dict:
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
    right = pd.Series((p > 0.5) == (y > 0), index=pred.index)
    gain = pd.Series(log_loss(pred["base"].to_numpy(), y) - log_loss(p, y), index=pred.index)
    r = {"predictions": len(pred), "accuracy": float(right.mean()), "always_up": float(y.mean()), "always_down": float(1 - y.mean())}
    r["gain"], r["gain_ci"] = mean_ci(gain, pred["month"])
    chosen = (pred["p"] - 0.5).abs() >= pred["thr"]
    sel = pred[chosen]
    direction = np.where(sel["p"] > 0.5, 1.0, -1.0)
    gross = direction * np.expm1(sel["fwd"]) - direction * sel["fund_hold"]
    r["selected"], r["selected_share"] = int(chosen.sum()), float(chosen.mean())
    r["selected_hit"], r["selected_hit_ci"] = mean_ci(right[chosen].astype(float), sel["month"])
    r["net_1x"], r["net_1x_ci"] = mean_ci(gross - 2 * SIDE_COST, sel["month"])
    r["net_2x"], r["net_2x_ci"] = mean_ci(gross - 4 * SIDE_COST, sel["month"])
    years = pd.to_datetime(pred["close_ms"], unit="ms").dt.year
    r["accuracy_by_year"] = {int(k): float(v) for k, v in right.groupby(years).mean().items()}
    r["selected_hit_by_year"] = {int(k): float(v) for k, v in right[chosen].groupby(years[chosen]).mean().items()}
    r["checks"] = {"T1a gain CI low>0": bool(r["gain_ci"][0] > 0),
                   "T1b accuracy>majority": bool(r["accuracy"] > max(r["always_up"], r["always_down"])),
                   "T2a selected hit CI low>50%": bool(r["selected_hit_ci"][0] > 0.5),
                   "T2b selected hit>accuracy": bool(r["selected_hit"] > r["accuracy"]),
                   "T2c net>0 at 1x and 2x": bool(r["net_1x"] > 0 and r["net_2x"] > 0)}
    return r


def versus(better: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Log-loss gain of `better` over `other` on the same rows."""
    assert (better["close_ms"].to_numpy() == other["close_ms"].to_numpy()).all() and (better["symbol"].to_numpy() == other["symbol"].to_numpy()).all()
    y = better["y"].to_numpy()
    gain = pd.Series(log_loss(other["p"].to_numpy(), y) - log_loss(better["p"].to_numpy(), y), index=better.index)
    mean, ci = mean_ci(gain, better["month"])
    acc = lambda d: float(((d["p"] > 0.5) == (d["y"] > 0)).mean())
    return {"rows": len(better), "gain": mean, "gain_ci": ci, "accuracy": acc(better), "accuracy_other": acc(other), "pass": bool(ci[0] > 0)}


def show(name: str, r: dict) -> None:
    print(f"{name}: {r['predictions']} predictions, accuracy {r['accuracy']:.2%} (always-up {r['always_up']:.2%}, always-down "
          f"{r['always_down']:.2%}), gain {r['gain'] * 1e4:+.2f}e-4 [{r['gain_ci'][0] * 1e4:+.2f} ~ {r['gain_ci'][1] * 1e4:+.2f}]")
    print(f"  top 10%: {r['selected']} ({r['selected_share']:.1%}), hit {r['selected_hit']:.2%} [{r['selected_hit_ci'][0]:.2%} ~ "
          f"{r['selected_hit_ci'][1]:.2%}], net per bet {r['net_1x']:+.3%} [{r['net_1x_ci'][0]:+.3%} ~ {r['net_1x_ci'][1]:+.3%}], "
          f"at 2x cost {r['net_2x']:+.3%}")
    print("  accuracy by year " + ", ".join(f"{y} {v:.1%}" for y, v in r["accuracy_by_year"].items())
          + " | top 10% " + ", ".join(f"{y} {v:.1%}" for y, v in r["selected_hit_by_year"].items()))
    print("  checks " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()), flush=True)


def main() -> None:
    os.nice(15)
    new = h4_data.choose_v6_symbols()
    everything = ORIGINAL + new
    out: dict = {"new_symbols": new, "horizons": {}}
    passed = []
    for name, h in HORIZONS.items():
        data = dataset(h, everything)
        transfer = walk_forward(data, h, TRAIN_START, SCORE_START, END, False, train_symbols=set(ORIGINAL))
        on_new, on_original = transfer[transfer["symbol"].isin(new)], transfer[transfer["symbol"].isin(ORIGINAL)]
        r = out["horizons"][name] = {"new_coins": metrics(on_new), "original_coins_reference": metrics(on_original)}
        show(f"[{name}] NEW 50 coins (model trained on the original 10)", r["new_coins"])
        show(f"[{name}] original 10 coins (already seen, reference)", r["original_coins_reference"])
        if all(r["new_coins"]["checks"].values()):
            passed.append(name)
        if h == 1:
            wide = walk_forward(data, h, TRAIN_START, SCORE_START, END, False)
            out["T4_train_on_60_vs_10_on_original"] = t4 = versus(wide[wide["symbol"].isin(ORIGINAL)], on_original)
            out["train_on_60_new_coins_reference"] = metrics(wide[wide["symbol"].isin(new)])
            by_coin = ((on_new["p"] > 0.5) == (on_new["y"] > 0)).groupby(on_new["symbol"]).mean()
            out["new_coin_accuracy"] = {s: float(v) for s, v in by_coin.sort_values().items()}
            print(f"T4 train on 60 vs 10, original coins: gain {t4['gain'] * 1e4:+.2f}e-4 [{t4['gain_ci'][0] * 1e4:+.2f} ~ "
                  f"{t4['gain_ci'][1] * 1e4:+.2f}], accuracy {t4['accuracy']:.2%} vs {t4['accuracy_other']:.2%} -> "
                  f"{'PASS' if t4['pass'] else 'FAIL'}")
            show("  (reference) NEW 50 coins, model trained on all 60", out["train_on_60_new_coins_reference"])
            print(f"  new coins above 50%: {int((by_coin > 0.5).sum())} of {len(by_coin)}; lowest {by_coin.min():.1%}, highest {by_coin.max():.1%}")
            ext = add_external(dataset(1, ORIGINAL), h4_data.load_external())
            plain = walk_forward(ext, 1, TRAIN_START, SCORE_START, END, False, require=EXTERNAL_FEATURES)
            richer = walk_forward(ext, 1, TRAIN_START, SCORE_START, END, False, features=FEATURES + EXTERNAL_FEATURES)
            out["T5_external_vs_v3"] = t5 = versus(richer, plain)
            print(f"T5 external features vs v3, original coins: gain {t5['gain'] * 1e4:+.2f}e-4 [{t5['gain_ci'][0] * 1e4:+.2f} ~ "
                  f"{t5['gain_ci'][1] * 1e4:+.2f}], accuracy {t5['accuracy']:.2%} vs {t5['accuracy_other']:.2%} -> "
                  f"{'PASS' if t5['pass'] else 'FAIL'}\n", flush=True)
    chosen = max(passed, key=lambda k: out["horizons"][k]["new_coins"]["net_2x"]) if passed else None
    out.update(passed=passed, chosen=chosen, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed: {passed}\nverdict: " + (f"CANDIDATE {chosen} (final check only with the user's approval)" if chosen
                                             else "NO_VERIFIED_DIRECTION_EDGE (final-check period left unused)"))
    (h4_data.DATA_DIR / "direction_v6_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
