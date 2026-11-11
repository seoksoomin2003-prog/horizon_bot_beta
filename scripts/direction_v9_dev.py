"""Runs the v9 tests: does predicting "beats the median coin" hold on 46 coins never seen, and is it steadier?
See docs/research/direction_preregistration_v9.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v9_dev.py
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
from research.direction import FEATURES, MARKET_FEATURES, add_relative, frame, log_loss, walk_forward
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START, mean_ci

HORIZONS = {"4h": 1, "8h": 2}
YEARS = range(2022, 2027)
FLOOR = 0.52


def dataset(horizon: int, reference: list[str], new: list[str]) -> pd.DataFrame:
    load = lambda s: h4_data.load_validation(s) if s in ORIGINAL else h4_data.load_v9(s) if s in new else h4_data.load_v6(s)
    btc = frame("BTCUSDT", *load("BTCUSDT"), None, horizon).set_index("close_ms")[["r1", "r6"]]
    return pd.concat([frame(s, *load(s), None if s == "BTCUSDT" else btc, horizon) for s in reference + new], ignore_index=True)


def score(pred: pd.DataFrame) -> dict:
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
    right = pd.Series((p > 0.5) == (y > 0), index=pred.index)
    year = pd.to_datetime(pred["close_ms"], unit="ms").dt.year
    recent = pred["close_ms"] >= h4_data.DEV_END_MS
    gain = pd.Series(log_loss(pred["base"].to_numpy(), y) - log_loss(p, y), index=pred.index)
    r = {"predictions": len(pred), "accuracy": float(right.mean()), "always_above": float(y.mean()),
         "accuracy_by_year": {int(k): float(v) for k, v in right.groupby(year).mean().items()}}
    r["gain"], r["gain_ci"] = mean_ci(gain, pred["month"])
    confidence = (pred["p"] - 0.5).abs()
    for name, column in (("top10", "thr"), ("top5", "thr95")):
        chosen = confidence >= pred[column]
        t = {"selected": int(chosen.sum()), "by_year": {int(k): float(v) for k, v in right[chosen].groupby(year[chosen]).mean().items()},
             "recent": float(right[chosen & recent].mean()), "recent_selected": int((chosen & recent).sum()),
             "mean_move": float((np.where(pred.loc[chosen, "p"] > 0.5, 1.0, -1.0) * pred.loc[chosen, "fwd"]).mean())}
        t["hit"], t["hit_ci"] = mean_ci(right[chosen].astype(float), pred.loc[chosen, "month"])
        t["worst_year"] = min(t["by_year"].get(y, 0.0) for y in YEARS)
        r[name] = t
    return r


def show(name: str, r: dict) -> None:
    print(f"{name}: {r['predictions']} predictions, accuracy {r['accuracy']:.2%} (always above {r['always_above']:.2%}), "
          f"gain {r['gain'] * 1e4:+.2f}e-4 [{r['gain_ci'][0] * 1e4:+.2f} ~ {r['gain_ci'][1] * 1e4:+.2f}]")
    for k in ("top10", "top5"):
        t = r[k]
        print(f"   {k}: n={t['selected']}, hit {t['hit']:.2%} [{t['hit_ci'][0]:.2%} ~ {t['hit_ci'][1]:.2%}] | "
              + " ".join(f"{y} {v:.1%}" for y, v in t["by_year"].items())
              + f" | since 2025-11 {t['recent']:.2%} (n={t['recent_selected']}) | mean move in the predicted direction {t['mean_move']:+.3%}", flush=True)


def main() -> None:
    os.nice(15)
    reference = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
    new = h4_data.choose_v9_symbols()
    out: dict = {"reference_coins": len(reference), "new_symbols": new, "horizons": {}}
    passed = []
    for name, h in HORIZONS.items():
        data = dataset(h, reference, new)
        run = lambda d, f: walk_forward(d, h, TRAIN_START, SCORE_START, END, False, features=f, train_symbols=set(reference))
        relative, plain = run(add_relative(data, set(reference)), FEATURES + MARKET_FEATURES), run(data, FEATURES)
        on = lambda pred, coins: pred[pred["symbol"].isin(coins)]
        r = out["horizons"][name] = {"relative_new": score(on(relative, new)), "direction_new": score(on(plain, new)),
                                     "relative_reference": score(on(relative, reference)), "direction_reference": score(on(plain, reference))}
        rel, top = r["relative_new"], r["relative_new"]["top10"]
        r["checks"] = {"R1a gain CI low>0": bool(rel["gain_ci"][0] > 0),
                       "R1b accuracy>majority": bool(rel["accuracy"] > max(rel["always_above"], 1 - rel["always_above"])),
                       "R2a top10 hit CI low>50%": bool(top["hit_ci"][0] > 0.5),
                       "R2b top10 hit>accuracy": bool(top["hit"] > rel["accuracy"]),
                       "R3 top10 above 52% every year and lately": bool(top["worst_year"] > FLOOR and top["recent"] > FLOOR),
                       "R4 worst year above the direction model's": bool(top["worst_year"] > r["direction_new"]["top10"]["worst_year"])}
        show(f"[{name}] RELATIVE, new coins", rel)
        show(f"[{name}] direction (current model), new coins", r["direction_new"])
        show(f"[{name}] (reference, seen) relative, 59 coins", r["relative_reference"])
        show(f"[{name}] (reference, seen) direction, 59 coins", r["direction_reference"])
        print("   checks " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()) + "\n", flush=True)
        if all(v for k, v in r["checks"].items() if not k.startswith("R4")):
            passed.append(name)
    out.update(passed=passed, steadier=[n for n in passed if out["horizons"][n]["checks"]["R4 worst year above the direction model's"]],
               when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed R1-R3: {passed}; also steadier than the direction model (R4): {out['steadier']}")
    (h4_data.DATA_DIR / "direction_v9_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
