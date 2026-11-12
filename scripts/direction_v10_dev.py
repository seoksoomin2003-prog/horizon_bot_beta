"""Runs the v10 test: do weights that change with the market's state beat the v9 model on the 46 v9 coins?
See docs/research/direction_preregistration_v10.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v10_dev.py
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
from research.direction import FEATURES, MARKET_FEATURES, REGIMES, add_regime, add_relative, walk_forward
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START, versus
from scripts.direction_v9_dev import FLOOR, HORIZONS, dataset, score, show


def compare(data: pd.DataFrame, horizon: int, features: list[str], reference: list[str], new: list[str]) -> dict:
    """The state model against the same model without states, on the rows where the state is known."""
    data, names = add_regime(data, set(reference), features)
    run = lambda f: walk_forward(data, horizon, TRAIN_START, SCORE_START, END, False, features=f, require=names,
                                 train_symbols=set(reference))
    state, plain = run(features + names), run(features)
    on = lambda pred, coins: pred[pred["symbol"].isin(coins)].reset_index(drop=True)
    recent = lambda pred: pred[pred["close_ms"] >= h4_data.DEV_END_MS].reset_index(drop=True)
    r = {"state_new": score(on(state, new)), "plain_new": score(on(plain, new)),
         "state_reference": score(on(state, reference)), "plain_reference": score(on(plain, reference)),
         "gain_over_plain": versus(on(state, new), on(plain, new)),
         "gain_over_plain_recent": versus(recent(on(state, new)), recent(on(plain, new)))}
    s = on(state, new)
    chosen = (s["p"] - 0.5).abs() >= s["thr"]
    right = (s["p"] > 0.5) == (s["y"] > 0)
    r["by_state"] = {name: {"rows": int((s["regime"] == k).sum()), "top10": int((chosen & (s["regime"] == k)).sum()),
                            "top10_hit": float(right[chosen & (s["regime"] == k)].mean())} for k, name in enumerate(REGIMES)}
    top, base = r["state_new"]["top10"], r["plain_new"]["top10"]
    r["checks"] = {"W1 log loss better, CI low>0": bool(r["gain_over_plain"]["gain_ci"][0] > 0),
                   "W2 top10 hit above plain and CI low>50%": bool(top["hit"] > base["hit"] and top["hit_ci"][0] > 0.5),
                   "W3 top10 above 52% every year and lately": bool(top["worst_year"] > FLOOR and top["recent"] > FLOOR),
                   "W4 log loss better lately": bool(r["gain_over_plain_recent"]["gain"] > 0)}
    return r


def report(title: str, r: dict) -> None:
    g, recent = r["gain_over_plain"], r["gain_over_plain_recent"]
    show(f"{title} STATE weights, new coins", r["state_new"])
    show(f"{title} plain, new coins (same rows)", r["plain_new"])
    show(f"{title} (seen) STATE weights, 59 coins", r["state_reference"])
    show(f"{title} (seen) plain, 59 coins", r["plain_reference"])
    print(f"   log-loss gain over plain {g['gain'] * 1e4:+.2f}e-4 [{g['gain_ci'][0] * 1e4:+.2f} ~ {g['gain_ci'][1] * 1e4:+.2f}], "
          f"since 2025-11 {recent['gain'] * 1e4:+.2f}e-4 [{recent['gain_ci'][0] * 1e4:+.2f} ~ {recent['gain_ci'][1] * 1e4:+.2f}]")
    print("   by state " + " | ".join(f"{k}: rows {v['rows']}, top10 n={v['top10']} hit {v['top10_hit']:.1%}" for k, v in r["by_state"].items()))
    print("   checks " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()) + "\n", flush=True)


def main() -> None:
    os.nice(15)
    reference = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
    new = h4_data.choose_v9_symbols()
    out: dict = {"reference_coins": len(reference), "new_coins": len(new), "horizons": {}}
    passed = []
    for name, h in HORIZONS.items():
        data = dataset(h, reference, new)
        r = out["horizons"][name] = {"relative": compare(add_relative(data, set(reference)), h, FEATURES + MARKET_FEATURES, reference, new)}
        report(f"[{name}] RELATIVE:", r["relative"])
        r["direction_reference_only"] = compare(data, h, FEATURES, reference, new)
        report(f"[{name}] direction (reference only, not judged):", r["direction_reference_only"])
        if all(r["relative"]["checks"].values()):
            passed.append(name)
    out.update(passed=passed, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed W1-W4: {passed}\nverdict: " + ("add state predictions to the live predictor for " + ", ".join(passed) if passed
                                                  else "NO CHANGE (state weights do not beat the v9 model)"))
    (h4_data.DATA_DIR / "direction_v10_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
