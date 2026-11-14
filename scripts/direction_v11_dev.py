"""Runs the v11 test: do the features' ranks among the coins beat the v9 model on the 46 v9 coins?
See docs/research/direction_preregistration_v11.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v11_dev.py
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
from research.direction import FEATURES, MARKET_FEATURES, RANK_FEATURES, add_ranks, add_relative, walk_forward
from scripts.direction_v6_dev import END, ORIGINAL, SCORE_START, TRAIN_START, mean_ci, versus
from scripts.direction_v9_dev import FLOOR, HORIZONS, YEARS, dataset, score, show

BASE = FEATURES + MARKET_FEATURES


def same_count(ranked: pd.DataFrame, plain: pd.DataFrame) -> dict:
    """Per month, as many of the rank model's surest predictions as the v9 model's top 10% holds."""
    counts = ((plain["p"] - 0.5).abs() >= plain["thr"]).groupby(plain["month"]).sum()
    confidence = (ranked["p"] - 0.5).abs()
    chosen = confidence.groupby(ranked["month"]).rank(method="first", ascending=False) <= ranked["month"].map(counts)
    right = ((ranked["p"] > 0.5) == (ranked["y"] > 0))[chosen]
    picked = ranked[chosen]
    year = pd.to_datetime(picked["close_ms"], unit="ms").dt.year
    recent = picked["close_ms"] >= h4_data.DEV_END_MS
    r = {"selected": int(chosen.sum()), "by_year": {int(k): float(v) for k, v in right.groupby(year).mean().items()},
         "recent": float(right[recent].mean()), "above_calls": float((picked["p"] > 0.5).mean())}
    r["hit"], r["hit_ci"] = mean_ci(right.astype(float), picked["month"])
    r["worst_year"] = min(r["by_year"].get(y, 0.0) for y in YEARS)
    return r


def main() -> None:
    os.nice(15)
    reference = [s for s in ORIGINAL + h4_data.choose_v6_symbols() if s != "BTCDOMUSDT"]
    new = h4_data.choose_v9_symbols()
    out: dict = {"reference_coins": len(reference), "new_coins": len(new), "horizons": {}}
    passed = []
    for name, h in HORIZONS.items():
        data = add_ranks(add_relative(dataset(h, reference, new), set(reference)), set(reference))
        run = lambda f: walk_forward(data, h, TRAIN_START, SCORE_START, END, False, features=f, require=RANK_FEATURES,
                                     train_symbols=set(reference))
        ranked, plain = run(BASE + RANK_FEATURES), run(BASE)
        on = lambda pred, coins: pred[pred["symbol"].isin(coins)].reset_index(drop=True)
        recent = lambda pred: pred[pred["close_ms"] >= h4_data.DEV_END_MS].reset_index(drop=True)
        r = out["horizons"][name] = {
            "ranked_new": score(on(ranked, new)), "plain_new": score(on(plain, new)),
            "ranked_reference": score(on(ranked, reference)), "plain_reference": score(on(plain, reference)),
            "gain_over_plain": versus(on(ranked, new), on(plain, new)),
            "gain_over_plain_recent": versus(recent(on(ranked, new)), recent(on(plain, new))),
            "same_count_new": same_count(on(ranked, new), on(plain, new)),
            "same_count_reference": same_count(on(ranked, reference), on(plain, reference))}
        same, base, g, late = r["same_count_new"], r["plain_new"]["top10"], r["gain_over_plain"], r["gain_over_plain_recent"]
        r["checks"] = {"X1 log loss better, CI low>0": bool(g["gain_ci"][0] > 0),
                       "X2 same-count hit above the v9 top 10%": bool(same["hit"] > base["hit"]),
                       "X3 same-count above 52% every year and lately": bool(same["worst_year"] > FLOOR and same["recent"] > FLOOR),
                       "X4 log loss better lately": bool(late["gain"] > 0)}
        show(f"[{name}] RANK model, new coins", r["ranked_new"])
        show(f"[{name}] v9 model, new coins (same rows)", r["plain_new"])
        show(f"[{name}] (seen) RANK model, 59 coins", r["ranked_reference"])
        show(f"[{name}] (seen) v9 model, 59 coins", r["plain_reference"])
        for label, s in (("new coins", same), ("(seen) 59 coins", r["same_count_reference"])):
            print(f"   same count, {label}: n={s['selected']}, hit {s['hit']:.2%} [{s['hit_ci'][0]:.2%} ~ {s['hit_ci'][1]:.2%}] | "
                  + " ".join(f"{y} {v:.1%}" for y, v in s["by_year"].items()) + f" | since 2025-11 {s['recent']:.2%} | above-calls {s['above_calls']:.0%}")
        print(f"   log-loss gain over v9 {g['gain'] * 1e4:+.2f}e-4 [{g['gain_ci'][0] * 1e4:+.2f} ~ {g['gain_ci'][1] * 1e4:+.2f}], "
              f"since 2025-11 {late['gain'] * 1e4:+.2f}e-4 [{late['gain_ci'][0] * 1e4:+.2f} ~ {late['gain_ci'][1] * 1e4:+.2f}]")
        print("   checks " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()) + "\n", flush=True)
        if all(r["checks"].values()):
            passed.append(name)
    out.update(passed=passed, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"passed X1-X4: {passed}\nverdict: " + ("switch the live relative model to the rank model for " + ", ".join(passed) if passed
                                                  else "NO CHANGE (rank features do not beat the v9 model)"))
    (h4_data.DATA_DIR / "direction_v11_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
