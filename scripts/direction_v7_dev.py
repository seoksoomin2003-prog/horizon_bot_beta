"""Runs the v7 tests: limit-order fills (M), positioning data (N) and three ways to a higher hit rate (H).
See docs/research/direction_preregistration_v7.md. Never touches the final-check period (from 2026-08-06).

  .venv/bin/python scripts/direction_v7_dev.py
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
from research.direction import (FEATURES, MARKET_FEATURES, POSITION_FEATURES, SIDE_COST, add_market, add_positioning,
                                fill_windows, maker_trades, walk_forward)
from scripts.direction_v6_dev import END, HORIZONS, ORIGINAL, SCORE_START, TRAIN_START, dataset, mean_ci, metrics, versus

GROUPS = {**POSITION_FEATURES, "XS": MARKET_FEATURES}
YEARS = range(2022, 2027)


def years_of(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["close_ms"], unit="ms").dt.year


def execution(pred: pd.DataFrame, windows: pd.DataFrame, horizon: int, new: list[str]) -> dict:
    """M: the top-10% predictions as limit-order trades, for both entry modes."""
    selected = pred[(pred["p"] - 0.5).abs() >= pred["thr"]]
    all_trades = maker_trades(selected, windows, horizon)
    out = {"selected": len(selected), "with_5m_bars": len(all_trades)}
    for mode, t in (("E1", all_trades[all_trades["entry_filled"]]), ("E2", all_trades)):
        year, group = years_of(t), t["symbol"].isin(new).map({True: "new", False: "original"})
        r = {"trades": len(t), "fill_share": len(t) / len(all_trades), "exit_fill_share": float(t["exit_filled"].mean()),
             "hit": float((t["move"] > 0).mean()), "net_2x": float(t["net_2x"].mean())}
        r["net_1x"], r["net_1x_ci"] = mean_ci(t["net_1x"], t["month"])
        r["net_1x_by_year"] = {int(k): float(v) for k, v in t["net_1x"].groupby(year).mean().items()}
        r["net_1x_by_group"] = {k: float(v) for k, v in t["net_1x"].groupby(group).mean().items()}
        monthly = t.groupby("month")["net_1x"].sum()
        r["months"], r["positive_months"] = len(monthly), int((monthly > 0).sum())
        r["checks"] = {"M1 enough trades and fills": bool(len(t) >= 1000 and (mode == "E2" or r["fill_share"] >= 0.30)),
                       "M2 net>0 with CI low>0 at 1x": bool(r["net_1x_ci"][0] > 0),
                       "M3 net>0 at 2x": bool(r["net_2x"] > 0),
                       "M4 positive in 4 of 5 years": bool(sum(r["net_1x_by_year"].get(y, 0) > 0 for y in YEARS) >= 4),
                       "M5 positive in both coin groups": bool(all(v > 0 for v in r["net_1x_by_group"].values()))}
        out[mode] = r
    filled = all_trades["entry_filled"]
    out["reference"] = {"market_both_sides_net_1x": float(all_trades["gross_market"].mean() - 2 * SIDE_COST),
                        "market_both_sides_net_2x": float(all_trades["gross_market"].mean() - 4 * SIDE_COST),
                        "gross_when_filled": float(all_trades.loc[filled, "gross_market"].mean()),
                        "gross_when_not_filled": float(all_trades.loc[~filled, "gross_market"].mean())}
    return out


def stricter(pred: pd.DataFrame) -> dict:
    """H1: the hit rate of the top 5% and top 2% against the top 10%, overall and year by year."""
    right, year, confidence = (pred["p"] > 0.5) == (pred["y"] > 0), years_of(pred), (pred["p"] - 0.5).abs()
    out = {}
    for name, column in (("top10", "thr"), ("top5", "thr95"), ("top2", "thr98")):
        chosen = confidence >= pred[column]
        r = {"selected": int(chosen.sum()), "share": float(chosen.mean()),
             "by_year": {int(k): float(v) for k, v in right[chosen].groupby(year[chosen]).mean().items()}}
        r["hit"], r["hit_ci"] = mean_ci(right[chosen].astype(float), pred.loc[chosen, "month"])
        out[name] = r
    for name in ("top5", "top2"):
        better_years = sum(out[name]["by_year"].get(y, 0) > out["top10"]["by_year"].get(y, 1) for y in YEARS)
        out[name]["better_years"] = better_years
        out[name]["pass"] = bool(out[name]["selected"] >= 2000 and out[name]["hit"] > out["top10"]["hit"] and better_years >= 4)
    return out


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["symbol", "close_ms"]).reset_index(drop=True)


def wide_transfer(data: pd.DataFrame, horizon: int, features: list[str], new: list[str], require: list[str] = ()) -> pd.DataFrame:
    """H2: every new coin predicted by a model that learned from the original 10 and the other half of the new coins."""
    halves = [sorted(new)[0::2], sorted(new)[1::2]]
    parts = []
    for learn, score in (halves, halves[::-1]):
        pred = walk_forward(data, horizon, TRAIN_START, SCORE_START, END, False, features=features, require=require,
                            train_symbols=set(ORIGINAL + learn))
        parts.append(pred[pred["symbol"].isin(score)])
    return aligned(pd.concat(parts))


def line(name: str, v: dict) -> str:
    return (f"{name}: gain {v['gain'] * 1e4:+.2f}e-4 [{v['gain_ci'][0] * 1e4:+.2f} ~ {v['gain_ci'][1] * 1e4:+.2f}], accuracy "
            f"{v['accuracy']:.2%} vs {v['accuracy_other']:.2%} on {v['rows']} rows -> {'PASS' if v['pass'] else 'FAIL'}")


def show_execution(name: str, e: dict) -> None:
    ref = e["reference"]
    print(f"{name}: {e['selected']} selected, {e['with_5m_bars']} with 5m bars; market both sides {ref['market_both_sides_net_1x']:+.3%} "
          f"(2x {ref['market_both_sides_net_2x']:+.3%}); gross when the limit filled {ref['gross_when_filled']:+.3%}, when not "
          f"{ref['gross_when_not_filled']:+.3%}")
    for mode in ("E1", "E2"):
        r = e[mode]
        print(f"  {mode}: {r['trades']} trades (entry fill {r['fill_share']:.1%}, exit fill {r['exit_fill_share']:.1%}), hit {r['hit']:.2%}, "
              f"net {r['net_1x']:+.3%} [{r['net_1x_ci'][0]:+.3%} ~ {r['net_1x_ci'][1]:+.3%}], 2x {r['net_2x']:+.3%}, positive months "
              f"{r['positive_months']}/{r['months']}")
        print("     by year " + ", ".join(f"{y} {v:+.3%}" for y, v in r["net_1x_by_year"].items())
              + " | " + ", ".join(f"{g} {v:+.3%}" for g, v in r["net_1x_by_group"].items())
              + " | " + ", ".join(f"{k.split()[0]} {'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()), flush=True)


def main() -> None:
    os.nice(15)
    new = h4_data.choose_v6_symbols()
    everything = ORIGINAL + new
    windows = pd.concat({s: fill_windows(h4_data.load_m5(s)) for s in everything}, names=["symbol", "T"])
    positioning = {s: h4_data.load_metrics(s) for s in everything}
    out: dict = {"horizons": {}}
    base_runs, candidates = {}, []
    for name, h in HORIZONS.items():
        data = add_market(add_positioning(dataset(h, everything), positioning))
        base = walk_forward(data, h, TRAIN_START, SCORE_START, END, False, train_symbols=set(ORIGINAL))
        on_new = aligned(base[base["symbol"].isin(new)])
        r = out["horizons"][name] = {"M": execution(base, windows, h, new), "H1": stricter(on_new), "features": {}}
        show_execution(f"[{name}] M limit orders, top 10%, 60 coins", r["M"])
        candidates += [(name, mode) for mode in ("E1", "E2") if all(r["M"][mode]["checks"].values())]
        print(f"[{name}] H1 new coins: " + "; ".join(
            f"{k} {v['hit']:.2%} [{v['hit_ci'][0]:.2%} ~ {v['hit_ci'][1]:.2%}] n={v['selected']}"
            + (f" better in {v['better_years']}/5 years -> {'PASS' if v['pass'] else 'FAIL'}" if "pass" in v else "")
            for k, v in r["H1"].items()), flush=True)
        for group, features in GROUPS.items():
            run = lambda f: walk_forward(data, h, TRAIN_START, SCORE_START, END, False, features=f, require=features,
                                         train_symbols=set(ORIGINAL))
            richer, plain = run(FEATURES + features), run(FEATURES)
            r["features"][group] = v = versus(aligned(richer[richer["symbol"].isin(new)]), aligned(plain[plain["symbol"].isin(new)]))
            print(line(f"[{name}] {group} features vs base, new coins", v), flush=True)
        r["H2"] = versus(wide_transfer(data, h, FEATURES, new), on_new)
        print(line(f"[{name}] H2 learning from 35 coins vs 10, new coins", r["H2"]) + "\n", flush=True)
        base_runs[name] = base

    passes = lambda pick: sum(pick(out["horizons"][n])["pass"] for n in HORIZONS) >= 2
    out["passed_groups"] = groups = [g for g in GROUPS if passes(lambda r: r["features"][g])]
    out["H2_pass"] = wide = passes(lambda r: r["H2"])
    out["H1_pass"] = {n: [k for k in ("top5", "top2") if out["horizons"][n]["H1"][k]["pass"]] for n in HORIZONS}
    print(f"feature groups passed: {groups or 'none'}; H2 (learn wider) {'PASS' if wide else 'FAIL'}; H1 {out['H1_pass']}")

    if groups or wide:      # the combined model: optimistic (chosen on this data), reported, never judged
        added = sum((GROUPS[g] for g in groups), [])
        out["combined"] = {"features": added, "wide": wide, "horizons": {}}
        for name, h in HORIZONS.items():
            data = add_market(add_positioning(dataset(h, everything), positioning))
            if wide:
                pred = wide_transfer(data, h, FEATURES + added, new)
            else:
                pred = walk_forward(data, h, TRAIN_START, SCORE_START, END, False, features=FEATURES + added, train_symbols=set(ORIGINAL))
                pred = aligned(pred[pred["symbol"].isin(new)])
            reference = base_runs[name][base_runs[name]["symbol"].isin(new)]
            c = out["combined"]["horizons"][name] = {
                "metrics": metrics(pred), "H1": stricter(pred), "M": execution(pred, windows, h, new),
                "base_M_same_coins": execution(reference, windows, h, new)}
            print(f"[{name}] combined model, new coins: accuracy {c['metrics']['accuracy']:.2%}; " + "; ".join(
                f"{k} {v['hit']:.2%} n={v['selected']}" for k, v in c["H1"].items()))
            show_execution(f"[{name}] combined, limit orders, new coins", c["M"])
            show_execution(f"[{name}] base, limit orders, new coins", c["base_M_same_coins"])

    best = max(candidates, key=lambda c: out["horizons"][c[0]]["M"][c[1]]["net_2x"]) if candidates else None
    out.update(candidates=candidates, chosen=best, when=time.strftime("%FT%TZ", time.gmtime()))
    print(f"M candidates: {candidates}\nverdict: " + (f"CANDIDATE {best} (final check only with the user's approval)" if best
                                                      else "NO_VERIFIED_DIRECTION_EDGE (final-check period left unused)"))
    (h4_data.DATA_DIR / "direction_v7_dev_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
